import datetime
import json
import logging
import wave

import moviepy.editor as mp
import scipy.io.wavfile as wav
import srt
from vosk import KaldiRecognizer


class Sharetape:
    def __init__(
        self,
        video,
        audio,
        mono_audio,
        transcript,
        words,
        subtitles,
        model,
        progress_cb=None,  # <-- NEW: optional progress callback
    ) -> None:
        self.video = video
        self.audio = audio
        self.mono_audio = mono_audio
        self.transcript = transcript
        self.words = words
        self.subtitles = subtitles
        self.model = model
        self.progress_cb = progress_cb

    # small helper to safely emit progress
    def _progress(self, desc: str, current: int, total: int, unit: str = ""):
        if self.progress_cb:
            try:
                self.progress_cb(desc, int(current), int(total), unit)
            except TypeError:
                # if someone's callback only takes 3 args
                self.progress_cb(desc, int(current), int(total))

    def load_data(self):
        try:
            with open(self.words, "r") as json_file:
                words = json.load(json_file)
        except Exception:
            words = []
        return words

    def save_data(self, data):
        with open(self.words, "w") as json_file:
            json.dump(data, json_file)

    def extract_transcript(self):
        # A) extract audio from video (if provided)
        if self.video != "":
            self._progress("Extracting audio", 0, 1, "steps")
            my_clip = mp.VideoFileClip(self.video)
            if my_clip.audio:
                # keep moviepy quiet; we just mark start/end to avoid messy mixed bars
                my_clip.audio.write_audiofile(self.audio, verbose=False, logger=None)
            self._progress("Extracting audio", 1, 1, "steps")

        # B+C) transcribe audio file
        transcript, words, subtitle = self.handle_speech_2_text()

        with open(self.transcript, "w+", encoding="utf8") as fil:
            fil.write(transcript)

        self.save_data(words)

        with open(self.subtitles, "w+", encoding="utf8") as f:
            f.writelines(subtitle)

    def handle_speech_2_text(self):
        # --- Convert to mono (robust to mono or stereo input) ---
        self._progress("Preparing audio", 0, 1, "steps")
        sample_rate, data = wav.read(self.audio)  # numpy array
        # If it's already mono (shape: [n]), keep as is; if stereo (shape: [n,2]), average channels
        if getattr(data, "ndim", 1) == 1:
            mono_data = data.astype("int16", copy=False)
        else:
            # use wider int to avoid overflow, then cast back
            left_channel = data[:, 0].astype("int32", copy=False)
            right_channel = data[:, 1].astype("int32", copy=False)
            mono_data = ((left_channel + right_channel) // 2).astype("int16", copy=False)

        wav.write(self.mono_audio, sample_rate, mono_data)
        self._progress("Preparing audio", 1, 1, "steps")

        # --- Open mono wav for Vosk ---
        wf = wave.open(self.mono_audio, "rb")
        if (
            wf.getnchannels() != 1
            or wf.getsampwidth() != 2
            or wf.getcomptype() != "NONE"
        ):
            logging.error("Audio file must be WAV format mono PCM.")
            return "", "", ""

        rec = KaldiRecognizer(self.model, wf.getframerate())
        rec.SetWords(True)
        rec.SetPartialWords(True)

        # --- Transcription loop with real progress ---
        results = []
        subs = []

        total_frames = wf.getnframes()
        processed_frames = 0
        chunk_frames = 4000

        self._progress("Transcribing", 0, total_frames, "frames")
        while True:
            data_bytes = wf.readframes(chunk_frames)
            if not data_bytes:
                break

            rec.AcceptWaveform(data_bytes)

            frames_read = len(data_bytes) // (wf.getsampwidth() * wf.getnchannels())
            processed_frames = min(processed_frames + frames_read, total_frames)
            self._progress("Transcribing", processed_frames, total_frames, "frames")

            # Collect full results as Vosk emits them
            # (AcceptWaveform returns True at segment boundaries)
            # We can't know that here; we’ll pull after loop too.
            if rec.AcceptWaveform(b""):
                results.append(rec.Result())

        # Final result
        results.append(rec.FinalResult())
        self._progress("Transcribing", total_frames, total_frames, "frames")

        # --- Build transcript + SRT ---
        WORDS_PER_LINE = 14
        total_lines = []
        total_words = []

        for res in results:
            jres = json.loads(res)
            if "result" not in jres:
                continue
            words = jres["result"]
            total_words.extend(words)

            for j in range(0, len(words), WORDS_PER_LINE):
                line = words[j : j + WORDS_PER_LINE]
                s = srt.Subtitle(
                    index=len(subs),
                    content=" ".join([l["word"] for l in line]),
                    start=datetime.timedelta(seconds=line[0]["start"]),
                    end=datetime.timedelta(seconds=line[-1]["end"]),
                )
                total_lines.append(s.content)
                subs.append(s)

        transcript = " ".join(total_lines)
        subtitle = srt.compose(subs)

        return (transcript, total_words, subtitle)
