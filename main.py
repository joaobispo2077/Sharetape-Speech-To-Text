# main.py
import argparse
import logging
import os
import uuid
from typing import Callable, Optional

from vosk import Model, SetLogLevel
from sharetape import Sharetape

# Optional: tqdm-based progress bars if available
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def make_progress_cb():
    """
    Returns a callback with signature:
        cb(desc: str, current: int, total: int, unit: str = "")
    It will render a tqdm bar if tqdm is installed; otherwise prints %.
    """
    if tqdm:
        bars = {}  # one bar per 'desc'

        def cb(desc: str, current: int, total: int, unit: str = ""):
            bar = bars.get(desc)
            if bar is None:
                bars[desc] = tqdm(total=total, desc=desc, unit=unit or "it", leave=True)
                bar = bars[desc]
            bar.n = current
            bar.refresh()
            if current >= total:
                bar.close()
                bars.pop(desc, None)

        return cb
    else:
        def cb(desc: str, current: int, total: int, unit: str = ""):
            pct = (current / total * 100) if total else 0.0
            print(f"\r{desc}: {pct:6.2f}% ({current}/{total} {unit})", end="", flush=True)
            if current >= total:
                print()
        return cb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--video", type=str, required=False, default="")
    parser.add_argument("-a", "--audio", type=str, required=False, default="")
    args = parser.parse_args()

    if not (args.video or args.audio):
        parser.error("No action requested, add --video or --audio")
    elif args.video and args.audio:
        parser.error("Only select one action --video or --audio")

    SetLogLevel(-1)
    print("Loading Vosk model (first time can take a bit)...")
    # model = Model(model_path="vosk-model-en-us-0.42-gigaspeech")
    model = Model(model_path="vosk-model-pt-fb-v0.1.1-20220516_2113")
    logging.info("sp2t setup")

    video_id = str(uuid.uuid4())
    os.makedirs(f"{video_id}")

    if args.audio != "":
        audio = args.audio
    else:
        audio = f"{video_id}/audio.wav"

    progress_cb = make_progress_cb()

    shartape = Sharetape(
        args.video,
        audio,
        f"{video_id}/mono_audio.wav",
        f"{video_id}/transcript.txt",
        f"{video_id}/words.json",
        f"{video_id}/captions.srt",
        model,
        progress_cb=progress_cb,  # <-- pass the callback you build in main.py
    )

    shartape.extract_transcript()


if __name__ == "__main__":
    main()
