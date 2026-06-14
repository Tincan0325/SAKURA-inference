"""
Kimi-Audio-7B-Instruct inference on SAKURA benchmark.
Greedy text decoding (text_temperature=0.0), load_detokenizer=False to save VRAM.
Resume-safe (skips already-done items).
"""
import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [Kimi] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

TRACKS = ["Gender", "Language", "Emotion", "Animal"]
SUBTRACKS = ["single", "multi"]
MODEL_PATH = "moonshotai/Kimi-Audio-7B-Instruct"
MODEL_NAME = "kimi"

SAMPLING_PARAMS = {
    "text_temperature": 0.0,   # greedy
    "text_top_k": 5,
    "text_repetition_penalty": 1.0,
    "text_repetition_window_size": 16,
    "audio_temperature": 0.8,
    "audio_top_k": 10,
    "audio_repetition_penalty": 1.0,
    "audio_repetition_window_size": 64,
}


def load_model():
    from kimia_infer.api.kimia import KimiAudio
    log.info(f"Loading {MODEL_PATH} (load_detokenizer=False) ...")
    model = KimiAudio(model_path=MODEL_PATH, load_detokenizer=False)
    log.info("Model loaded.")
    return model


def infer_single(model, audio_path, instruction):
    messages = [
        {"role": "user", "message_type": "text", "content": instruction},
        {"role": "user", "message_type": "audio", "content": audio_path},
    ]
    _, text_out = model.generate(messages, **SAMPLING_PARAMS, output_type="text")
    return text_out.strip()


def run_track(model, track, subtrack, sakura_dir, pred_dir):
    base = Path(sakura_dir) / "data" / track
    instr_col = f"{subtrack}_instruction"
    ans_col = f"{subtrack}_answer"

    rows = list(csv.DictReader(open(base / "metadata.csv")))
    out_path = Path(pred_dir) / f"{MODEL_NAME}_{track}_{subtrack}.json"

    if out_path.exists():
        data = json.loads(out_path.read_text())
    else:
        data = {"attribute": track, "type": subtrack, "results": {}}

    done = len(data["results"])
    log.info(f"[{track}/{subtrack}] {done} done, {len(rows) - done} remaining")

    for i, r in enumerate(rows):
        audio_path = str(base / "audio" / r["file"])
        if audio_path in data["results"]:
            continue

        try:
            resp = infer_single(model, audio_path, r[instr_col])
        except Exception as e:
            log.error(f"Failed on {r['file']}: {e}")
            resp = ""

        data["results"][audio_path] = {
            "instruction": r[instr_col],
            "response": resp,
            "label": r[ans_col],
        }
        out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))

        if (i + 1) % 50 == 0:
            log.info(f"[{track}/{subtrack}] {i+1}/{len(rows)}  last: {resp[:60]!r}")

    log.info(f"[{track}/{subtrack}] COMPLETE → {out_path}")
    return str(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sakura-dir", default="SAKURA")
    parser.add_argument("--pred-dir", default="preds")
    parser.add_argument("--tracks", nargs="+", default=TRACKS, choices=TRACKS)
    parser.add_argument("--subtracks", nargs="+", default=SUBTRACKS, choices=SUBTRACKS)
    args = parser.parse_args()

    os.makedirs(args.pred_dir, exist_ok=True)

    model = load_model()

    for track in args.tracks:
        for subtrack in args.subtracks:
            run_track(model, track, subtrack, args.sakura_dir, args.pred_dir)

    log.info("All Kimi inference done.")


if __name__ == "__main__":
    main()
