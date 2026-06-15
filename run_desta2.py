"""
DeSTA2 (DeSTA-ntu/DeSTA2-8B-beta) inference on SAKURA benchmark.
Greedy decoding, single-sample (no batch API), resume-safe.
"""
import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [DeSTA2] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

TRACKS = ["Gender", "Language", "Emotion", "Animal"]
SUBTRACKS = ["single", "multi"]
MODEL_ID = "DeSTA-ntu/DeSTA2-8B-beta"
MODEL_NAME = "desta2"


def load_model(hf_token: str):
    log.info(f"Loading {MODEL_ID} ...")
    # Install path: DeSTA2/desta package must be on sys.path
    desta_path = Path(__file__).parent / "DeSTA2"
    if str(desta_path) not in sys.path:
        sys.path.insert(0, str(desta_path))

    from desta import DestaModel

    model = DestaModel.from_pretrained(MODEL_ID, token=hf_token)
    model.to("cuda")
    model.eval()
    log.info(f"Model loaded on: {next(model.parameters()).device}")
    return model


def infer_single(model, audio_path: str, instruction: str) -> str:
    messages = [
        {"role": "system", "content": "Focus on the input audio."},
        {"role": "audio", "content": audio_path},
        {"role": "user", "content": instruction},
    ]
    with torch.no_grad():
        generated_ids = model.chat(
            messages, max_new_tokens=128, do_sample=False
        )
    response = model.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response.strip()


def run_track(model, track: str, subtrack: str, sakura_dir: str, pred_dir: str):
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
    pending = [r for r in rows if str(base / "audio" / r["file"]) not in data["results"]]
    log.info(f"[{track}/{subtrack}] {done} done, {len(pending)} remaining")

    for i, row in enumerate(pending):
        audio_path = str(base / "audio" / row["file"])
        instruction = row[instr_col]
        label = row[ans_col]
        try:
            response = infer_single(model, audio_path, instruction)
        except Exception as e:
            log.error(f"Inference failed for {audio_path}: {e}")
            response = ""

        data["results"][audio_path] = {
            "instruction": instruction,
            "response": response,
            "label": label,
        }

        if (i + 1) % 10 == 0 or (i + 1) == len(pending):
            out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))

        completed = done + i + 1
        log.info(f"[{track}/{subtrack}] {completed}/{len(rows)}  last: {response[:60]!r}")

    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    log.info(f"[{track}/{subtrack}] COMPLETE → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sakura-dir", default="SAKURA")
    parser.add_argument("--pred-dir", default="preds")
    parser.add_argument("--tracks", nargs="+", default=TRACKS, choices=TRACKS)
    parser.add_argument("--subtracks", nargs="+", default=SUBTRACKS, choices=SUBTRACKS)
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""))
    args = parser.parse_args()

    if not args.hf_token:
        log.error("HF_TOKEN not set. Pass --hf-token or set HF_TOKEN env var.")
        sys.exit(1)

    os.makedirs(args.pred_dir, exist_ok=True)
    model = load_model(args.hf_token)

    for track in args.tracks:
        for subtrack in args.subtracks:
            run_track(model, track, subtrack, args.sakura_dir, args.pred_dir)

    log.info("All DeSTA2 inference done.")


if __name__ == "__main__":
    main()
