"""
Audio Flamingo 3 (nvidia/audio-flamingo-3-hf) inference on SAKURA benchmark.
Greedy decoding, bf16, batch=8, resume-safe (skips already-done items).
"""
import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path

import torch
from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [AF3] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

TRACKS = ["Gender", "Language", "Emotion", "Animal"]
SUBTRACKS = ["single", "multi"]
MODEL_ID = "nvidia/audio-flamingo-3-hf"
MODEL_NAME = "af3"
BATCH_SIZE = 8


def load_model():
    log.info(f"Loading {MODEL_ID} in bf16 ...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    log.info(f"Model loaded on: {next(model.parameters()).device}")
    return processor, model


def infer_batch(processor, model, items):
    """items: list of (audio_path, instruction)  →  list of response str"""
    convs = [
        [{"role": "user", "content": [
            {"type": "text", "text": instr},
            {"type": "audio", "path": audio},
        ]}]
        for audio, instr in items
    ]
    inputs = processor.apply_chat_template(
        convs, tokenize=True, add_generation_prompt=True, return_dict=True
    ).to(next(model.parameters()).device, dtype=model.dtype)

    with torch.no_grad():
        out = model.generate(**inputs, do_sample=False, max_new_tokens=128)

    responses = processor.batch_decode(
        out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
    )
    return [r.strip() for r in responses]


def run_track(processor, model, track, subtrack, sakura_dir, pred_dir):
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

    for i in range(0, len(pending), BATCH_SIZE):
        batch = pending[i : i + BATCH_SIZE]
        items = [(str(base / "audio" / r["file"]), r[instr_col]) for r in batch]
        try:
            responses = infer_batch(processor, model, items)
        except Exception as e:
            log.error(f"Batch {i}–{i+len(batch)} failed: {e}; falling back to single")
            responses = []
            for audio, instr in items:
                try:
                    resp = infer_batch(processor, model, [(audio, instr)])[0]
                except Exception as e2:
                    log.error(f"Single inference failed for {audio}: {e2}")
                    resp = ""
                responses.append(resp)

        for r, resp in zip(batch, responses):
            key = str(base / "audio" / r["file"])
            data["results"][key] = {
                "instruction": r[instr_col],
                "response": resp,
                "label": r[ans_col],
            }
        out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))

        completed = done + i + len(batch)
        log.info(f"[{track}/{subtrack}] {completed}/{len(rows)} done  last: {responses[-1][:60]!r}")

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

    processor, model = load_model()

    for track in args.tracks:
        for subtrack in args.subtracks:
            run_track(processor, model, track, subtrack, args.sakura_dir, args.pred_dir)

    log.info("All AF3 inference done.")


if __name__ == "__main__":
    main()
