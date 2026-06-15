"""
Multi-judge script for SAKURA benchmark evaluation.
Supports: gpt-4o, gpt-4o-mini, gemini-2.5-flash-lite, gemini-3.1-flash-lite
Concurrent API calls via ThreadPoolExecutor; resume-safe.
"""
import os
import re
import json
import argparse
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

SUPPORTED_JUDGES = [
    "gpt-4o",
    "gpt-4o-mini",
    "gemini-2.5-flash-lite",
    "gemini-3.1-flash-lite",
]

# RPM limits per judge (PAYG tier)
JUDGE_RPM = {
    "gpt-4o":                  30,   # Tier 1: ~30K TPM / ~1K tok/req
    "gpt-4o-mini":             200,  # Tier 1: ~200K TPM / ~1K tok/req
    "gemini-2.5-flash-lite":   200,  # PAYG: ~150-300 RPM
    "gemini-3.1-flash-lite":   200,  # PAYG: ~150-300 RPM
}

# Default workers per judge
JUDGE_WORKERS = {
    "gpt-4o":                  5,
    "gpt-4o-mini":             20,
    "gemini-2.5-flash-lite":   20,
    "gemini-3.1-flash-lite":   20,
}

JUDGE_MODEL_IDS = {
    "gpt-4o": "gpt-4o-2024-11-20",
    "gpt-4o-mini": "gpt-4o-mini",
    "gemini-2.5-flash-lite": "gemini-2.5-flash-lite",
    "gemini-3.1-flash-lite": "gemini-3.1-flash-lite",
}

USER_PROMPT_TEMPLATE = """
    You will be given a question with list of possible options, a ground truth answer and a model generated response. Determine whether the model generated response is correct based on the following criteria:
    1. Since there is one and only one corect answer, it should be judged incorrect if the model do not choose any option from the option list or it choose more than one option.
    2. If the model choose one option from the option list, it should be judged correct if the chosen option aligns with the ground truth answer, otherwise it should be judged incorrect.
    3. Read the question, options, ground truth answer and model generated response carefully before making a decision.

    Considering the following examples:
    Question: What is the capital of France? (a) Paris (b) London (c) Berlin (d) Madrid
    Ground truth answer: (a) Paris
    If the model generated response is: "The capital of France is Tokyo.", it should be judged incorrect since it does not choose any option from the option list.
    If the model generated response is: "The capital of France is Paris and London.", it should be judged incorrect since it chooses more than one option from the option list.
    If the model generated response is: "The capital of France is London.", it should be judged incorrect since it chooses one option from the option list but the chosen option does not align with the ground truth answer.
    If the model generated response is: "The capital of France is Paris.", it should be judged correct since it chooses one option from the option list and the chosen option aligns with the ground truth answer.
    Another Question: What is the underlying emotion of the speaker? (a) Happy (b) Sad (c) Angry (d) Neutral
    Ground truth answer: (a) Happy
    If the model generated response is: "The speaker is happy.", it should be judged correct since it chooses one option from the option list and the chosen option aligns with the ground truth answer.
    If the model generated response is: "The speaker expresses happiness.", it should be judged correct since "happiness" aligns with the ground truth answer "happy", and they are just different part of speech of the same word.
    If the model generated response is: "Happiness," it should be judged correct since it is also a valid derivative of the ground truth answer "happy".

    Now here is the question and the model generated response for you to judge:
    Question: [QUESTION]
    Ground truth answer: [GROUND_TRUTH_ANSWER]
    Model generated response: [MODEL_GENERATED_RESPONSE]

    Carefully make your decision based on the above criteria. Return your judgement with the following format:
    Explanation: <Your explanation on your judgement>
    Judgement: <Your judgement, either "correct" or "incorrect">
    """

SYSTEM_PROMPT = (
    "You are a good judge. You will be given a question with list of possible options, "
    "a ground truth answer and a model generated response. "
    "You have to determine whether the model generated answer is correct."
)


class RateLimiter:
    """Slot-based rate limiter: each caller gets a reserved send-slot.
    Sleep is outside the lock so concurrent threads spread over time correctly."""
    def __init__(self, rpm: int):
        self.interval = 60.0 / rpm
        self._lock = threading.Lock()
        self._next = 0.0  # next available send slot

    def acquire(self):
        with self._lock:
            now = time.monotonic()
            send_at = max(now, self._next)
            self._next = send_at + self.interval  # reserve this slot
        wait = send_at - time.monotonic()
        if wait > 0:
            time.sleep(wait)


def _parse_retry_after(exc_str: str, default: float = 65.0) -> float:
    m = re.search(r"retry in (\d+(?:\.\d+)?)\s*s", exc_str)
    return float(m.group(1)) + 2 if m else default


def build_user_prompt(question, ground_truth, response):
    return (
        USER_PROMPT_TEMPLATE
        .replace("[QUESTION]", question)
        .replace("[GROUND_TRUTH_ANSWER]", ground_truth)
        .replace("[MODEL_GENERATED_RESPONSE]", response)
    )


def query_openai(client, user_prompt, model_id, temperature=0.0, top_p=0.9):
    completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        model=model_id,
        temperature=temperature,
        top_p=top_p,
    )
    return completion.choices[0].message.content


def query_gemini(client, user_prompt, model_id):
    from google import genai as google_genai
    full_prompt = SYSTEM_PROMPT + "\n\n" + user_prompt
    response = client.models.generate_content(
        model=model_id,
        contents=full_prompt,
    )
    return response.text


def build_client(judge: str):
    if judge in ("gpt-4o", "gpt-4o-mini"):
        from openai import OpenAI
        return OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    else:
        from google import genai as google_genai
        return google_genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def query_judge(client, judge: str, user_prompt: str, rate_limiter: RateLimiter,
                retries: int = 5) -> str:
    model_id = JUDGE_MODEL_IDS[judge]
    for attempt in range(retries):
        rate_limiter.acquire()
        try:
            if judge in ("gpt-4o", "gpt-4o-mini"):
                return query_openai(client, user_prompt, model_id)
            else:
                return query_gemini(client, user_prompt, model_id)
        except Exception as e:
            msg = str(e)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                wait = _parse_retry_after(msg)
                print(f"\n[{judge}] Rate limited, sleeping {wait:.0f}s (attempt {attempt+1}/{retries})")
                time.sleep(wait)
            elif attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise e
    raise RuntimeError(f"Failed after {retries} retries")


def extract_judgement(text: str) -> dict:
    pattern = r"Explanation: (.*?)\nJudgement: (.*?)(?:\n\n|$)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return {"Explanation": match.group(1), "Judgement": match.group(2)}
    return {"Explanation": "No extracted explanation", "Judgement": "No extracted judgement"}


def judge_one(wav_file, item, client, judge, rate_limiter):
    user_prompt = build_user_prompt(item["instruction"], item["label"], item["response"])
    raw = query_judge(client, judge, user_prompt, rate_limiter)
    result = extract_judgement(raw)
    result["instruction"] = item["instruction"]
    result["model_response"] = item["response"]
    result["label"] = item["label"]
    return wav_file, result


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, help="Input pred JSON file")
    parser.add_argument("--output_dir", "-o", required=True, help="Output directory")
    parser.add_argument(
        "--judge", "-j", required=True, choices=SUPPORTED_JUDGES,
        help="Judge model to use"
    )
    parser.add_argument("--workers", "-w", type=int, default=None,
                        help="Concurrent API requests (default: per-model)")
    return parser.parse_args()


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.input, "r") as f:
        model_responses = json.load(f)

    judge = args.judge
    model_id = JUDGE_MODEL_IDS[judge]
    workers = args.workers if args.workers is not None else JUDGE_WORKERS[judge]
    rate_limiter = RateLimiter(JUDGE_RPM[judge])

    input_stem = os.path.basename(args.input).replace(".json", "")
    out_path = os.path.join(args.output_dir, f"{input_stem}_judgements.json")

    # Resume: load existing results, skip errors so they get retried
    if os.path.exists(out_path):
        with open(out_path, "r") as f:
            judgements = json.load(f)
        done = {k for k, v in judgements["results"].items()
                if v.get("Judgement", "").lower() not in ("error", "")}
        print(f"[{judge}] Resuming: {len(done)} valid done "
              f"({len(judgements['results'])-len(done)} errors will retry)")
        judgements["results"] = {k: v for k, v in judgements["results"].items() if k in done}
    else:
        judgements = {
            "judge": judge,
            "judge_model": model_id,
            "temperature": 0.0,
            "results": {},
        }
        done = set()

    pending = {k: v for k, v in model_responses["results"].items() if k not in done}
    print(f"[{judge}] workers={workers}, RPM limit={JUDGE_RPM[judge]}")
    print(f"[{judge}] {input_stem}: {len(pending)} pending / {len(model_responses['results'])} total")

    if not pending:
        print(f"[{judge}] Already complete: {out_path}")
        return

    client = build_client(judge)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(judge_one, wav, item, client, judge, rate_limiter): wav
            for wav, item in pending.items()
        }
        with tqdm(total=len(futures), desc=f"[{judge}] {input_stem}") as pbar:
            for future in as_completed(futures):
                try:
                    wav_file, result = future.result()
                    judgements["results"][wav_file] = result
                except Exception as e:
                    wav = futures[future]
                    print(f"\n[{judge}] ERROR on {wav}: {e}")
                    judgements["results"][wav] = {
                        "Explanation": f"Error: {e}",
                        "Judgement": "error",
                    }
                pbar.update(1)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(judgements, f, ensure_ascii=False, indent=4)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
