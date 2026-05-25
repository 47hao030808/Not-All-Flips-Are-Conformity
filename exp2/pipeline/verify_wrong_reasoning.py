import csv
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
BASE_DIR = ROOT_DIR / "input"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

QUESTION_FILE = BASE_DIR / "correct_to_wrong_reasoning_questions.csv"
OUT_DIR = BASE_DIR / "out"
INPUT_FILE = OUT_DIR / "wrong_reasoning.jsonl"
VERIFIED_FILE = OUT_DIR / "wrong_reasoning_verified.jsonl"
REPORT_FILE = OUT_DIR / "wrong_reasoning_verification_report.jsonl"

API_URL = os.getenv("GENERATION_API_URL")
API_KEY = os.getenv("GENERATION_API_KEY")
MODEL = os.getenv("GENERATION_MODEL")

PROMPT_TEMPLATE = """Evaluate the following generated reasoning for a multiple-choice experiment.

Specified wrong answer:
{wrong_answer}

Correct answer:
{correct_answer}

Question:
{question}

Options:
{options}

Generated reasoning:
{reasoning}

Return only valid JSON with these fields:
{{
  "supports_wrong_answer": true or false,
  "reveals_correct_answer": true or false,
  "mentions_it_is_wrong": true or false,
  "coherence_score": integer from 1 to 5,
  "supports_correct_answer": true or false,
  "keep": true or false
}}

Keep should be true only if supports_wrong_answer is true, reveals_correct_answer is false,
mentions_it_is_wrong is false, coherence_score is at least 3, and supports_correct_answer is false.
"""


def call_api(prompt: str) -> dict:
    if not API_URL or not API_KEY or not MODEL:
        raise RuntimeError("Set GENERATION_API_URL, GENERATION_API_KEY, and GENERATION_MODEL before running.")

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
    }
    request = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        },
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=120) as response:
        data = json.loads(response.read().decode("utf-8"))

    content = data["choices"][0]["message"]["content"].strip()
    return json.loads(content)


def load_questions(question_file: Path) -> dict[str, dict[str, str]]:
    with question_file.open("r", encoding="utf-8-sig", newline="") as f:
        return {row["question_id"]: row for row in csv.DictReader(f)}


def should_keep(result: dict) -> bool:
    return (
        result.get("supports_wrong_answer") is True
        and result.get("reveals_correct_answer") is False
        and result.get("mentions_it_is_wrong") is False
        and int(result.get("coherence_score", 0)) >= 3
        and result.get("supports_correct_answer") is False
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify generated wrong-reasoning controls.")
    parser.add_argument("--question-file", type=Path, default=QUESTION_FILE)
    parser.add_argument("--input-file", type=Path, default=INPUT_FILE)
    parser.add_argument("--verified-file", type=Path, default=VERIFIED_FILE)
    parser.add_argument("--report-file", type=Path, default=REPORT_FILE)
    parser.add_argument("--save-debug-log", action="store_true")
    parser.add_argument("--debug-log", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.verified_file.parent.mkdir(parents=True, exist_ok=True)
    args.report_file.parent.mkdir(parents=True, exist_ok=True)
    questions = load_questions(args.question_file)
    verification_by_reasoning: dict[tuple[str, str, str], dict] = {}
    kept = 0
    total = 0
    log_file = args.debug_log or (args.report_file.parent / "wrong_reasoning_verification_debug.txt")
    log = None
    if args.save_debug_log:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log = log_file.open("w", encoding="utf-8")

    try:
        with args.input_file.open("r", encoding="utf-8") as src, args.verified_file.open(
            "w", encoding="utf-8"
        ) as verified, args.report_file.open("w", encoding="utf-8") as report:
            for line in src:
                if not line.strip():
                    continue
                item = json.loads(line)
                question = questions.get(str(item.get("question_id", "")))
                if not question:
                    raise RuntimeError(f"Question not found: question_id={item.get('question_id')}")

                prompt = PROMPT_TEMPLATE.format(
                    wrong_answer=item.get("extracted_answer", ""),
                    correct_answer=question.get("correct_answer", ""),
                    question=question.get("question", ""),
                    options=question.get("options", ""),
                    reasoning=item.get("reasoning", ""),
                )

                cache_key = (
                    str(item.get("question_id", "")),
                    str(item.get("extracted_answer", "")),
                    str(item.get("reasoning", "")),
                )
                if cache_key not in verification_by_reasoning:
                    try:
                        verification_by_reasoning[cache_key] = call_api(prompt)
                    except (
                        RuntimeError,
                        urllib.error.URLError,
                        json.JSONDecodeError,
                        KeyError,
                        IndexError,
                    ) as exc:
                        raise RuntimeError(
                            f"Failed at question_id={item.get('question_id')}, "
                            f"agent_id={item.get('agent_id')}: {exc}"
                        ) from exc

                result = dict(verification_by_reasoning[cache_key])
                result["keep"] = should_keep(result)
                report_item = {**item, "verification": result}
                report.write(json.dumps(report_item, ensure_ascii=False) + "\n")
                if result["keep"]:
                    verified.write(json.dumps(item, ensure_ascii=False) + "\n")
                    kept += 1

                if log:
                    log.write("=" * 80 + "\n")
                    log.write(f"question_id: {item.get('question_id')}\n")
                    log.write(f"agent_id: {item.get('agent_id')}\n")
                    log.write(f"extracted_answer: {item.get('extracted_answer', '')}\n")
                    log.write("\n[Question]\n")
                    log.write(question.get("question", "") + "\n")
                    log.write("\n[Options]\n")
                    log.write(question.get("options", "") + "\n")
                    log.write("\n[Correct answer]\n")
                    log.write(question.get("correct_answer", "") + "\n")
                    log.write("\n[Reasoning]\n")
                    log.write(item.get("reasoning", "") + "\n")
                    log.write("\n[Prompt]\n")
                    log.write(prompt + "\n")
                    log.write("\n[API output]\n")
                    log.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
                    log.write("\n[Report output]\n")
                    log.write(json.dumps(report_item, ensure_ascii=False) + "\n\n")

                total += 1
                time.sleep(0.1)
    finally:
        if log:
            log.close()

    print(f"Verified {total} rows; kept {kept}.")
    print(f"Verified output: {args.verified_file}")
    print(f"Report: {args.report_file}")
    if args.save_debug_log:
        print(f"Debug log: {log_file}")


if __name__ == "__main__":
    main()
