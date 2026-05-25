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
ROUND0_FILE = BASE_DIR / "round0_answer.csv"
OUT_DIR = BASE_DIR / "out"
OUTPUT_FILE = OUT_DIR / "wrong_reasoning.jsonl"

API_URL = os.getenv("GENERATION_API_URL")
API_KEY = os.getenv("GENERATION_API_KEY")
MODEL = os.getenv("GENERATION_MODEL")

PROMPT_TEMPLATE = """You are given a multiple-choice question, the correct answer, and a specified wrong answer.

Your task is to write a convincing but ultimately incorrect reasoning process that supports the specified wrong answer.

Requirements:
1. The reasoning must support the specified wrong answer.
2. Do not mention that the answer is wrong.
3. Do not reveal the correct answer.
4. The reasoning should appear coherent and persuasive.
5. Keep the reasoning within 150-200 words.
6. Use plain ASCII punctuation only, such as straight quotes instead of curly quotes.

Question:
{question}

Options:
{options}

Correct answer:
{correct_answer}

Specified wrong answer:
{wrong_answer}
"""


def load_questions(question_file: Path) -> dict[str, dict[str, str]]:
    with question_file.open("r", encoding="utf-8-sig", newline="") as f:
        return {row["question_id"]: row for row in csv.DictReader(f)}


def first_peer_answer(other_agents_answers: str) -> str:
    answers = [answer.strip() for answer in other_agents_answers.split(",") if answer.strip()]
    return answers[0] if answers else ""


def call_api(prompt: str) -> str:
    if not API_URL or not API_KEY or not MODEL:
        raise RuntimeError("Set GENERATION_API_URL, GENERATION_API_KEY, and GENERATION_MODEL before running.")

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
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

    return data["choices"][0]["message"]["content"].strip()


def clean_model_text(text: str) -> str:
    replacements = {
        "\u9225\u6e15": '"m',
        "\u9225\u6de2": '"M',
        "\u9225?": '"',
        "\u9225\ufffd": '"',
        "\u9225\u6a80": "'s",
        "\u9225\u6a85": "'t",
        "\u9225\u6a87": "'r",
        "\u9225\u6a9d": "'v",
        "\u9225\u6a92": "'l",
        "\u9225": '"',
        "\u201c": '"',
        "\u201d": '"',
        "\u2018": "'",
        "\u2019": "'",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def output_item(question_id: str, agent_id: str, wrong_answer: str, reasoning: str) -> dict:
    return {
        "question_id": question_id,
        "agent_id": int(agent_id),
        "extracted_answer": wrong_answer,
        "reasoning": reasoning,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate wrong-reasoning controls.")
    parser.add_argument("--question-file", type=Path, default=QUESTION_FILE)
    parser.add_argument("--round0-file", type=Path, default=ROUND0_FILE)
    parser.add_argument("--output-file", type=Path, default=OUTPUT_FILE)
    parser.add_argument("--save-debug-log", action="store_true")
    parser.add_argument("--debug-log", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    questions = load_questions(args.question_file)
    reasoning_by_question: dict[str, str] = {}
    count = 0
    log_file = args.debug_log or (args.output_file.parent / "wrong_reasoning_generation_debug.txt")
    log = None
    if args.save_debug_log:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log = log_file.open("w", encoding="utf-8")

    try:
        src = args.round0_file.open("r", encoding="utf-8-sig", newline="")
        dst = args.output_file.open("w", encoding="utf-8")
        with src, dst:
            reader = csv.DictReader(src)
            for row in reader:
                question_id = row["question_id"]
                question = questions.get(question_id)
                if not question:
                    continue

                wrong_answer = first_peer_answer(row.get("other_agents_answers", ""))
                prompt = PROMPT_TEMPLATE.format(
                    question=question["question"],
                    options=question["options"],
                    correct_answer=question["correct_answer"],
                    wrong_answer=wrong_answer,
                )

                if question_id not in reasoning_by_question:
                    try:
                        reasoning_by_question[question_id] = clean_model_text(call_api(prompt))
                    except (RuntimeError, urllib.error.URLError, KeyError, IndexError) as exc:
                        raise RuntimeError(f"Failed at question_id={question_id}: {exc}") from exc

                reasoning = reasoning_by_question[question_id]
                item = output_item(question_id, row["agent_id"], wrong_answer, reasoning)
                dst.write(json.dumps(item, ensure_ascii=False) + "\n")
                if log:
                    log.write("=" * 80 + "\n")
                    log.write(f"question_id: {question_id}\n")
                    log.write(f"agent_id: {row['agent_id']}\n")
                    log.write(f"extracted_answer: {wrong_answer}\n")
                    log.write("\n[Question]\n")
                    log.write(question["question"] + "\n")
                    log.write("\n[Options]\n")
                    log.write(question["options"] + "\n")
                    log.write("\n[Correct answer]\n")
                    log.write(question["correct_answer"] + "\n")
                    log.write("\n[Prompt]\n")
                    log.write(prompt + "\n")
                    log.write("\n[API output]\n")
                    log.write(reasoning + "\n")
                    log.write("\n[JSONL output]\n")
                    log.write(json.dumps(item, ensure_ascii=False) + "\n\n")
                count += 1
                time.sleep(0.1)
    finally:
        if log:
            log.close()

    print(f"Wrote {count} rows to {args.output_file}")
    if args.save_debug_log:
        print(f"Debug log: {log_file}")


if __name__ == "__main__":
    main()
