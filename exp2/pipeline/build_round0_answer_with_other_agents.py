import csv
import argparse
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
BASE_DIR = ROOT_DIR / "input"
ROUND0_FILE = BASE_DIR / "round0_answer.csv"
DETAILS_FILE = BASE_DIR / "reasoning_change_analysis_details.csv"
QUESTION_FILE = BASE_DIR / "correct_to_wrong_reasoning_questions.csv"
ANSWER_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
INVALID_ANSWERS = {"", "N"}

OUTPUT_FIELDS = [
    "question_id",
    "agent_id",
    "correct_answer",
    "initial_answer",
    "other_agents_answers",
]


def split_answers(value: str) -> list[str]:
    return [answer.strip() for answer in value.split(",") if answer.strip()]


def count_options(options: str) -> int:
    quoted_options = re.findall(r"'(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\"", options)
    return len(quoted_options)


def load_valid_answer_labels(question_file: Path) -> dict[str, set[str]]:
    labels_by_question: dict[str, set[str]] = {}

    with question_file.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required_fields = {"question_id", "options"}
        missing_fields = required_fields - set(reader.fieldnames or [])
        if missing_fields:
            raise ValueError(f"Missing question columns: {sorted(missing_fields)}")

        for row in reader:
            option_count = count_options(row["options"])
            if option_count <= 0:
                option_count = 10
            labels_by_question[row["question_id"]] = set(
                ANSWER_LABELS[:option_count]
            )

    return labels_by_question


def choose_wrong_answer(
    correct_answer: str,
    answers: list[str],
    valid_labels: set[str],
    rng: random.Random,
) -> str:
    wrong_answers = [
        answer
        for answer in answers
        if answer != correct_answer
        and answer not in INVALID_ANSWERS
        and answer in valid_labels
    ]
    if not wrong_answers:
        fallback_answers = sorted(valid_labels - {correct_answer} - INVALID_ANSWERS)
        if not fallback_answers:
            raise ValueError("No wrong answers available")
        return rng.choice(fallback_answers)

    counts = Counter(wrong_answers)
    max_count = max(counts.values())
    candidates = [answer for answer, count in counts.items() if count == max_count]
    return rng.choice(sorted(candidates))


def load_question_answer_pool(details_file: Path) -> dict[str, list[str]]:
    answers_by_question: dict[str, list[str]] = defaultdict(list)

    with details_file.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required_fields = {"question_id", "other_agents_answers"}
        missing_fields = required_fields - set(reader.fieldnames or [])
        if missing_fields:
            raise ValueError(f"Missing detail columns: {sorted(missing_fields)}")

        for row in reader:
            answers_by_question[row["question_id"]].extend(
                split_answers(row["other_agents_answers"])
            )

    return answers_by_question


def build_rows(
    round0_file: Path,
    answers_by_question: dict[str, list[str]],
    labels_by_question: dict[str, set[str]],
    rng: random.Random,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    chosen_answers: dict[str, str] = {}

    with round0_file.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required_fields = {
            "question_id",
            "agent_id",
            "correct_answer",
            "initial_answer",
        }
        missing_fields = required_fields - set(reader.fieldnames or [])
        if missing_fields:
            raise ValueError(f"Missing round0 columns: {sorted(missing_fields)}")

        for row in reader:
            question_id = row["question_id"]
            if question_id not in chosen_answers:
                source_answers = answers_by_question.get(question_id, [])
                valid_labels = labels_by_question.get(question_id, set(ANSWER_LABELS[:10]))
                chosen_answers[question_id] = choose_wrong_answer(
                    row["correct_answer"], source_answers, valid_labels, rng
                )

            wrong_answer = chosen_answers[question_id]
            rows.append(
                {
                    "question_id": question_id,
                    "agent_id": row["agent_id"],
                    "correct_answer": row["correct_answer"],
                    "initial_answer": row["initial_answer"],
                    "other_agents_answers": ",".join([wrong_answer] * 4),
                }
            )

    return rows


def write_output(output_file: Path, rows: list[dict[str, str]]) -> None:
    with output_file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a Round 0 answer file augmented with repeated wrong peer answers."
    )
    parser.add_argument("--round0-file", type=Path, default=ROUND0_FILE)
    parser.add_argument("--details-file", type=Path, default=DETAILS_FILE)
    parser.add_argument("--question-file", type=Path, default=QUESTION_FILE)
    parser.add_argument(
        "--output-file",
        type=Path,
        default=BASE_DIR / "round0_answer_with_other_agents.csv",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_file.resolve() == args.round0_file.resolve():
        raise ValueError("Refusing to overwrite the input Round 0 file. Use a different --output-file.")

    rng = random.Random(args.seed)
    answers_by_question = load_question_answer_pool(args.details_file)
    labels_by_question = load_valid_answer_labels(args.question_file)
    rows = build_rows(args.round0_file, answers_by_question, labels_by_question, rng)
    write_output(args.output_file, rows)
    print(f"Wrote {len(rows)} rows to {args.output_file}")


if __name__ == "__main__":
    main()
