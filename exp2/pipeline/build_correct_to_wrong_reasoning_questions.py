import csv
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DETAILS_FILE = BASE_DIR / "reasoning_change_analysis_details.csv"
SOURCE_FILE = BASE_DIR.parent / "data" / "demo" / "demo_questions.csv"
OUTPUT_FILE = BASE_DIR / "correct_to_wrong_reasoning_questions.csv"

TARGET_CHANGE_TYPE = "Correct_to_Wrong"
OUTPUT_FIELDS = [
    "question_id",
    "correct_answer",
    "initial_answer",
    "change_type",
    "question",
    "options",
]


def load_source_questions(source_file: Path) -> dict[str, dict[str, str]]:
    with source_file.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required_fields = {"question_id", "question", "options"}
        missing_fields = required_fields - set(reader.fieldnames or [])
        if missing_fields:
            raise ValueError(f"Missing source columns: {sorted(missing_fields)}")

        return {row["question_id"]: row for row in reader}


def build_rows(
    details_file: Path, source_questions: dict[str, dict[str, str]]
) -> tuple[list[dict[str, str]], int, list[str]]:
    rows: list[dict[str, str]] = []
    seen_question_ids: set[str] = set()
    duplicate_count = 0
    missing_question_ids: list[str] = []

    with details_file.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required_fields = {
            "question_id",
            "correct_answer",
            "initial_answer",
            "change_type",
        }
        missing_fields = required_fields - set(reader.fieldnames or [])
        if missing_fields:
            raise ValueError(f"Missing detail columns: {sorted(missing_fields)}")

        for detail_row in reader:
            if detail_row["change_type"] != TARGET_CHANGE_TYPE:
                continue

            question_id = detail_row["question_id"]
            if question_id in seen_question_ids:
                duplicate_count += 1
                continue

            source_row = source_questions.get(question_id)
            if source_row is None:
                missing_question_ids.append(question_id)
                continue

            seen_question_ids.add(question_id)
            rows.append(
                {
                    "question_id": question_id,
                    "correct_answer": detail_row["correct_answer"],
                    "initial_answer": detail_row["initial_answer"],
                    "change_type": detail_row["change_type"],
                    "question": source_row["question"],
                    "options": source_row["options"],
                }
            )

    return rows, duplicate_count, missing_question_ids


def write_output(output_file: Path, rows: list[dict[str, str]]) -> None:
    with output_file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    source_questions = load_source_questions(SOURCE_FILE)
    rows, duplicate_count, missing_question_ids = build_rows(
        DETAILS_FILE, source_questions
    )
    write_output(OUTPUT_FILE, rows)

    print(f"Wrote {len(rows)} questions to {OUTPUT_FILE}")
    print(f"Skipped {duplicate_count} duplicate Correct_to_Wrong rows")
    if missing_question_ids:
        print(
            "Missing question IDs in source data: "
            + ", ".join(sorted(set(missing_question_ids)))
        )


if __name__ == "__main__":
    main()
