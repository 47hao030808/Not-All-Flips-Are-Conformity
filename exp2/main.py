# -*- coding: utf-8 -*-
"""
Main Experiment Script - Multi-Agent Debate with Blind Following Analysis
"""

import argparse
import os
import sys

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

import csv
from datetime import datetime
from collections import defaultdict
from debate_system import DebateSystem
from agent import SignalType
from config import get_output_dir, get_output_path, set_output_dir
from sampling import (
    create_permanent_order,
    stratified_sampling_by_category,
    load_category_pointer,
    save_category_pointer,
    preview_sampling,
    get_category_distribution,
)
from api_client import GLOBAL_SEED, APIClient, MODEL_ANSWER

# Initialize output directory configuration
OUTPUT_DIR = os.getenv("OUTPUT_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs"))
set_output_dir(OUTPUT_DIR)

# Question loading, filtering, or selection logic
APIClient.set_fingerprint_log_path(OUTPUT_DIR)

# =============================================================================
# Round 0 data and bookkeeping
# Question loading, filtering, or selection logic
# =============================================================================

class Tee:
    """Class to output to both console and file"""

    def __init__(self, file_path):
        self.file = open(file_path, 'w', encoding='utf-8')
        self.stdout = sys.stdout
        sys.stdout = self

    def write(self, text):
        self.stdout.write(text)
        self.file.write(text)
        self.file.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        sys.stdout = self.stdout
        self.file.close()


def load_round0_answers(round0_answer_csv_path: str) -> dict:
    """load_round0_answers helper."""
    round0_data = {}  # {question_id: {agent_id: answer}}

    if os.path.exists(round0_answer_csv_path):
        with open(round0_answer_csv_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                question_id = row.get('question_id', '').strip()
                agent_id = int(row.get('agent_id', 0))
                extracted_answer = (
                    row.get('extracted_answer', '').strip()
                    or row.get('initial_answer', '').strip()
                    or row.get('round0_answer', '').strip()
                )

                if question_id and agent_id and extracted_answer:
                    if question_id not in round0_data:
                        round0_data[question_id] = {}
                    round0_data[question_id][agent_id] = extracted_answer

    print(f"Loaded round0 answers for {len(round0_data)} questions from {round0_answer_csv_path}")
    return round0_data


def load_round0_reasoning(round0_reasoning_jsonl_path: str) -> dict:
    """load_round0_reasoning helper."""
    import json
    round0_data = {}  # {question_id: {agent_id: reasoning}}

    if os.path.exists(round0_reasoning_jsonl_path):
        with open(round0_reasoning_jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    question_id = str(data.get('question_id', '')).strip()
                    agent_id = int(data.get('agent_id', 0))
                    reasoning = data.get('reasoning', '').strip()

                    if question_id and agent_id and reasoning:
                        if question_id not in round0_data:
                            round0_data[question_id] = {}
                        round0_data[question_id][agent_id] = reasoning
                except json.JSONDecodeError:
                    continue

    print(f"Loaded round0 reasoning for {len(round0_data)} questions from {round0_reasoning_jsonl_path}")
    return round0_data


def load_round0_peer_answers(round0_answer_csv_path: str) -> dict:
    peer_data = {}

    if os.path.exists(round0_answer_csv_path):
        with open(round0_answer_csv_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                question_id = row.get('question_id', '').strip()
                agent_id_text = row.get('agent_id', '').strip()
                peer_answers_text = row.get('other_agents_answers', '').strip()
                if not question_id or not agent_id_text or not peer_answers_text:
                    continue

                try:
                    agent_id = int(agent_id_text)
                except ValueError:
                    continue

                peer_answers = [
                    answer.strip()
                    for answer in peer_answers_text.split(',')
                    if answer.strip()
                ]
                if peer_answers:
                    peer_data.setdefault(question_id, {})[agent_id] = peer_answers

    print(f"Loaded peer answers for {len(peer_data)} questions from {round0_answer_csv_path}")
    return peer_data


def parse_options_string(options_str: str) -> list:
    import ast
    import re

    if not options_str:
        return []

    candidates = [options_str]
    normalized = re.sub(r"(?<=['\"])\s+(?=['\"])", ", ", options_str.strip())
    if normalized != options_str:
        candidates.insert(0, normalized)

    for candidate in candidates:
        try:
            parsed_options = ast.literal_eval(candidate)
        except Exception:
            continue
        if isinstance(parsed_options, list):
            options = [str(option).strip() for option in parsed_options if str(option).strip()]
            if len(options) > 1:
                return options

    matches = re.findall(r"'((?:\\'|[^'])*)'|\"((?:\\\"|[^\"])*)\"", options_str, re.DOTALL)
    options = []
    for single_quoted, double_quoted in matches:
        option = single_quoted or double_quoted
        option = option.replace("\\'", "'").replace('\\"', '"').strip()
        if option:
            options.append(option)
    return options


def load_controlled_questions(question_csv_path: str, verified_only: bool = False) -> list:
    questions = []

    if not os.path.exists(question_csv_path):
        print(f"Error: controlled question file not found: {question_csv_path}")
        return questions

    with open(question_csv_path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            verification_status = row.get("verification_status", "").strip()
            if verified_only and verification_status != "verified_passed":
                continue

            question_id = row.get("question_id", "").strip()
            question_text = row.get("question", "").strip()
            options_str = row.get("options", "").strip()
            correct_answer = row.get("correct_answer", "").strip().upper()

            options = parse_options_string(options_str)

            if options:
                options_text = "\n".join(
                    [f"{chr(65 + i)}. {option}" for i, option in enumerate(options)]
                )
                formatted_question = f"{question_text}\n\nOptions:\n{options_text}"
            else:
                formatted_question = question_text

            questions.append({
                "question_id": question_id,
                "question": formatted_question,
                "original_question": question_text,
                "options_str": options_str,
                "options": options,
                "correct_answer": correct_answer,
                "label": row.get("change_type", "").strip(),
                "verification_status": verification_status,
            })

    status_text = " verified-passed" if verified_only else ""
    print(f"Loaded {len(questions)}{status_text} controlled questions from {question_csv_path}")
    return questions


def get_run_status():
    """get_run_status helper."""
    output_dir = get_output_dir()
    round1_question_path = get_output_path("round1_question.csv")
    
    # Round 1 data and bookkeeping
    divided_question_ids_ordered = []
    
    if os.path.exists(round1_question_path):
        with open(round1_question_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                question_id = row.get('question_id', '').strip()
                if question_id:
                    divided_question_ids_ordered.append(question_id)
    
    if not divided_question_ids_ordered:
        return {"completed_question_ids": set(), "last_question_id": None, "total_questions": 0}
    
    # Round 1 data and bookkeeping
    # Round 1 data and bookkeeping
    signal_file_names = [
        "self_reflection",
        "stance_only",
        "reasoning",
        "invalid_reasoning",
        "wrong_reasoning",
    ]
    completed_question_ids = set()
    
    for file_key in signal_file_names:
        answer_path = get_output_path(f"round1_answer_{file_key}.csv")
        if os.path.exists(answer_path):
            with open(answer_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    question_id = row.get('question_id', '').strip()
                    if question_id:
                        completed_question_ids.add(question_id)
    last_question_id = None
    for qid in reversed(divided_question_ids_ordered):
        if qid in completed_question_ids:
            last_question_id = qid
            break
    
    return {
        "completed_question_ids": completed_question_ids,
        "last_question_id": last_question_id,
        "total_questions": len(divided_question_ids_ordered)
    }


def get_unrun_questions(test_questions_data: list, completed_question_ids: set) -> list:
    """get_unrun_questions helper."""
    unrun_questions = []
    for q in test_questions_data:
        question_id = q.get("question_id", "")
        if question_id and question_id not in completed_question_ids:
            unrun_questions.append(q)
    return unrun_questions


def load_divided_questions_from_label(label_csv_path: str, dataset_csv_path: str = None, 
                                      start_after_question_id: str = None) -> list:
    """Load divided questions from labels and merge full question text from the dataset."""
    divided_question_ids = []
    label_map = {}  # question_id -> label
    dataset_questions = {}

    if os.path.exists(label_csv_path):
        with open(label_csv_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                label = row.get('label', '').strip()
                question_id = row.get('question_id', '').strip()
                if label in ['divided_correct', 'divided_incorrect'] and question_id:
                    divided_question_ids.append(question_id)
                    label_map[question_id] = label

    print(f"Found {len(divided_question_ids)} divided questions in question_label.csv")
    print(f"  - divided_correct: {sum(1 for l in label_map.values() if l == 'divided_correct')}")
    print(f"  - divided_incorrect: {sum(1 for l in label_map.values() if l == 'divided_incorrect')}")

    if not divided_question_ids:
        return []

    if dataset_csv_path and os.path.exists(dataset_csv_path):
        import re
        with open(dataset_csv_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            
            for row in reader:
                if not row or len(row) < 4:
                    continue
                
                question_id = row[0].strip()
                if question_id not in divided_question_ids:
                    continue
                
                question_text = row[1].strip()
                
                options_parts = row[2:-3]
                combined_options_str = ''.join(options_parts)
                combined_options_str = combined_options_str.replace('\n', ' ')
                options_str = combined_options_str.strip()
                
                options_str = re.sub(r"(?<=['\"])\s+(?=['\"])", ", ", options_str)
                
                # Question loading, filtering, or selection logic
                if options_str.startswith('[') and not options_str.endswith(']'):
                    last_bracket = options_str.rfind(']')
                    if last_bracket > 0:
                        options_str = options_str[:last_bracket + 1]
                
                answer_index = row[4].strip() if len(row) > 4 else ''
                category = row[6].strip() if len(row) > 6 else ''
                correct_answer = row[3].strip().upper()
                
                options = []
                if options_str.startswith('[') and options_str.endswith(']'):
                    try:
                        import ast
                        options_raw = ast.literal_eval(options_str)
                        if isinstance(options_raw, list):
                            options = [str(opt).strip() for opt in options_raw if opt]
                    except Exception:
                        pass
                
                if not options:
                    pattern = r'''(['"])(.*?)(\1)'''
                    matches = re.findall(pattern, options_str, re.DOTALL)
                    if matches:
                        options = [match[1].strip() for match in matches if match[1].strip()]
                        options = [opt.strip().strip("'\"").strip() for opt in options]
                        options = [opt for opt in options if opt]

                if options:
                    options_text = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
                    formatted_question = f"{question_text}\n\nOptions:\n{options_text}"
                else:
                    formatted_question = question_text

                dataset_questions[question_id] = {
                    "question": formatted_question,
                    "original_question": question_text,
                    "options_str": options_str,
                    "options": options,
                    "correct_answer": correct_answer,
                    "answer_index": answer_index,
                    "dataset": category,
                }

        print(f"Loaded {len(dataset_questions)} dataset questions.")
    else:
        print("Warning: dataset file was not found; full question text is unavailable")

    questions = []
    start_adding = start_after_question_id is None
    for question_id in divided_question_ids:
        if not start_adding:
            if question_id == start_after_question_id:
                start_adding = True
            continue
        
        if question_id in dataset_questions:
            q = dataset_questions[question_id]
            q["question_id"] = question_id
            q["label"] = label_map.get(question_id, "")
            questions.append(q)

    return questions


def load_questions_from_dataset(file_path: str):
    """load_questions_from_dataset helper."""
    import re
    import csv as csv_module
    
    questions = []
    if not os.path.exists(file_path):
        return questions

    try:
        with open(file_path, "r", encoding="utf-8-sig") as f:
            reader = csv_module.reader(f)
            header = next(reader, None)
            
            for row in reader:
                if not row or len(row) < 4:
                    continue
                
                try:
                    question_id = row[0].strip()
                    question_text = row[1].strip()
                    options_str = row[2].strip()
                    correct_answer = row[3].strip().upper()
                    category = row[6].strip() if len(row) > 6 else "unknown"
                    
                    # Question loading, filtering, or selection logic
                    options = []
                    
                    def is_complete_list(s):
                        """Check whether a string is a complete Python list literal."""
                        s = s.strip()
                        return s.startswith('[') and s.endswith(']') and s.count('[') == s.count(']')
                    
                    # Question loading, filtering, or selection logic
                    options_parts = row[2:-3]  # Answer extraction or answer bookkeeping
                    combined_options_str = ''.join(options_parts)
                    # Question loading, filtering, or selection logic
                    options_str = combined_options_str.strip()
                    
                    import re
                    options_str = re.sub(r"(?<=['\"])\s+(?=['\"])", ", ", options_str)
                    
                    if options_str.startswith('[') and not options_str.endswith(']'):
                        last_bracket = options_str.rfind(']')
                        if last_bracket > 0:
                            options_str = options_str[:last_bracket + 1]
                    
                    
                    
                    if is_complete_list(options_str):
                        import ast
                        try:
                            options_raw = ast.literal_eval(options_str)
                            if isinstance(options_raw, list):
                                options = [str(opt).strip() for opt in options_raw if opt]
                        except Exception:
                            pass
                    
                    if not options:
                        import re
                        pattern = r'''(['"])(.*?)(\1)'''
                        matches = re.findall(pattern, options_str, re.DOTALL)
                        if matches and len(matches) >= 2:
                            options = [match[1].strip() for match in matches if match[1].strip()]
                        
                        if not options:
                            all_text = ''.join(row)
                            matches = re.findall(pattern, all_text, re.DOTALL)
                            if matches and len(matches) >= 2:
                                options = [match[1].strip() for match in matches if match[1].strip()]
                    
                    # Question loading, filtering, or selection logic
                    if options:
                        options = [opt.strip().strip("'\"").strip() for opt in options]
                        options = [opt for opt in options if opt]
                    
                    if options:
                        options_text = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
                        formatted_question = f"{question_text}\n\nOptions:\n{options_text}"
                    else:
                        formatted_question = question_text
                    
                    questions.append({
                        "question": formatted_question,
                        "original_question": question_text,
                        "options_str": options_str,  # Question loading, filtering, or selection logic
                        "options": options,
                        "correct_answer": correct_answer,
                        "dataset": category,
                        "question_id": question_id,
                    })
                except Exception as e:
                    print(f"Failed to parse row: {row[:4]}, error: {e}")
                    continue
                    
    except Exception as e:
        print(f"Failed to load questions from {file_path}: {e}")
    
    print(f"Loaded {len(questions)} questions from the dataset")
    return questions


def save_round0_answer_csv(results: list, output_path: str):
    """save_round0_answer_csv helper."""
    fieldnames = ["question_id", "agent_id", "extracted_answer", "is_correct"]
    file_exists = os.path.exists(output_path) and os.path.getsize(output_path) > 0
    with open(output_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(results)


def save_round0_reasoning_jsonl(results: list, output_path: str):
    """save_round0_reasoning_jsonl helper."""
    import json

    with open(output_path, 'a', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def save_round0_raw_csv(results: list, output_path: str):
    """save_round0_raw_csv helper."""
    fieldnames = ["question_id", "agent_id", "extracted_answer", "correct_answer", "tokens", "explicit_confidence", "implicit_confidence", "is_correct", "reasoning_length"]

    file_exists = os.path.exists(output_path) and os.path.getsize(output_path) > 0
    with open(output_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(results)


def save_question_summary_csv(questions_data: list, output_path: str):
    """save_question_summary_csv helper."""
    fieldnames = ["question_id", "question", "options", "answer", "type"]

    file_exists = os.path.exists(output_path) and os.path.getsize(output_path) > 0
    with open(output_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(questions_data)


def save_exacted_question_csv(questions_data: list, output_path: str):
    """save_exacted_question_csv helper."""
    fieldnames = ["question_id", "correct_answer", "question_with_options"]
    file_exists = os.path.exists(output_path) and os.path.getsize(output_path) > 0
    with open(output_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for q in questions_data:
            question_text = q.get("original_question", "")
            correct_answer = q.get("correct_answer", "")
            options = q.get("options", [])
            if options:
                options_text = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
                question_with_options = f"{question_text}\n\nOptions:\n{options_text}"
            else:
                options_str = q.get("options_str", "")
                question_with_options = f"{question_text}\n\nOptions:\n{options_str}"
            writer.writerow({
                "question_id": q.get("question_id", ""),
                "correct_answer": correct_answer,
                "question_with_options": question_with_options
            })


def save_question_label_csv(results: list, output_path: str):
    """save_question_label_csv helper."""
    fieldnames = ["question_id", "correct_answer", "label"]

    file_exists = os.path.exists(output_path) and os.path.getsize(output_path) > 0
    with open(output_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(results)


def save_round1_answer_csv(results: list, output_path: str):
    """save_round1_answer_csv helper."""
    fieldnames = ["question_id", "agent_id", "round1_answer", "is_correct", "round0_answer", "changed_answer"]

    file_exists = os.path.exists(output_path) and os.path.getsize(output_path) > 0
    with open(output_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(results)


def save_round1_raw_csv(results: list, output_path: str):
    """save_round1_raw_csv helper."""
    fieldnames = ["question_id", "agent_id", "extracted_answer", "correct_answer", "explicit_confidence", "implicit_confidence", "is_correct", "reasoning_length"]

    file_exists = os.path.exists(output_path) and os.path.getsize(output_path) > 0
    with open(output_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(results)


def init_round1_output_files(signal_types=None):
    answer_fields = ["question_id", "agent_id", "round1_answer", "is_correct", "round0_answer", "changed_answer"]
    raw_fields = ["question_id", "agent_id", "extracted_answer", "correct_answer", "explicit_confidence", "implicit_confidence", "is_correct", "reasoning_length"]

    if signal_types is None:
        signal_types = [
            SignalType.SELF_REFLECTION,
            SignalType.STANCE_ONLY,
            SignalType.WRONG_REASONING,
            SignalType.INVALID_REASONING,
        ]

    for signal_type in signal_types:
        signal_name = signal_type.name.lower()
        answer_path = get_output_path(f"round1_answer_{signal_name}.csv")
        raw_path = get_output_path(f"round1_raw_{signal_name}.csv")
        reasoning_path = get_output_path(f"round1_reasoning_{signal_name}.jsonl")

        if not os.path.exists(answer_path):
            with open(answer_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=answer_fields)
                writer.writeheader()

        if not os.path.exists(raw_path):
            with open(raw_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=raw_fields)
                writer.writeheader()

        if not os.path.exists(reasoning_path):
            with open(reasoning_path, 'w', encoding='utf-8'):
                pass

        print(f"  checked output files: {signal_name}")
        print(f"    - {answer_path}")
        print(f"    - {raw_path}")
        print(f"    - {reasoning_path}")

def save_round1_reasoning_jsonl(results: list, output_path: str):
    """save_round1_reasoning_jsonl helper."""
    import json

    with open(output_path, 'a', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def save_round1_question_csv(questions_data: list, label_map: dict, output_path: str, append: bool = False):
    fieldnames = ["question_id", "label", "correct_answer", "question"]

    existing_question_ids = set()
    if append and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        with open(output_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                question_id = row.get('question_id', '').strip()
                if question_id:
                    existing_question_ids.add(question_id)

    results = []
    for q in questions_data:
        question_id = q.get("question_id", "")
        if append and question_id in existing_question_ids:
            continue

        options = q.get("options", [])
        original_question = q.get("original_question", "")
        if options:
            options_text = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
            full_question = f"{original_question}\n\nOptions:\n{options_text}"
        else:
            full_question = original_question

        results.append({
            "question_id": question_id,
            "label": label_map.get(question_id, ""),
            "correct_answer": q.get("correct_answer", ""),
            "question": full_question,
        })

    if not results:
        print("  round1_question.csv: no new questions to add")
        return

    file_exists = os.path.exists(output_path) and os.path.getsize(output_path) > 0
    with open(output_path, 'a' if append else 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists or not append:
            writer.writeheader()
        writer.writerows(results)

    action = "appended" if append else "saved"
    print(f"  round1_question.csv: {action} {len(results)} questions")

def run_round0_experiment(test_questions_data, test_questions, num_agents=5, num_rounds=1):
    """run_round0_experiment helper."""
    # Round 0 data and bookkeeping
    question_summary_data = []
    for q in test_questions_data:
        options_str = q.get("options_str", str(q.get("options", [])))
        question_summary_data.append({
            "question_id": q.get("question_id", ""),
            "question": q.get("original_question", ""),
            "options": options_str,
            "answer": q.get("correct_answer", ""),
            "type": q.get("dataset", "")
        })
    summary_csv_path = get_output_path("question_summary.csv")
    save_question_summary_csv(question_summary_data, summary_csv_path)
    print(f"  question summary saved to: {summary_csv_path}")

    exacted_csv_path = get_output_path("exacted_question.csv")
    save_exacted_question_csv(test_questions_data, exacted_csv_path)
    print(f"  exact question file saved to: {exacted_csv_path}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = get_output_path(f"debate_log_round0_{timestamp}.txt")
    tee = Tee(log_filename)
    round0_results = []
    round0_reasoning_results = []
    round0_raw_results = []
    
    try:
        print("=" * 60)
        print("Round 0 experiment - independent answers")
        print("=" * 60)
        print(f"Configuration:")
        print(f"  - Number of agents: {num_agents}")
        print(f"  - Number of questions: {len(test_questions)}")
        print(f"  - Model: {MODEL_ANSWER}")
        print(f"  - Log file: {log_filename}")
        print("=" * 60)
        print()
        debate_system = DebateSystem(num_agents=num_agents)
        
        for i, (question, question_data) in enumerate(zip(test_questions, test_questions_data)):
            current_question_index = i + 1
            question_id = question_data.get("question_id", f"q_{current_question_index}")
            correct_answer = question_data.get("correct_answer", "")
            
            print(f"\n{'#'*60}")
            print(f"Experiment {i+1}/{len(test_questions)} (Question Index: {current_question_index})")
            print(f"Question ID: {question_id}")
            print(f"{'#'*60}")
            
            result = debate_system.run_debate(
                question=question,
                num_rounds=num_rounds,
                question_index=current_question_index,
                question_id=question_id,
                correct_answer=correct_answer,
                signal_type=None,  # Round 0 data and bookkeeping
                debate_round=0,
            )
            
            # Round 0 data and bookkeeping
            if result.get("round_results"):
                for agent in debate_system.agents:
                    extracted_answer = ""
                    reasoning = ""
                    implicit_confidence = None
                    explicit_confidence = None
                    if agent.history:
                        last_entry = agent.history[-1]
                        extracted_answer = last_entry.get("extracted_answer", "")
                        reasoning = last_entry.get("response", "")
                        implicit_confidence = last_entry.get("answer_confidence")
                        explicit_confidence = last_entry.get("explicit_confidence")
                    
                    is_correct = (extracted_answer.upper() == correct_answer.upper()) if extracted_answer and correct_answer else False
                    
                    round0_results.append({
                        "question_id": question_id,
                        "agent_id": agent.agent_id,
                        "extracted_answer": extracted_answer,
                        "is_correct": is_correct
                    })
                    
                    csv_path = get_output_path("round0_answer.csv")
                    save_round0_answer_csv([round0_results[-1]], csv_path)
                    
                    round0_reasoning_results.append({
                        "question_id": question_id,
                        "agent_id": agent.agent_id,
                        "extracted_answer": extracted_answer,
                        "reasoning": reasoning
                    })
                    # Question loading, filtering, or selection logic
                    jsonl_path = get_output_path("round0_reasoning.jsonl")
                    save_round0_reasoning_jsonl([round0_reasoning_results[-1]], jsonl_path)
                    
                    round0_raw_results.append({
                        "question_id": question_id,
                        "agent_id": agent.agent_id,
                        "extracted_answer": extracted_answer,
                        "correct_answer": correct_answer,
                        "tokens": "",
                        "explicit_confidence": explicit_confidence if explicit_confidence is not None else "",
                        "implicit_confidence": implicit_confidence if implicit_confidence is not None else "",
                        "is_correct": is_correct,
                        "reasoning_length": len(reasoning)
                    })
                    # Question loading, filtering, or selection logic
                    raw_csv_path = get_output_path("round0_raw.csv")
                    save_round0_raw_csv([round0_raw_results[-1]], raw_csv_path)
        
        # Agent-level state or peer information
        if round0_results:
            answers_by_question = defaultdict(list)
            for item in round0_results:
                answers_by_question[item["question_id"]].append(item)
            
            correct_answers_map = {}
            for raw in round0_raw_results:
                qid = raw["question_id"]
                if qid not in correct_answers_map:
                    correct_answers_map[qid] = raw.get("correct_answer", "")
            
            question_labels = []
            for question_id, answers in answers_by_question.items():
                correct_answer = correct_answers_map.get(question_id, "")
                unique_answers = set(a["extracted_answer"] for a in answers)
                
                if len(unique_answers) == 1:
                    label = "correct_consensus" if answers[0]["is_correct"] else "incorrect_consensus"
                else:
                    has_correct = any(a["is_correct"] for a in answers)
                    label = "divided_correct" if has_correct else "divided_incorrect"
                
                question_labels.append({
                    "question_id": question_id,
                    "correct_answer": correct_answer,
                    "label": label
                })
            
            if question_labels:
                label_path = get_output_path("question_label.csv")
                save_question_label_csv(question_labels, label_path)
                print(f"  question labels saved to: {label_path}")
        
        print(f"\n\n{'='*60}")
        print("Round 0 experiment complete")
        print(f"{'='*60}")
        print(f"Log file: {log_filename}")
        
    finally:
        tee.close()
        print(f"Log file closed: {log_filename}")


def run_round1_experiment(test_questions_data, test_questions, round0_answers, round0_reasoning, 
                          signal_type, num_agents=5, num_rounds=1, round0_peer_answers=None):
    """run_round1_experiment helper."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    signal_name = signal_type.name.lower()
    log_filename = get_output_path(f"debate_log_{signal_name}_{timestamp}.txt")
    tee = Tee(log_filename)
    results = {
        "answers": [],
        "raw_results": [],
        "reasoning_results": [],
        "log_filename": log_filename,
        "signal_type": signal_type
    }
    
    try:
        print("=" * 60)
        print(f"Round 1 experiment - {signal_type.name}")
        print("=" * 60)
        print(f"Configuration:")
        print(f"  - Signal type: {signal_type.value} ({signal_type.name})")
        print(f"  - Number of agents: {num_agents}")
        print(f"  - Number of questions: {len(test_questions)}")
        print(f"  - Model: {MODEL_ANSWER}")
        print(f"  - Log file: {log_filename}")
        print("=" * 60)
        print()
        
        # Round 1 data and bookkeeping
        print("\nInitializing Round 1 output files...")
        init_round1_output_files([signal_type])
        print()
        debate_system = DebateSystem(
            num_agents=num_agents,
            round0_answers=round0_answers,
            round0_reasoning=round0_reasoning,
            round0_peer_answers=round0_peer_answers or {}
        )
        
        for i, (question, question_data) in enumerate(zip(test_questions, test_questions_data)):
            current_question_index = i + 1
            question_id = question_data.get("question_id", f"q_{current_question_index}")
            
            print(f"\n{'#'*60}")
            print(f"Experiment {i+1}/{len(test_questions)} (Question Index: {current_question_index})")
            print(f"Question ID: {question_id}")
            print(f"{'#'*60}")
            
            result = debate_system.run_debate(
                question=question,
                num_rounds=num_rounds,
                question_index=current_question_index,
                question_id=question_id,
                correct_answer=question_data.get("correct_answer"),
                signal_type=signal_type,
                debate_round=1,
            )
            
            # Round 1 data and bookkeeping
            round1_answers = getattr(debate_system, 'round1_answers', {})
            q_round0_answers = round0_answers.get(question_id, {})
            correct_answer = question_data.get("correct_answer", "")
            
            for agent in debate_system.agents:
                agent_id = agent.agent_id
                r1_answer = round1_answers.get(question_id, {}).get(agent_id, "")
                r0_answer = q_round0_answers.get(agent_id, "")
                changed = (r1_answer != r0_answer and r0_answer != "") if r1_answer else False
                is_correct = (r1_answer.upper() == correct_answer.upper()) if r1_answer and correct_answer else False
                results["answers"].append({
                    "question_id": question_id,
                    "agent_id": agent_id,
                    "round1_answer": r1_answer,
                    "is_correct": is_correct,
                    "round0_answer": r0_answer,
                    "changed_answer": changed
                })
                # Question loading, filtering, or selection logic
                csv_path = get_output_path(f"round1_answer_{signal_name}.csv")
                save_round1_answer_csv([results["answers"][-1]], csv_path)
                if agent.history:
                    last_entry = agent.history[-1]
                    explicit_conf = last_entry.get("explicit_confidence")
                    implicit_conf = last_entry.get("answer_confidence")
                    response_text = last_entry.get("response", "")
                    
                    results["raw_results"].append({
                        "question_id": question_id,
                        "agent_id": agent_id,
                        "extracted_answer": r1_answer,
                        "correct_answer": correct_answer,
                        "explicit_confidence": explicit_conf if explicit_conf is not None else "",
                        "implicit_confidence": implicit_conf if implicit_conf is not None else "",
                        "is_correct": is_correct,
                        "reasoning_length": len(response_text)
                    })
                    # Question loading, filtering, or selection logic
                    raw_path = get_output_path(f"round1_raw_{signal_name}.csv")
                    save_round1_raw_csv([results["raw_results"][-1]], raw_path)
                    results["reasoning_results"].append({
                        "question_id": question_id,
                        "agent_id": agent_id,
                        "extracted_answer": r1_answer,
                        "reasoning": response_text
                    })
                    # Question loading, filtering, or selection logic
                    reasoning_path = get_output_path(f"round1_reasoning_{signal_name}.jsonl")
                    save_round1_reasoning_jsonl([results["reasoning_results"][-1]], reasoning_path)
        
        print(f"\n\n{'='*60}")
        print(f"Round 1 {signal_type.name} experiment complete")
        print(f"{'='*60}")
        
    finally:
        tee.close()
        print(f"Log file closed: {log_filename}")
    
    return results


def save_round1_experiment_results(results):
    """save_round1_experiment_results helper."""
    signal_name = results["signal_type"].name.lower()
    if results["answers"]:
        csv_path = get_output_path(f"round1_answer_{signal_name}.csv")
        save_round1_answer_csv(results["answers"], csv_path)
        print(f"Round 1 {results['signal_type'].name} answers saved to: {csv_path}")
    if results["raw_results"]:
        raw_path = get_output_path(f"round1_raw_{signal_name}.csv")
        save_round1_raw_csv(results["raw_results"], raw_path)
        print(f"Round 1 {results['signal_type'].name} raw data saved to: {raw_path}")
    if results["reasoning_results"]:
        reasoning_path = get_output_path(f"round1_reasoning_{signal_name}.jsonl")
        save_round1_reasoning_jsonl(results["reasoning_results"], reasoning_path)
        print(f"Round 1 {results['signal_type'].name} reasoning saved to: {reasoning_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run wrong-reasoning and invalid-reasoning control experiments.")
    parser.add_argument("--output_dir", default=os.getenv("OUTPUT_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")))
    parser.add_argument("--input_dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "input"))
    parser.add_argument("--questions", default=None, help="Controlled questions CSV. Defaults to input_dir/correct_to_wrong_reasoning_questions.csv.")
    parser.add_argument("--round0", default=None, help="Round 0 answer CSV. Defaults to input_dir/round0_answer.csv.")
    parser.add_argument("--wrong_reasoning", default=None, help="Verified wrong-reasoning JSONL.")
    parser.add_argument("--invalid_reasoning", default=None, help="Invalid-reasoning JSONL.")
    parser.add_argument("--num_agents", type=int, default=5)
    parser.add_argument("--num_rounds", type=int, default=1)
    return parser.parse_args()

def run_experiment(args=None):
    args = args or parse_args()
    set_output_dir(args.output_dir)
    APIClient.set_fingerprint_log_path(args.output_dir)
    num_agents = args.num_agents
    num_rounds = args.num_rounds
    base_dir = os.path.dirname(os.path.abspath(__file__))

    print("\n" + "=" * 60)
    print("Wrong-reasoning and invalid-reasoning control experiment")
    print("conditions: self_reflection, stance_only, wrong_reasoning, invalid_reasoning")
    print("=" * 60)

    input_dir = args.input_dir
    question_csv_path = args.questions or os.path.join(input_dir, "correct_to_wrong_reasoning_questions.csv")
    round0_answer_csv_path = args.round0 or os.path.join(input_dir, "round0_answer.csv")

    test_questions_data = load_controlled_questions(question_csv_path, verified_only=True)
    test_questions = [q["question"] for q in test_questions_data]
    round0_answers = load_round0_answers(round0_answer_csv_path)
    round0_peer_answers = load_round0_peer_answers(round0_answer_csv_path)

    if not test_questions_data:
        print("Error: no controlled questions loaded")
        return
    if not round0_answers:
        print("Error: no round0 answers loaded")
        return

    multiple_signal_types = [
        SignalType.SELF_REFLECTION,
        SignalType.STANCE_ONLY,
        SignalType.WRONG_REASONING,
        SignalType.INVALID_REASONING,
    ]
    controlled_reasoning_files = {
        SignalType.WRONG_REASONING: args.wrong_reasoning or os.path.join(input_dir, "out", "wrong_reasoning_verified.jsonl"),
        SignalType.INVALID_REASONING: args.invalid_reasoning or os.path.join(input_dir, "invalid_reasoning.jsonl"),
    }

    for index, signal_type in enumerate(multiple_signal_types):
        print("\n\n" + "#" * 60)
        print(f"Starting c{index}: {signal_type.name}")
        print("#" * 60)

        condition_reasoning = {}
        reasoning_path = controlled_reasoning_files.get(signal_type)
        if reasoning_path:
            condition_reasoning = load_round0_reasoning(reasoning_path)
            if not condition_reasoning:
                print(f"Warning: no reasoning loaded for {signal_type.name}: {reasoning_path}")

        run_round1_experiment(
            test_questions_data,
            test_questions,
            round0_answers,
            condition_reasoning,
            signal_type,
            num_agents,
            num_rounds,
            round0_peer_answers=round0_peer_answers,
        )

    print("\nWrong-reasoning and invalid-reasoning control experiment completed")

if __name__ == "__main__":
    try:
        run_experiment()
    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user.")
    except Exception as e:
        print(f"\n\nError occurred: {e}")
        import traceback
        traceback.print_exc()
