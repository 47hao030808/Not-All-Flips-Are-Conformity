# -*- coding: utf-8 -*-
"""Module helpers for the experiment pipeline."""

import os
import hashlib
import json
from collections import defaultdict
from typing import List, Dict, Tuple, Optional


def hash_question_id(qid: str) -> int:
    """hash_question_id helper."""
    hash_bytes = hashlib.sha256(qid.encode('utf-8')).digest()
    return int.from_bytes(hash_bytes[:8], byteorder='big')


def create_permanent_order(questions: List[Dict]) -> List[Dict]:
    """create_permanent_order helper."""
    sorted_questions = sorted(questions, key=lambda q: hash_question_id(q["question_id"]))
    return sorted_questions


def stratified_sampling_by_category(
    globally_sorted_questions: List[Dict], 
    k: int, 
    category_pointer: Optional[Dict[str, int]] = None
) -> Tuple[List[Dict], Dict[str, int]]:
    """stratified_sampling_by_category helper."""
    # Category-level sampling logic
    category_questions = defaultdict(list)
    for q in globally_sorted_questions:
        category = q.get("dataset", "unknown")
        category_questions[category].append(q)
    
    # Persistent pointer for resumable sampling
    if category_pointer is None:
        category_pointer = {cat: 0 for cat in category_questions.keys()}
    
    # Persistent pointer for resumable sampling
    selected_questions = []
    updated_pointer = {}
    
    for category, questions_list in category_questions.items():
        start_idx = category_pointer.get(category, 0)
        end_idx = start_idx + k
        batch = questions_list[start_idx:end_idx]
        selected_questions.extend(batch)
        
        # Persistent pointer for resumable sampling
        updated_pointer[category] = end_idx
    
    return selected_questions, updated_pointer


def load_category_pointer(pointer_file: str) -> Optional[Dict[str, int]]:
    """load_category_pointer helper."""
    if os.path.exists(pointer_file):
        with open(pointer_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def save_category_pointer(pointer_file: str, pointer: Dict[str, int]) -> None:
    """save_category_pointer helper."""
    os.makedirs(os.path.dirname(pointer_file), exist_ok=True)
    with open(pointer_file, "w", encoding="utf-8") as f:
        json.dump(pointer, f, indent=2)


def get_category_distribution(questions: List[Dict]) -> Dict[str, int]:
    """get_category_distribution helper."""
    counts = defaultdict(int)
    for q in questions:
        category = q.get("dataset", "unknown")
        counts[category] += 1
    return dict(counts)


def preview_sampling(
    questions: List[Dict], 
    k: int, 
    category_pointer: Optional[Dict[str, int]] = None
) -> None:
    """preview_sampling helper."""
    sorted_questions = create_permanent_order(questions)
    
    # Category-level sampling logic
    category_questions = defaultdict(list)
    for q in sorted_questions:
        category = q.get("dataset", "unknown")
        category_questions[category].append(q)
    
    # Persistent pointer for resumable sampling
    if category_pointer is None:
        category_pointer = {cat: 0 for cat in category_questions.keys()}
    
    print("\n=== Sampling Preview ===")
    print(f"Total questions: {len(sorted_questions)}")
    print(f"Number of categories: {len(category_questions)}")
    print(f"Samples per category: {k}")
    print()
    
    print("Category details:")
    print("-" * 60)
    print(f"{'Category':<20} {'Total':>8} {'Start':>8} {'End':>8} {'Selected':>10}")
    print("-" * 60)
    
    total_selected = 0
    for category in sorted(category_questions.keys()):
        questions_list = category_questions[category]
        start_idx = category_pointer.get(category, 0)
        end_idx = start_idx + k
        batch_size = min(k, len(questions_list) - start_idx)
        total_selected += batch_size
        
        print(f"{category:<20} {len(questions_list):>8} {start_idx:>8} {end_idx:>8} {batch_size:>10}")
    
    print("-" * 60)
    print(f"{'Total':<20} {len(sorted_questions):>8} {'':<8} {'':<8} {total_selected:>10}")
    print()
