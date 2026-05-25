# -*- coding: utf-8 -*-
"""Module helpers for the experiment pipeline."""

import re
from typing import Optional


def extract_answer_from_response(response: str) -> str:
    """extract_answer_from_response helper."""
    if not response:
        return ""

    response_lower = response.lower().strip()

    answer_markers = [
        "answer is", "conclusion is", "i think", "my opinion is",
        "therefore", "thus", "conclusion", "my answer is"
    ]

    for marker in answer_markers:
        if marker in response_lower:
            idx = response_lower.find(marker)
            answer = response[idx + len(marker):].strip()
            if "." in answer:
                answer = answer.split(".")[0]
            if len(answer) > 50:
                answer = answer[:50]
            if answer:
                return answer.strip()

    sentences = response.split(".")
    if sentences:
        last_sentence = sentences[-1].strip()
        if last_sentence:
            return last_sentence[:50].strip()

    return response[:30].strip()


def extract_answer_with_agent(response: str, question: str) -> str:
    """extract_answer_with_agent helper."""
    if not response:
        return ""

    # Strict extraction: only accept explicit answer patterns
    # Pattern 1: [Position] tag with letter
    position_match = re.search(r'\[Position\]\s*[: ]?\s*([A-Z])', response, flags=re.IGNORECASE)
    if position_match:
        return position_match.group(1).upper()

    # Pattern 2: Explicit answer patterns with clear markers
    answer_patterns = [
        r'(?:final\s+)?answer\s+is\s+([A-Z])(?:\s|\.|$|,)',
        r'(?:final\s+)?answer\s*[: ]\s*([A-Z])(?:\s|\.|$|,)',
        r'(?:final\s+)?conclusion\s+is\s+([A-Z])(?:\s|\.|$|,)',
        r'my\s+answer\s+is\s+([A-Z])(?:\s|\.|$|,)',
        r'the\s+(?:correct\s+)?answer\s+(?:is\s+)?([A-Z])(?:\s|\.|$|,)',
    ]
    for pattern in answer_patterns:
        match = re.search(pattern, response, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()

    # Pattern 3: Number conversion (1=A, 2=B, etc.)
    if re.search(r'\[Position\]\s*[: ]?\s*(\d+)', response):
        num_match = re.search(r'\[Position\]\s*[: ]?\s*(\d+)', response)
        if num_match:
            idx = int(num_match.group(1)) - 1
            if 0 <= idx <= 25:
                return chr(ord('A') + idx)

    # No valid answer found
    return ""


def extract_explicit_confidence(response: str) -> Optional[int]:
    """extract_explicit_confidence helper."""
    patterns = [
        r'\[Confidence\]\s*[: ]?\s*(\d+)',
        r'confidence[: ]\s*(\d+)',
        r' Confidence [: ]\s*(\d+)',
        r'\[Cc]onfidence\]\s*[: ]?\s*(\d+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, response)
        if match:
            value = int(match.group(1))
            if 1 <= value <= 100:
                return value
    return None
