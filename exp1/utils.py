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
    position_match = re.search(r'\[Position\]\s*[: ]?\s*([A-Z])', response, flags=re.IGNORECASE)
    if position_match:
        return position_match.group(1).upper()
    answer_patterns = [
        r'(?:answer|option|choice|final answer|selected option)\s*[: ]\s*([A-Z])',
        r'(?:final\s+)?(?:answer|option)\s+(?:is\s+)?([A-Z])(?:\s|\.|$|,)',
        r'\b([A-Z])\s*(?:\.|,|$)',
    ]
    for pattern in answer_patterns:
        match = re.search(pattern, response, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
    numbers = re.findall(r'\b(\d{1,2})\b', response)
    for num_str in numbers:
        idx = int(num_str) - 1
        if 0 <= idx <= 25:
            return chr(ord('A') + idx)
    all_letters = re.findall(r'\b([A-Z])\b', response)
    valid_options = [l for l in all_letters if l not in ('I', 'A')]
    if valid_options:
        return valid_options[-1].upper()

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
