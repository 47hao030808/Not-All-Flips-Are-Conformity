# -*- coding: utf-8 -*-
"""
API Client - Handles interaction with a configured LLM API
"""

import json
import os
import math
import re
from typing import List, Dict, Tuple, Optional
from openai import OpenAI

API_KEY = os.getenv("ANSWER_API_KEY")
EXTRACT_API_KEY = os.getenv("EXTRACT_API_KEY")
BASE_URL = os.getenv("ANSWER_BASE_URL")
MODEL_ANSWER = os.getenv("ANSWER_MODEL")
EXTRACT_BASE_URL = os.getenv("EXTRACT_BASE_URL")
MODEL_EXTRACT = os.getenv("EXTRACT_MODEL")
GLOBAL_SEED = 42
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 2
MAX_RETRY_DELAY = 30
RETRY_MULTIPLIER = 2


def calculate_implicit_confidence(logprobs_data, answer: str = None) -> Optional[float]:
    """calculate_implicit_confidence helper."""
    if logprobs_data is None:
        return None
    if answer is None:
        return None

    # Answer extraction or answer bookkeeping
    # Compute or record implicit confidence from log probabilities
    if isinstance(logprobs_data, str):
        return None

    try:
        answer_upper = answer.upper().strip()
        if not answer_upper:
            return None
        confidence_value = None
        content = None
        if hasattr(logprobs_data, 'content'):
            content = logprobs_data.content

        # Compute or record implicit confidence from log probabilities
        top_logprobs = None
        if hasattr(logprobs_data, 'top_logprobs'):
            top_logprobs = logprobs_data.top_logprobs

        # Compute or record implicit confidence from log probabilities
        token_logprobs = None
        if hasattr(logprobs_data, 'token_logprobs'):
            token_logprobs = logprobs_data.token_logprobs
        if confidence_value is None and content is not None:
            if isinstance(content, list):
                for i, token_obj in enumerate(content):
                    if hasattr(token_obj, 'token') and hasattr(token_obj, 'logprob'):
                        token = token_obj.token
                        logprob = token_obj.logprob
                        token_stripped = token.strip().upper()
                        if token_stripped == answer_upper:
                            confidence_value = math.exp(logprob) if logprob <= 0 else None
                            break
                        elif token_stripped.startswith(answer_upper) or (answer_upper and token_stripped.startswith(' ' + answer_upper)):
                            if token_stripped.strip() == answer_upper:
                                confidence_value = math.exp(logprob) if logprob <= 0 else None
                                break
                if confidence_value is None and len(content) > 0:
                    last_tokens = content[-5:]
                    for token_obj in last_tokens:
                        if hasattr(token_obj, 'token') and hasattr(token_obj, 'logprob'):
                            token = token_obj.token
                            logprob = token_obj.logprob
                            token_stripped = token.strip().upper()
                            if answer_upper in token_stripped or token_stripped in answer_upper:
                                confidence_value = math.exp(logprob) if logprob <= 0 else None
                                break

        # Compute or record implicit confidence from log probabilities
        if hasattr(logprobs_data, 'confidence'):
            confidence_value = logprobs_data.confidence
        elif hasattr(logprobs_data, 'answer_confidence'):
            confidence_value = logprobs_data.answer_confidence

        # Compute or record implicit confidence from log probabilities
        if confidence_value is None and token_logprobs is not None:
            if isinstance(token_logprobs, list):
                for i, logprob_dict in enumerate(token_logprobs):
                    if isinstance(logprob_dict, dict):
                        for key, value in logprob_dict.items():
                            if key.strip().upper() == answer_upper:
                                confidence_value = math.exp(value) if value < 0 else None
                                break
                    elif isinstance(logprob_dict, (int, float)):
                        confidence_value = math.exp(logprob_dict) if logprob_dict < 0 else None
                    if confidence_value is not None:
                        break

        # Compute or record implicit confidence from log probabilities
        if confidence_value is None and token_logprobs is not None:
            if isinstance(token_logprobs, list) and len(token_logprobs) > 0:
                last_entry = token_logprobs[-1]
                if isinstance(last_entry, dict):
                    for key, value in last_entry.items():
                        if key.strip().upper() == answer_upper:
                            confidence_value = math.exp(value) if value < 0 else None
                            break
                elif isinstance(last_entry, (int, float)):
                    confidence_value = math.exp(last_entry) if last_entry < 0 else None

        # Compute or record implicit confidence from log probabilities
        # 1. list of dict: [{"token": " A", "logprob": -0.001}, ...]
        # 2. list of TopLogprob objects with .token and .logprob attributes
        # 3. list of dict: [{" A": -0.001}, ...]
        if confidence_value is None and top_logprobs is not None:
            if isinstance(top_logprobs, list) and len(top_logprobs) > 0:
                for idx, entry in enumerate(top_logprobs):
                    if isinstance(entry, dict):
                        if 'token' in entry and 'logprob' in entry:
                            token = entry['token']
                            logprob = entry['logprob']
                            token_stripped = token.strip().upper()
                            if token_stripped == answer_upper or token_stripped.startswith(answer_upper + ' ') or token_stripped.startswith(answer_upper + ','):
                                confidence_value = math.exp(logprob) if logprob <= 0 else None
                                break
                        else:
                            for key, value in entry.items():
                                key_stripped = key.strip().upper()
                                if key_stripped == answer_upper or key_stripped.startswith(answer_upper):
                                    confidence_value = math.exp(value) if value < 0 else None
                                    break
                    elif hasattr(entry, 'token') and hasattr(entry, 'logprob'):
                        # TopLogprob object format: entry.token and entry.logprob
                        token = entry.token
                        logprob = entry.logprob
                        token_stripped = token.strip().upper()
                        if token_stripped == answer_upper or token_stripped.startswith(answer_upper + ' ') or token_stripped.startswith(answer_upper + ','):
                            confidence_value = math.exp(logprob) if logprob <= 0 else None
                            break
                    elif hasattr(entry, 'top_logprobs'):
                        nested_top_logprobs = entry.top_logprobs
                        if isinstance(nested_top_logprobs, list):
                            for nested_entry in nested_top_logprobs:
                                if isinstance(nested_entry, dict):
                                    if 'token' in nested_entry and 'logprob' in nested_entry:
                                        token = nested_entry['token']
                                        logprob = nested_entry['logprob']
                                        token_stripped = token.strip().upper()
                                        if token_stripped == answer_upper or token_stripped.startswith(answer_upper):
                                            confidence_value = math.exp(logprob) if logprob <= 0 else None
                                            break
                                elif hasattr(nested_entry, 'token') and hasattr(nested_entry, 'logprob'):
                                    token = nested_entry.token
                                    logprob = nested_entry.logprob
                                    token_stripped = token.strip().upper()
                                    if token_stripped == answer_upper or token_stripped.startswith(answer_upper):
                                        confidence_value = math.exp(logprob) if logprob <= 0 else None
                                        break
                                if confidence_value is not None:
                                    break
                    if confidence_value is not None:
                        break
        if confidence_value is None and hasattr(logprobs_data, 'raw_data'):
            raw_data = logprobs_data.raw_data
            if isinstance(raw_data, dict):
                # Compute or record implicit confidence from log probabilities
                logprobs = raw_data.get('logprobs', {})
                if isinstance(logprobs, dict):
                    token_logprobs = logprobs.get('token_logprobs', [])
                    if isinstance(token_logprobs, list) and len(token_logprobs) > 0:
                        for entry in token_logprobs:
                            if isinstance(entry, dict):
                                for key, value in entry.items():
                                    if key.strip().upper() == answer_upper:
                                        confidence_value = math.exp(value) if value < 0 else None
                                        break
                            if confidence_value is not None:
                                break
        if confidence_value is None and isinstance(logprobs_data, dict):
            # Compute or record implicit confidence from log probabilities
            token_logprobs = logprobs_data.get('token_logprobs', [])
            if isinstance(token_logprobs, list) and len(token_logprobs) > 0:
                for entry in token_logprobs:
                    if isinstance(entry, dict):
                        for key, value in entry.items():
                            if key.strip().upper() == answer_upper:
                                confidence_value = math.exp(value) if value < 0 else None
                                break
                    if confidence_value is not None:
                        break

        # Compute or record implicit confidence from log probabilities
        if confidence_value is None and hasattr(logprobs_data, 'content'):
            content = logprobs_data.content
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        # Compute or record implicit confidence from log probabilities
                        if 'top_logprobs' in item:
                            top_logprobs = item['top_logprobs']
                            if isinstance(top_logprobs, list):
                                for entry in top_logprobs:
                                    if isinstance(entry, dict):
                                        if 'token' in entry and 'logprob' in entry:
                                            token = entry['token']
                                            logprob = entry['logprob']
                                            token_stripped = token.strip().upper()
                                            if token_stripped == answer_upper or token_stripped.startswith(answer_upper):
                                                confidence_value = math.exp(logprob) if logprob <= 0 else None
                                                break
                    if confidence_value is not None:
                        break

        if confidence_value is not None:
            return min(max(confidence_value, 0.0), 1.0)

        return None

    except Exception as e:
        print(f"[calculate_implicit_confidence] Exception: {type(e).__name__}: {e}")
        return None


# Answer extraction or answer bookkeeping
calculate_answer_confidence = calculate_implicit_confidence


class APIClient:
    """API Client for both answering and extracting"""
    FINGERPRINT_LOG_FILE = None
    CONFIDENCE_LOG_FILE = None

    def __init__(self):
        missing = [
            name
            for name, value in {
                "ANSWER_API_KEY": API_KEY,
                "ANSWER_BASE_URL": BASE_URL,
                "ANSWER_MODEL": MODEL_ANSWER,
                "EXTRACT_API_KEY": EXTRACT_API_KEY,
                "EXTRACT_BASE_URL": EXTRACT_BASE_URL,
                "EXTRACT_MODEL": MODEL_EXTRACT,
            }.items()
            if not value
        ]
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
        self.client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        self.extract_client = OpenAI(api_key=EXTRACT_API_KEY, base_url=EXTRACT_BASE_URL)
        self.MAX_RETRIES = MAX_RETRIES
        self.INITIAL_RETRY_DELAY = INITIAL_RETRY_DELAY
        self.MAX_RETRY_DELAY = MAX_RETRY_DELAY
        self.RETRY_MULTIPLIER = RETRY_MULTIPLIER

    @classmethod
    def set_fingerprint_log_path(cls, log_dir: str):
        """set_fingerprint_log_path helper."""
        os.makedirs(log_dir, exist_ok=True)
        cls.FINGERPRINT_LOG_FILE = os.path.join(log_dir, "fingerprint_log.jsonl")
        cls.CONFIDENCE_LOG_FILE = os.path.join(log_dir, "confidence_log.jsonl")

    @classmethod
    def log_fingerprint(cls, call_index: int, system_fingerprint: Optional[str], model: str, seed: Optional[int], temperature: float):
        """log_fingerprint helper."""
        if cls.FINGERPRINT_LOG_FILE is None:
            return

        log_entry = {
            "call_index": call_index,
            "system_fingerprint": system_fingerprint,
            "model": model,
            "seed": seed,
            "temperature": temperature,
        }

        with open(cls.FINGERPRINT_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    @classmethod
    def log_confidence(cls, call_index: int, confidence: Optional[float], model: str, seed: Optional[int], temperature: float):
        """log_confidence helper."""
        if cls.CONFIDENCE_LOG_FILE is None:
            return

        log_entry = {
            "call_index": call_index,
            "implicit_confidence": confidence,
            "model": model,
            "seed": seed,
            "temperature": temperature,
        }

        with open(cls.CONFIDENCE_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    @classmethod
    def update_confidence_log(cls, call_index: int, new_confidence: Optional[float]):
        """update_confidence_log helper."""
        if cls.CONFIDENCE_LOG_FILE is None or not os.path.exists(cls.CONFIDENCE_LOG_FILE):
            return
        entries = []
        with open(cls.CONFIDENCE_LOG_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    entries.append(entry)
                except json.JSONDecodeError:
                    continue
        for entry in entries:
            if entry.get("call_index") == call_index:
                entry["implicit_confidence"] = new_confidence
                break
        with open(cls.CONFIDENCE_LOG_FILE, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def chat(self, model: str, messages: List[Dict], max_tokens: int = 2000, temperature: float = 1.0, seed: Optional[int] = None, return_logprobs: bool = False) -> Tuple[str, int, Optional[str], Optional[float], Optional[object]]:
        """
        Call chat API with automatic retry on failure

        Args:
            model: Model name
            messages: Message list
            max_tokens: Maximum token count
            temperature: Temperature parameter
            seed: Random seed for reproducibility (requires temperature=0 for determinism)
            return_logprobs: If True, also return the raw logprobs data object

        Returns:
            (response_text, tokens_used, system_fingerprint, implicit_confidence, logprobs_data)
            - For non-answer-extraction calls: all five values are returned
            - For answer-extraction calls (max_tokens <= 500): logprobs_data is None (not logged)
        """
        retry_count = 0
        retry_delay = self.INITIAL_RETRY_DELAY

        while True:
            try:
                # Only print detailed info for non-answer-extraction scenarios
                if max_tokens > 500:  # Answer extraction uses smaller max_tokens
                    print(f"Calling API: model={model}, max_tokens={max_tokens}, temperature={temperature}, seed={seed}")

                resp = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    seed=seed,
                    logprobs=True,
                    top_logprobs=1
                )

                msg = resp.choices[0].message
                response_text = msg.content if hasattr(msg, 'content') else ""

                # Calculate tokens used
                tokens_used = resp.usage.total_tokens if hasattr(resp, 'usage') and resp.usage else 0

                # Get system fingerprint
                system_fingerprint = getattr(resp, 'system_fingerprint', None)

                # Get logprobs and calculate implicit confidence
                logprobs_data = getattr(resp.choices[0], 'logprobs', None)
                # Agent-level state or peer information
                implicit_confidence = None

                # Log to files (only for main calls, not answer extraction)
                if max_tokens > 500:
                    call_index = getattr(self, '_api_call_counter', 0) + 1
                    self._api_call_counter = call_index
                    self.log_fingerprint(call_index, system_fingerprint, model, seed, temperature)
                    # Agent-level state or peer information
                    self.log_confidence(call_index, None, model, seed, temperature)
                    # Compute or record implicit confidence from log probabilities
                    if logprobs_data is not None:
                        setattr(logprobs_data, 'call_index', call_index)

                # Only print detailed info for non-answer-extraction scenarios
                if max_tokens > 500:
                    print(f"API response successful: tokens={tokens_used}, fingerprint={system_fingerprint}, confidence={implicit_confidence}, response_length={len(response_text)}")

                # Return logprobs_data only for main calls, not answer extraction
                return response_text, tokens_used, system_fingerprint, implicit_confidence, logprobs_data if max_tokens > 500 else None

            except Exception as e:
                error_str = str(e)
                retry_count += 1
                should_retry = (
                    retry_count <= self.MAX_RETRIES and (
                        "502" in error_str or
                        "503" in error_str or
                        "504" in error_str or
                        "429" in error_str or
                        "rate limit" in error_str.lower() or
                        "timeout" in error_str.lower() or
                        "connection" in error_str.lower() or
                        "network" in error_str.lower() or
                        "Bad Gateway" in error_str or
                        "Service Unavailable" in error_str or
                        "html" in error_str.lower()
                    )
                )

                if should_retry:
                    print(f"API call failed (attempt {retry_count}/{self.MAX_RETRIES}): {e}")
                    print(f"Retrying in {retry_delay} seconds...")
                    import time
                    time.sleep(retry_delay)
                    retry_delay = min(retry_delay * self.RETRY_MULTIPLIER, self.MAX_RETRY_DELAY)
                    continue
                else:
                    print(f"API call failed: model={model}, error={e}")
                    print(f"(No more retries after {retry_count - 1} attempts)")
                    return "", 0, None, None, None
