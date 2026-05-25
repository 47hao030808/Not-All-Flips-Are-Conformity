# -*- coding: utf-8 -*-
"""
Agent - Represents a single agent in the multi-agent debate system
"""

import os
import time
from typing import List, Dict, Optional, Tuple
from enum import Enum
from api_client import APIClient, MODEL_ANSWER, GLOBAL_SEED, MAX_RETRIES, INITIAL_RETRY_DELAY, calculate_implicit_confidence
from utils import extract_answer_with_agent, extract_explicit_confidence

# Verbose mode for debugging
VERBOSE = os.getenv("VERBOSE", "false").lower() in ("true", "1", "yes")


class SignalType(Enum):
    """Signal types for controlled round-1 conditions."""
    SELF_REFLECTION = "self_reflection"
    STANCE_ONLY = "stance_only"
    REASONING = "reasoning"
    INVALID_REASONING = "invalid_reasoning"
    WRONG_REASONING = "wrong_reasoning"


class Agent:
    """Single agent in the debate system"""

    def __init__(self, agent_id: int, api_client: APIClient):
        """
        Initialize an agent

        Args:
            agent_id: Unique identifier for the agent
            api_client: API client for making requests
        """
        self.agent_id = agent_id
        self.api_client = api_client
        self.history: List[Dict] = []  # Debate history
        self.current_opinion: Optional[str] = None
        self.extracted_answer: Optional[str] = None

    def _has_shared_peer_reasoning(self, other_opinions: List[Dict]) -> bool:
        answers = {
            str(op.get("opinion", "")).strip()
            for op in other_opinions
            if str(op.get("opinion", "")).strip()
        }
        reasonings = {
            str(op.get("reasoning", "")).strip()
            for op in other_opinions
            if str(op.get("reasoning", "")).strip()
        }
        return len(answers) == 1 and len(reasonings) == 1

    def get_system_prompt(self) -> str:
        """Get system prompt for Round 0 (independent thinking, no reference to other agents)"""
        return (
            "You are an expert with professional knowledge in various fields. "
            "Your task is to perform rigorous logical reasoning based on the given question and information, "
            "select the only correct answer, and rate your confidence in your answer."
        )

    def get_debate_system_prompt(self) -> str:
        """Get system prompt for debate rounds (Round 1-3, can consider other agents' opinions)"""
        return (
            "You are an expert with professional knowledge in various fields. "
            "Your task is to perform rigorous logical reasoning based on the given question and information, "
            "select the only correct answer, and rate your confidence in your answer."
        )

    def _api_call_with_retry(self, model: str, messages: List[Dict], max_tokens: int = 2000, temperature: float = 1.0, seed: Optional[int] = None) -> Tuple[str, int, Optional[str], Optional[float], Optional[object]]:
        """
        Make API call with agent-level retry on failure (supplementary to APIClient retry)

        This provides an additional layer of retry specifically for each agent,
        ensuring that if one agent's call fails, we can retry that specific agent
        without affecting other agents.

        Args:
            model: Model name
            messages: Message list
            max_tokens: Maximum token count
            temperature: Temperature parameter
            seed: Random seed for reproducibility

        Returns:
            Same as APIClient.chat()
        """
        retry_delay = INITIAL_RETRY_DELAY
        last_error = None

        for attempt in range(MAX_RETRIES):
            try:
                response, tokens, fingerprint, confidence, logprobs_data = self.api_client.chat(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    seed=seed
                )
                if response:
                    return response, tokens, fingerprint, confidence, logprobs_data
                if max_tokens > 500 and attempt < MAX_RETRIES - 1:
                    last_error = "Empty response received"
                    print(f"[Agent {self.agent_id}] Empty response (attempt {attempt + 1}/{MAX_RETRIES}), retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue

                return response, tokens, fingerprint, confidence, logprobs_data

            except Exception as e:
                last_error = e
                if attempt < MAX_RETRIES - 1:
                    print(f"[Agent {self.agent_id}] API call exception (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
                    print(f"[Agent {self.agent_id}] Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
        print(f"[Agent {self.agent_id}] API call failed after {MAX_RETRIES} attempts. Last error: {last_error}")
        return "", 0, None, None, None

    def generate_initial_response(self, question: str) -> str:
        """
        Generate initial response to a question

        Args:
            question: The question to answer

        Returns:
            Agent's response
        """
        messages = [
            {"role": "system", "content": self.get_system_prompt()},
            {"role": "user", "content": (
                f"Please solve the following question:\n"
                f"{question}\n\n"
                f"Please answer in the following format:\n"
                f"[Reasoning]\n"
                f"[Position]\n"
                f"[Confidence]: (Please rate your certainty about this answer from 0-100, "
                f"where 0 means completely uncertain and 100 means completely certain)"
            )}
        ]

        if VERBOSE:
            print(f"\n[Agent {self.agent_id}] Initial round prompt:")
            for msg in messages:
                print(f"  {msg['role']}: {msg['content']}")

        response, tokens, fingerprint, confidence, logprobs_data = self._api_call_with_retry(
            model=MODEL_ANSWER,
            messages=messages,
            max_tokens=2000,
            temperature=0,
            seed=GLOBAL_SEED
        )

        if VERBOSE:
            print(f"[Agent {self.agent_id}] Initial round response (tokens={tokens}, fingerprint={fingerprint}, confidence={confidence}):\n{response}\n{'-'*40}")

        self.current_opinion = response
        explicit_conf = extract_explicit_confidence(response)
        # Compute or record implicit confidence from log probabilities
        call_index = getattr(logprobs_data, 'call_index', None) if logprobs_data else None
        self.extracted_answer = extract_answer_with_agent(response, "").strip() if response.strip() else None
        implicit_conf = calculate_implicit_confidence(logprobs_data, self.extracted_answer) if logprobs_data and self.extracted_answer else None
        if VERBOSE:
            print(f"[DEBUG] Agent {self.agent_id} - extracted_answer: '{self.extracted_answer}', implicit_conf: {implicit_conf}, call_index: {call_index}")
        if call_index is not None and implicit_conf is not None:
            APIClient.update_confidence_log(call_index, implicit_conf)
            if VERBOSE:
                print(f"[DEBUG] Agent {self.agent_id} - Updated confidence log with value: {implicit_conf}")

        self.history.append({
            "round": 0,
            "type": "initial",
            "question": question,
            "response": response,
            "tokens": tokens,
            "confidence": implicit_conf,
            "answer_confidence": implicit_conf,
            "logprobs_data": logprobs_data,
            "explicit_confidence": explicit_conf,
            "call_index": call_index,
            "extracted_answer": self.extracted_answer
        })

        return response

    def generate_debate_response(self, question: str, round_num: int, self_previous_opinion: Optional[str], other_opinions: List[Dict], signal_type: SignalType = SignalType.REASONING) -> str:
        """
        Generate response in a debate round after seeing own previous opinion and other agents' opinions

        Signal types:
        - SELF_REFLECTION: Question + own previous stance only (no other agents' info)
        - STANCE_ONLY: Question + own previous answer + other agents' answers
        - REASONING: Question + own previous answer + other agents' answers + other agents' full reasoning

        Args:
            question: The original question
            round_num: Current round number
            self_previous_opinion: Agent's own previous round opinion
            other_opinions: List of other agents' opinions in format [{"agent_id": int, "opinion": str}]
            signal_type: Signal type (SELF_REFLECTION, STANCE_ONLY, REASONING)

        Returns:
            Agent's response
        """
        question_part = f"Question: {question}"

        # Signal-type specific behavior
        answer_format = (
            f"\nPlease answer in the following format:\n"
            f"[Reasoning]\n"
            f"[Position]\n"
            f"[Confidence]: (Please rate your certainty about this answer from 0-100, "
            f"where 0 means completely uncertain and 100 means completely certain)"
        )
        if signal_type is SignalType.SELF_REFLECTION:
            # Agent-level state or peer information
            agent_n_answer_part = (
                f"Your previous round answer: "
                f"{self_previous_opinion if self_previous_opinion else 'None'}"
            )

            user_content = f"""Please solve the following question:

{question_part}

{agent_n_answer_part}

Please reconsider this question based on the above information.
{answer_format}"""
            signal_type_name = "Self-Reflection"

        elif signal_type is SignalType.STANCE_ONLY:
            # Agent-level state or peer information
            agent_n_answer_part = (
                f"Your previous round answer: "
                f"{self_previous_opinion if self_previous_opinion else 'None'}"
            )

            if other_opinions and self._has_shared_peer_reasoning(other_opinions):
                shared_answer = str(other_opinions[0].get("opinion", "")).strip()
                shared_reasoning = str(other_opinions[0].get("reasoning", "")).strip()
                other_agents_part = (
                    "Other agents' answers:\n"
                    f"All other agents' answer: {shared_answer}"
                )
                other_agents_reasoning_part = (
                    f"\n\nOther agents' reasoning:\n{shared_reasoning}"
                )
            elif other_opinions:
                other_agents_text = "\n".join([
                    f"Agent {op['agent_id']}'s answer: {op['opinion']}"
                    for op in other_opinions
                ])
                other_agents_part = f"Other agents' answers:\n{other_agents_text}"
            else:
                other_agents_part = "Other agents' answers: None"

            user_content = f"""Please solve the following question:

{question_part}

{agent_n_answer_part}

{other_agents_part}

Please reconsider this question based on the above information.
{answer_format}"""
            signal_type_name = "Stance-Only"

        else:  # SignalType.REASONING
            # Agent-level state or peer information
            agent_n_answer_part = (
                f"Your previous round answer: "
                f"{self_previous_opinion if self_previous_opinion else 'None'}"
            )

            if other_opinions and self._has_shared_peer_reasoning(other_opinions):
                shared_reasoning = str(other_opinions[0].get("reasoning", "")).strip()
                other_agents_answers_text = "\n".join([
                    f"Agent {op['agent_id']}'s answer: {op['opinion']}"
                    for op in other_opinions
                ])
                other_agents_answers_part = (
                    "Other agents' answers:\n"
                    f"{other_agents_answers_text}"
                )
                other_agents_reasoning_part = (
                    f"\n\nOther agents' reasoning:\n{shared_reasoning}"
                )
            elif other_opinions:
                # Agent-level state or peer information
                other_agents_answers_text = "\n".join([
                    f"Agent {op['agent_id']}'s answer: {op['opinion']}"
                    for op in other_opinions
                ])
                other_agents_answers_part = f"Other agents' answers:\n{other_agents_answers_text}"
                # Round 0 data and bookkeeping
                other_agents_reasoning_text = "\n\n".join([
                    f"Agent {op['agent_id']}'s reasoning:\n{op.get('reasoning', op['opinion'])}"
                    for op in other_opinions
                ])
                other_agents_reasoning_part = f"\n\nOther agents' reasoning:\n{other_agents_reasoning_text}"
            else:
                other_agents_answers_part = "Other agents' answers: None"
                other_agents_reasoning_part = ""

            user_content = f"""Please solve the following question:

{question_part}

{agent_n_answer_part}

{other_agents_answers_part}{other_agents_reasoning_part}

Please reconsider this question based on the above information.
{answer_format}"""
            signal_type_name = {
                SignalType.REASONING: "Full Reasoning",
                SignalType.WRONG_REASONING: "Wrong Reasoning",
                SignalType.INVALID_REASONING: "Invalid Reasoning",
            }.get(signal_type, "Full Reasoning")

        messages = [
            {"role": "system", "content": self.get_debate_system_prompt()},
            {"role": "user", "content": user_content}
        ]

        if VERBOSE:
            print(f"\n[Agent {self.agent_id}] Round {round_num} prompt (Signal Type: {signal_type_name}):")
            for msg in messages:
                print(f"  {msg['role']}: {msg['content']}")

        response, tokens, fingerprint, confidence, logprobs_data = self._api_call_with_retry(
            model=MODEL_ANSWER,
            messages=messages,
            max_tokens=2000,
            temperature=0,
            seed=GLOBAL_SEED
        )

        if VERBOSE:
            print(f"[Agent {self.agent_id}] Round {round_num} response (tokens={tokens}, fingerprint={fingerprint}, confidence={confidence}):\n{response}\n{'-'*40}")

        self.current_opinion = response
        explicit_conf = extract_explicit_confidence(response)
        # Compute or record implicit confidence from log probabilities
        call_index = getattr(logprobs_data, 'call_index', None) if logprobs_data else None
        self.extracted_answer = extract_answer_with_agent(response, "").strip() if response.strip() else None
        implicit_conf = calculate_implicit_confidence(logprobs_data, self.extracted_answer) if logprobs_data and self.extracted_answer else None
        if call_index is not None and implicit_conf is not None:
            APIClient.update_confidence_log(call_index, implicit_conf)

        self.history.append({
            "round": round_num,
            "type": "debate",
            "question": question,
            "self_previous_opinion": self_previous_opinion,
            "other_opinions": other_opinions,
            "response": response,
            "tokens": tokens,
            "confidence": implicit_conf,
            "answer_confidence": implicit_conf,
            "logprobs_data": logprobs_data,
            "explicit_confidence": explicit_conf,
            "call_index": call_index
        })

        return response

    def get_current_opinion(self) -> Optional[str]:
        """Get current opinion of the agent"""
        return self.current_opinion

    def get_extracted_answer(self) -> Optional[str]:
        """Get extracted answer of the agent (single option like A, B, C)"""
        return self.extracted_answer
