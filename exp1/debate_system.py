# -*- coding: utf-8 -*-
"""
Debate System - Manages multi-agent debates
"""

import os
from typing import List, Dict, Optional, Tuple
from agent import Agent, SignalType
from api_client import APIClient, MODEL_EXTRACT, GLOBAL_SEED, calculate_answer_confidence, calculate_implicit_confidence, EXTRACT_BASE_URL, EXTRACT_API_KEY, MAX_RETRIES, INITIAL_RETRY_DELAY
from utils import extract_answer_with_agent as extract_answer_with_agent_simple, extract_answer_from_response
import re
import time
from openai import OpenAI

# Verbose mode for debugging
VERBOSE = os.getenv("VERBOSE", "false").lower() in ("true", "1", "yes")


def extract_answer_with_agent(response: str, question: str, api_client: APIClient) -> str:
    """
    Use the configured extraction model to extract the final answer from an agent response.

    This method uses an AI model to more accurately identify and extract the final answer,
    rather than simple text matching.
    """
    if not response:
        return ""

    system_prompt = (
        "You are an answer extraction expert. Your task is to extract the final answer from "
        "an agent's complete response.\n\n"
        "CRITICAL: Look for the answer after '[Position]' or '[Position]' tag in the response.\n"
        "Extract ONLY the letter option (A, B, C, D, E, F, G, H, I, J, etc.).\n\n"
        "Rules:\n"
        "1. Look for '[Position]' followed by a letter - extract that letter.\n"
        "2. Look for '[Position]' followed by a number (1-26) - convert to letter (1=A, 2=B, etc.).\n"
        "3. If no [Position] tag found, look for patterns like 'answer is X' where X is a single letter.\n"
        "4. Output ONLY the extracted letter/number, nothing else.\n"
        "5. If the response appears cut off or has no clear answer, output 'NO_ANSWER'."
    )

    user_prompt = f"""Question: {question}

Agent's complete response:
{response}

Extract the final answer letter from [Position] tag or similar patterns. Output only the letter."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    try:
        extract_client = OpenAI(api_key=EXTRACT_API_KEY, base_url=EXTRACT_BASE_URL)
        resp = extract_client.chat.completions.create(
            model=MODEL_EXTRACT,
            messages=messages,
            max_tokens=50,
            temperature=0
        )
        extracted_answer = resp.choices[0].message.content if resp.choices else ""
        return extracted_answer.strip()
    except Exception as e:
        print(f"Error extracting answer with the extraction model, falling back to simple method: {e}")
        return extract_answer_from_response(response)


class DebateSystem:
    """Multi-agent debate system"""

    def __init__(self, num_agents: int = 5, question_start_index: int = 1, round0_answers: dict = None, round0_reasoning: dict = None):
        """__init__ helper."""
        self.num_agents = num_agents
        self.api_client = APIClient()
        self.agents: List[Agent] = []
        self.debate_history: List[Dict] = []
        self.question_index = question_start_index
        self.current_correct_answer = ""
        # Round 0 data and bookkeeping
        self.round0_answers = round0_answers if round0_answers else {}
        # Round 0 data and bookkeeping
        self.round0_reasoning = round0_reasoning if round0_reasoning else {}
        # Round 1 data and bookkeeping
        self.round1_answers: Dict[str, Dict[int, str]] = {}

        # Initialize agents
        for i in range(num_agents):
            agent = Agent(agent_id=i + 1, api_client=self.api_client)
            self.agents.append(agent)

    def set_round0_answers(self, round0_answers: dict):
        """set_round0_answers helper."""
        self.round0_answers = round0_answers

    def set_round0_reasoning(self, round0_reasoning: dict):
        """set_round0_reasoning helper."""
        self.round0_reasoning = round0_reasoning

    def _get_round0_answer_for_agent(self, question_id: str, agent_id: int) -> Optional[str]:
        """_get_round0_answer_for_agent helper."""
        if question_id in self.round0_answers:
            agent_answers = self.round0_answers[question_id]
            if agent_id in agent_answers:
                return agent_answers[agent_id]
        return None

    def _get_round0_reasoning_for_agent(self, question_id: str, agent_id: int) -> Optional[str]:
        """_get_round0_reasoning_for_agent helper."""
        if question_id in self.round0_reasoning:
            agent_reasonings = self.round0_reasoning[question_id]
            if agent_id in agent_reasonings:
                return agent_reasonings[agent_id]
        return None

    def _extract_and_normalize_answer(self, response: str, question: str) -> str:
        """_extract_and_normalize_answer helper."""
        extracted = extract_answer_with_agent(response, question, self.api_client).strip()
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
        m = re.search(r'([A-Z])', extracted, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()
        numbers = re.findall(r'\b(\d{1,2})\b', extracted)
        for num_str in numbers:
            idx = int(num_str) - 1
            if 0 <= idx <= 25:
                return chr(ord('A') + idx)
        all_letters = re.findall(r'\b([A-Z])\b', response)
        valid_options = [l for l in all_letters if l not in ('I', 'A')]
        if valid_options:
            return valid_options[-1].upper()

        return extracted[:200].strip()

    def run_debate(self, question: str, num_rounds: int = 3, question_index: Optional[int] = None, question_id: str = None, correct_answer: Optional[str] = None, signal_type: SignalType = SignalType.REASONING, debate_round: int = None, max_retries_per_agent: int = 2) -> Dict:
        """
        Run a multi-agent debate

        Round setup:
        - Round 0: Each agent answers independently (no interaction)
        - Round 1-3: Interaction rounds based on signal_type

        Signal types:
        - SELF_REFLECTION: Question + own previous answer only, without peer information.
        - STANCE_ONLY: Agents see question + own previous answer + other agents' answers.
        - REASONING: Agents see question + own previous answer + other agents' full reasoning.

        Debate round control:
        - debate_round = None: Run all rounds (Round 0 + Round 1-n)
        - debate_round = 0: Run only Round 0 (independent thinking)
        - debate_round = 1: Run only Round 1 (interactive round, requires existing Round 0 data)

        Args:
            question: The question to debate
            num_rounds: Number of interaction rounds (default: 3, total rounds = 4)
            question_index: Question index; if None, uses self.question_index
            question_id: Question ID for looking up Round 0 answers
            correct_answer: The correct answer for comparison
            signal_type: Signal type for the experiment (SELF_REFLECTION, STANCE_ONLY, REASONING)
            debate_round: Which round to run (0 or 1), None to run all
            max_retries_per_agent: Maximum retries for failed agents per round

        Returns:
            Dictionary containing debate results
        """
        current_question_index = question_index if question_index is not None else self.question_index
        self.current_correct_answer = correct_answer if correct_answer else ""

        print(f"\n{'='*60}")
        print(f"Starting debate on question {current_question_index}: {question[:100]}...")
        print(f"Number of agents: {self.num_agents}, Number of rounds: {num_rounds}")
        print(f"Debate round to run: {debate_round if debate_round is not None else 'All rounds'}")
        if debate_round == 0:
            print("Signal type: N/A (independent thinking, no interaction)")
        else:
            print(f"Signal type: {signal_type.value} ({signal_type.name})")
        print(f"{'='*60}\n")

        all_round_results = []

        run_round_0 = (debate_round is None or debate_round == 0)
        run_round_1 = (debate_round is None or debate_round == 1)

        # Round 0 data and bookkeeping
        if run_round_0:
            print("Round 0: Initial responses (independent thinking)")
            print("-" * 60)
            initial_responses = []

            # Agent-level state or peer information
            failed_agents = set()
            for agent in self.agents:
                response = agent.generate_initial_response(question)
                initial_responses.append(response)
                # Agent-level state or peer information
                if not response.strip():
                    failed_agents.add(agent.agent_id)

            # Agent-level state or peer information
            retry_delay = INITIAL_RETRY_DELAY
            for retry in range(max_retries_per_agent):
                if not failed_agents:
                    break

                print(f"\n[Retry {retry + 1}/{max_retries_per_agent}] Retrying failed agents: {failed_agents}")
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 30)

                still_failed = set()
                for agent in self.agents:
                    if agent.agent_id in failed_agents:
                        print(f"[Agent {agent.agent_id}] Retrying initial response...")
                        response = agent.generate_initial_response(question)
                        if agent.history and response.strip():
                            agent.history[-1]["response"] = response
                            logprobs_data = agent.history[-1].get("logprobs_data")
                            if logprobs_data:
                                extracted_answer = agent.history[-1].get("extracted_answer", "")
                                if not extracted_answer:
                                    extracted_answer = self._extract_and_normalize_answer(response, question)
                                    agent.history[-1]["extracted_answer"] = extracted_answer
                                answer_conf = calculate_answer_confidence(logprobs_data, extracted_answer)
                                agent.history[-1]["answer_confidence"] = answer_conf
                                call_index = getattr(logprobs_data, 'call_index', None)
                                if call_index:
                                    APIClient.update_confidence_log(call_index, answer_conf)
                        idx = agent.agent_id - 1
                        if idx < len(initial_responses):
                            initial_responses[idx] = response

                        if not response.strip():
                            still_failed.add(agent.agent_id)

                failed_agents = still_failed
            if failed_agents:
                print(f"\nWarning: {len(failed_agents)} agent(s) still failed after retries: {failed_agents}")
            else:
                print(f"\nAll agents completed Round 0 successfully!")

            # Agent-level state or peer information
            for i, (agent, response) in enumerate(zip(self.agents, initial_responses)):
                # Compute or record implicit confidence from log probabilities
                logprobs_data = agent.history[-1].get("logprobs_data") if agent.history else None

                if VERBOSE:
                    print(f"Agent {agent.agent_id} response length: {len(response)} chars")
                if response.strip():
                    if VERBOSE:
                        print(f"Agent {agent.agent_id} full response (Round 0):\n{response}\n{'-'*40}")
                    extracted_answer = self._extract_and_normalize_answer(response, question)
                    if VERBOSE:
                        print(f"Agent {agent.agent_id} extracted answer: {extracted_answer}")

                    # Agent-level state or peer information
                    if agent.history:
                        agent.history[-1]["extracted_answer"] = extracted_answer
                    if logprobs_data:
                        answer_conf = calculate_answer_confidence(logprobs_data, extracted_answer)
                        if VERBOSE:
                            print(f"Agent {agent.agent_id} answer confidence (implicit): {answer_conf}")
                        agent.history[-1]["answer_confidence"] = answer_conf
                        call_index = agent.history[-1].get("call_index")
                        if call_index:
                            APIClient.update_confidence_log(call_index, answer_conf)
                else:
                    print(f"Agent {agent.agent_id} response is EMPTY (API call failed)")

            all_round_results.append({
                "round": 0,
                "responses": initial_responses,
                "failed_agents": list(failed_agents),
            })

            print()
            final_round_responses = initial_responses
        else:
            print("Warning: Running Round 1 without Round 0 data. Agents will start with empty previous opinion.")
            final_round_responses = []

        # Round 1 data and bookkeeping
        if run_round_1 and num_rounds >= 1:
            # Round 0 data and bookkeeping
            # Answer extraction or answer bookkeeping
            # Reasoning trace handling
            previous_opinions = []
            if question_id and question_id in self.round0_answers:
                # Round 0 data and bookkeeping
                agent_answers = self.round0_answers[question_id]
                agent_reasonings = self.round0_reasoning.get(question_id, {}) if self.round0_reasoning else {}
                for agent_id in range(1, self.num_agents + 1):
                    if agent_id in agent_answers:
                        previous_opinions.append({
                            "agent_id": agent_id,
                            "opinion": agent_answers[agent_id],
                            "reasoning": agent_reasonings.get(agent_id, "")
                        })
                print(f"Using external Round 0 data (question_id={question_id}):")
                print(f"  Answers: {agent_answers}")
            else:
                # Agent-level state or peer information
                for agent in self.agents:
                    opinion = agent.get_current_opinion()
                    reasoning = ""
                    if opinion:
                        previous_opinions.append({
                            "agent_id": agent.agent_id,
                            "opinion": opinion,
                            "reasoning": reasoning
                        })

            if not previous_opinions and debate_round == 1:
                print("Warning: No Round 0 data found. Skipping Round 1.")
                return {
                    "question": question,
                    "question_id": question_id,
                    "question_index": current_question_index,
                    "correct_answer": correct_answer,
                    "num_agents": self.num_agents,
                    "num_rounds": num_rounds,
                    "round_results": all_round_results,
                    "aggregation": None,
                    "warning": "Round 1 skipped due to missing Round 0 data",
                }

            print(f"Round 1: Debate with interaction")
            print("-" * 60)

            round_responses = []

            # Agent-level state or peer information
            failed_agents = set()
            for agent in self.agents:
                # Round 0 data and bookkeeping
                self_previous_opinion = None
                for op in previous_opinions:
                    if op["agent_id"] == agent.agent_id:
                        self_previous_opinion = op["opinion"]
                        break
                
                other_opinions = [op for op in previous_opinions if op["agent_id"] != agent.agent_id]

                response = agent.generate_debate_response(
                    question=question,
                    round_num=1,
                    self_previous_opinion=self_previous_opinion,
                    other_opinions=other_opinions,
                    signal_type=signal_type
                )
                round_responses.append(response)

                # Agent-level state or peer information
                if not response.strip():
                    failed_agents.add(agent.agent_id)

            # Agent-level state or peer information
            retry_delay = INITIAL_RETRY_DELAY
            for retry in range(max_retries_per_agent):
                if not failed_agents:
                    break

                print(f"\n[Round 1 Retry {retry + 1}/{max_retries_per_agent}] Retrying failed agents: {failed_agents}")
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 30)

                still_failed = set()
                for agent in self.agents:
                    if agent.agent_id in failed_agents:
                        print(f"[Agent {agent.agent_id}] Retrying debate response...")
                        # Round 0 data and bookkeeping
                        self_previous_opinion = None
                        for op in previous_opinions:
                            if op["agent_id"] == agent.agent_id:
                                self_previous_opinion = op["opinion"]
                                break
                        
                        other_opinions = [op for op in previous_opinions if op["agent_id"] != agent.agent_id]

                        response = agent.generate_debate_response(
                            question=question,
                            round_num=1,
                            self_previous_opinion=self_previous_opinion,
                            other_opinions=other_opinions,
                            signal_type=signal_type
                        )
                        if agent.history and response.strip():
                            agent.history[-1]["response"] = response
                            logprobs_data = agent.history[-1].get("logprobs_data")
                            if logprobs_data:
                                extracted_answer = agent.history[-1].get("extracted_answer", "")
                                if not extracted_answer:
                                    extracted_answer = self._extract_and_normalize_answer(response, question)
                                    agent.history[-1]["extracted_answer"] = extracted_answer
                                answer_conf = calculate_answer_confidence(logprobs_data, extracted_answer)
                                agent.history[-1]["answer_confidence"] = answer_conf
                                call_index = getattr(logprobs_data, 'call_index', None)
                                if call_index:
                                    APIClient.update_confidence_log(call_index, answer_conf)
                        idx = agent.agent_id - 1
                        if idx < len(round_responses):
                            round_responses[idx] = response

                        if not response.strip():
                            still_failed.add(agent.agent_id)

                failed_agents = still_failed
            if failed_agents:
                print(f"\nWarning: {len(failed_agents)} agent(s) still failed Round 1 after retries: {failed_agents}")
            else:
                print(f"\nAll agents completed Round 1 successfully!")

            # Agent-level state or peer information
            for i, (agent, response) in enumerate(zip(self.agents, round_responses)):
                logprobs_data = agent.history[-1].get("logprobs_data") if agent.history else None

                if VERBOSE:
                    print(f"Agent {agent.agent_id} response length: {len(response)} chars")
                if response.strip():
                    if VERBOSE:
                        print(f"Agent {agent.agent_id} full response (Round 1):\n{response}\n{'-'*40}")
                    extracted_answer = self._extract_and_normalize_answer(response, question)
                    if VERBOSE:
                        print(f"Agent {agent.agent_id} extracted answer: {extracted_answer}")

                    # Agent-level state or peer information
                    if agent.history:
                        agent.history[-1]["extracted_answer"] = extracted_answer
                    if logprobs_data:
                        answer_conf = calculate_answer_confidence(logprobs_data, extracted_answer)
                        if VERBOSE:
                            print(f"Agent {agent.agent_id} answer confidence (implicit): {answer_conf}")
                        agent.history[-1]["answer_confidence"] = answer_conf
                        call_index = agent.history[-1].get("call_index")
                        if call_index:
                            APIClient.update_confidence_log(call_index, answer_conf)

                    # Round 1 data and bookkeeping
                    if question_id:
                        if question_id not in self.round1_answers:
                            self.round1_answers[question_id] = {}
                        self.round1_answers[question_id][agent.agent_id] = extracted_answer
                else:
                    print(f"Agent {agent.agent_id} response is EMPTY (API call failed)")

            all_round_results.append({
                "round": 1,
                "responses": round_responses,
                "failed_agents": list(failed_agents),
            })
            final_round_responses = round_responses

            print()
        if debate_round == 0:
            print("Round 0 completed. Skipping remaining rounds as configured.")
            return {
                "question": question,
                "question_index": current_question_index,
                "correct_answer": correct_answer,
                "num_agents": self.num_agents,
                "num_rounds": num_rounds,
                "round_results": all_round_results,
                "aggregation": None,
                "note": "Only Round 0 was run as configured",
            }

        # Summary
        print(f"\n{'='*60}")
        print("Debate Summary")
        print(f"{'='*60}")
        print(f"{'Round':<10} {'Responses':<50}")
        print("-" * 60)

        for round_data in all_round_results:
            round_num = round_data["round"]
            num_responses = len(round_data["responses"])
            print(f"{round_num:<10} {num_responses} agents responded")

        return {
            "question": question,
            "question_index": current_question_index,
            "correct_answer": correct_answer,
            "num_agents": self.num_agents,
            "num_rounds": num_rounds,
            "round_results": all_round_results,
        }
