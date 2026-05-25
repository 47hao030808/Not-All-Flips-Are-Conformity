# -*- coding: utf-8 -*-
"""Data Loader Module - Load questions from configured CSV datasets."""

import csv
import os
import random
from typing import List, Dict, Optional


class DataLoader:
    """Load questions from one or more CSV dataset files."""
    
    def __init__(self, data_dir: str = "data/demo", dataset_files: Optional[List[str]] = None):
        """
        Initialize data loader
        
        Args:
            data_dir: Dataset directory path
        """
        self.data_dir = data_dir
        self.dataset_files = dataset_files or ["demo_questions.csv"]
        self.all_questions: List[Dict] = []
        self._load_all_questions()
    
    def _load_all_questions(self):
        """Load all questions from professional datasets"""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(base_dir, self.data_dir)
        
        for dataset_file in self.dataset_files:
            file_path = os.path.join(data_path, dataset_file)
            if os.path.exists(file_path):
                questions = self._load_questions_from_file(file_path, dataset_file)
                self.all_questions.extend(questions)
                print(f"Loaded {len(questions)} questions from {dataset_file}")
            else:
                print(f"Warning: file does not exist {file_path}")
    
    def _load_questions_from_file(self, file_path: str, dataset_name: str) -> List[Dict]:
        """
        Load questions from a single CSV file.
        
        Args:
            file_path: CSV file path
            dataset_name: Dataset name
            
        Returns:
            List of questions
        """
        questions = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row_num, row in enumerate(reader, 1):
                    if not row or len(row) < 2:
                        continue
                    
                    # Expected format: question,optionA,optionB,optionC,optionD,correct_answer
                    question_text = row[0].strip()
                    if not question_text:
                        continue
                    
                    options = []
                    correct_answer = None
                    
                    for i in range(len(row) - 1, 0, -1):
                        cell = row[i].strip()
                        if cell and len(cell) == 1 and cell.upper() in ['A', 'B', 'C', 'D', 'E']:
                            correct_answer = cell.upper()
                            options = [row[j].strip() for j in range(1, i) if row[j].strip()]
                            break
                    
                    if not correct_answer and len(row) > 1:
                        last_cell = row[-1].strip()
                        if last_cell and len(last_cell) == 1 and last_cell.upper() in ['A', 'B', 'C', 'D', 'E']:
                            correct_answer = last_cell.upper()
                            options = [row[j].strip() for j in range(1, len(row) - 1) if row[j].strip()]
                        else:
                            options = [row[j].strip() for j in range(1, min(5, len(row))) if row[j].strip()]
                    
                    if len(options) < 2:
                        continue
                    
                    options = options[:4]
                    
                    if options:
                        options_text = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
                        formatted_question = f"{question_text}\n\nOptions:\n{options_text}"
                    else:
                        formatted_question = question_text
                    
                    questions.append({
                        "question": formatted_question,
                        "original_question": question_text,
                        "options": options,
                        "correct_answer": correct_answer,
                        "dataset": dataset_name,
                        "row_number": row_num
                    })
        
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            import traceback
            traceback.print_exc()
        
        return questions
    
    def get_random_question(self) -> Optional[Dict]:
        """
        Get a random question
        
        Returns:
            Question dictionary containing question, options, correct answer, etc.
        """
        if not self.all_questions:
            return None
        
        return random.choice(self.all_questions)
    
    def get_question_count(self) -> int:
        """
        Get total number of questions
        
        Returns:
            Total number of questions
        """
        return len(self.all_questions)
    
    def get_questions_by_dataset(self, dataset_name: str) -> List[Dict]:
        """
        Get all questions from a specific dataset
        
        Args:
            dataset_name: Dataset filename
            
        Returns:
            List of questions
        """
        return [q for q in self.all_questions if q["dataset"] == dataset_name]


def get_random_question_from_professional_datasets() -> Optional[str]:
    """
    Get a random question from professional datasets (returns formatted question text)
    
    Returns:
        Formatted question text, or None if loading fails
    """
    loader = DataLoader()
    question_data = loader.get_random_question()
    
    if question_data:
        return question_data["question"]
    return None

