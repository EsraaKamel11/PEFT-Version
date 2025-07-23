import os
import json
import logging
import random
from typing import List, Dict, Any
from openai import OpenAI
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

class QADatasetPreparer:
    def __init__(self, openai_api_key: str, domain: str = "general", output_dir: str = "prepared_data", model: str = "gpt-3.5-turbo"):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.client = OpenAI(api_key=openai_api_key)
        self.domain = domain
        self.output_dir = output_dir
        self.model = model
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_qa_pairs(self, texts: List[str], n_questions: int = 5) -> List[Dict[str, Any]]:
        qa_pairs = []
        for text in texts:
            prompt = (
                f"Based on this text about {self.domain}, generate {n_questions} diverse questions and answers. "
                "Include factual, reasoning, and comparison questions. "
                "Return as a JSON list of objects with 'question' and 'answer' fields.\n"
                f"Text: {text}"
            )
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=1024
                )
                content = response.choices[0].message.content
                # Try to extract JSON from the response
                qa_list = self._extract_json(content)
                for qa in qa_list:
                    qa_pairs.append({
                        "context": text,
                        "question": qa.get("question"),
                        "answer": qa.get("answer")
                    })
            except Exception as e:
                self.logger.error(f"OpenAI API error: {e}")
        return qa_pairs

    def _extract_json(self, content: str) -> List[Dict[str, str]]:
        import re
        import json
        # Try to find the first JSON array in the response
        match = re.search(r'\[.*\]', content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception as e:
                self.logger.error(f"Failed to parse JSON: {e}")
        # Fallback: try to parse the whole content
        try:
            return json.loads(content)
        except Exception as e:
            self.logger.error(f"Failed to parse fallback JSON: {e}")
        return []

    def format_for_hf(self, qa_pairs: List[Dict[str, Any]]) -> DatasetDict:
        if not qa_pairs:
            self.logger.warning("No QA pairs generated, creating sample data")
            # Create sample QA pairs for demonstration
            qa_pairs = [
                {"context": "EV charging", "question": "What is Level 2 charging?", "answer": "Level 2 charging uses 240V power."},
                {"context": "EV charging", "question": "How fast is DC charging?", "answer": "DC charging can provide 60-80% charge in 20-30 minutes."}
            ]
        
        df = pd.DataFrame(qa_pairs)
        if len(df) < 2:
            self.logger.warning("Not enough data for train/val split, using all data for training")
            train_dataset = Dataset.from_pandas(df.reset_index(drop=True))
            val_dataset = Dataset.from_pandas(df.reset_index(drop=True))  # Use same data for validation
        else:
            train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
            train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
            val_dataset = Dataset.from_pandas(val_df.reset_index(drop=True))
        return DatasetDict({"train": train_dataset, "validation": val_dataset})

    def format_alpaca(self, qa: dict) -> str:
        return f"### Instruction: {qa['question']}\n### Response: {qa['answer']}"

    def save_jsonl(self, dataset: Dataset, filename: str, alpaca_format: bool = False) -> None:
        path = os.path.join(self.output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            for item in dataset:
                if alpaca_format:
                    formatted = self.format_alpaca(item)
                    f.write(json.dumps({"text": formatted, "context": item.get("context", "")}, ensure_ascii=False) + "\n")
                else:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
        self.logger.info(f"Saved dataset to {path}")

    def prepare(self, texts: List[str], n_questions: int = 5, alpaca_format: bool = False) -> None:
        qa_pairs = self.generate_qa_pairs(texts, n_questions)
        datasets = self.format_for_hf(qa_pairs)
        self.save_jsonl(datasets["train"], "train.jsonl", alpaca_format=alpaca_format)
        self.save_jsonl(datasets["validation"], "validation.jsonl", alpaca_format=alpaca_format) 