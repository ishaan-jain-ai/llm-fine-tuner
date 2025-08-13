"""
Data processing module for LLM Fine Tuner Agent
"""
import json
import csv
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from datasets import Dataset, DatasetDict, load_dataset
from transformers import PreTrainedTokenizer
import logging

logger = logging.getLogger(__name__)


class DataProcessor:
    """Handles data loading, preprocessing, and formatting for fine-tuning"""
    
    def __init__(self, config):
        self.config = config
        self.tokenizer = None
    
    def set_tokenizer(self, tokenizer: PreTrainedTokenizer):
        """Set the tokenizer for data processing"""
        self.tokenizer = tokenizer
    
    def load_data(self, file_path: str) -> Dataset:
        """Load data from various file formats"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        file_extension = file_path.suffix.lower()
        
        if file_extension == '.json':
            return self._load_json(file_path)
        elif file_extension == '.csv':
            return self._load_csv(file_path)
        elif file_extension == '.txt':
            return self._load_txt(file_path)
        elif file_extension == '.jsonl':
            return self._load_jsonl(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
    
    def _load_json(self, file_path: Path) -> Dataset:
        """Load JSON data"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            return Dataset.from_list(data)
        elif isinstance(data, dict):
            return Dataset.from_dict(data)
        else:
            raise ValueError("JSON data must be a list or dict")
    
    def _load_csv(self, file_path: Path) -> Dataset:
        """Load CSV data"""
        df = pd.read_csv(file_path)
        return Dataset.from_pandas(df)
    
    def _load_txt(self, file_path: Path) -> Dataset:
        """Load text data (one example per line)"""
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        data = [{"text": line} for line in lines]
        return Dataset.from_list(data)
    
    def _load_jsonl(self, file_path: Path) -> Dataset:
        """Load JSONL data (one JSON object per line)"""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        
        return Dataset.from_list(data)
    
    def prepare_dataset(self, train_file: str, validation_file: Optional[str] = None) -> DatasetDict:
        """Prepare training and validation datasets"""
        # Load training data
        train_dataset = self.load_data(train_file)
        
        # Handle validation data
        if validation_file:
            val_dataset = self.load_data(validation_file)
        else:
            # Split training data for validation
            split_dataset = train_dataset.train_test_split(
                test_size=self.config.data.validation_split_percentage / 100
            )
            train_dataset = split_dataset['train']
            val_dataset = split_dataset['test']
        
        # Create dataset dictionary
        dataset_dict = DatasetDict({
            'train': train_dataset,
            'validation': val_dataset
        })
        
        logger.info(f"Dataset prepared: {len(train_dataset)} training, {len(val_dataset)} validation examples")
        return dataset_dict
    
    def tokenize_function(self, examples: Dict[str, List]) -> Dict[str, List]:
        """Tokenize examples for training"""
        if not self.tokenizer:
            raise ValueError("Tokenizer not set. Call set_tokenizer() first.")
        
        # Handle different data formats
        if self.config.model.model_type == "causal":
            return self._tokenize_causal(examples)
        else:  # seq2seq
            return self._tokenize_seq2seq(examples)
    
    def _tokenize_causal(self, examples: Dict[str, List]) -> Dict[str, List]:
        """Tokenize for causal language modeling"""
        text_column = self.config.data.text_column
        
        # Concatenate all texts
        texts = examples[text_column]
        if isinstance(texts, str):
            texts = [texts]
        
        # Tokenize with padding and truncation
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.config.data.max_seq_length,
            return_tensors=None
        )
        
        # For causal LM, labels are the same as input_ids
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
    
    def _tokenize_seq2seq(self, examples: Dict[str, List]) -> Dict[str, List]:
        """Tokenize for sequence-to-sequence tasks"""
        # This would need to be customized based on your specific task
        # For now, using the same approach as causal
        return self._tokenize_causal(examples)
    
    def create_data_collator(self):
        """Create a data collator for batching"""
        from transformers import DataCollatorForLanguageModeling
        
        return DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # We're doing causal LM, not masked LM
        )
    
    def get_dataset_stats(self, dataset: Dataset) -> Dict[str, any]:
        """Get statistics about the dataset"""
        stats = {
            'num_examples': len(dataset),
            'columns': dataset.column_names,
            'features': dataset.features
        }
        
        # Get text length statistics if text column exists
        text_column = self.config.data.text_column
        if text_column in dataset.column_names:
            if self.tokenizer:
                # Tokenize to get actual token counts
                tokenized = self.tokenize_function(dataset)
                input_ids = tokenized['input_ids']
                lengths = [len(ids) for ids in input_ids]
                
                stats['token_lengths'] = {
                    'min': min(lengths),
                    'max': max(lengths),
                    'mean': sum(lengths) / len(lengths),
                    'median': sorted(lengths)[len(lengths) // 2]
                }
            else:
                # Just character counts
                texts = dataset[text_column]
                char_lengths = [len(str(text)) for text in texts]
                stats['char_lengths'] = {
                    'min': min(char_lengths),
                    'max': max(char_lengths),
                    'mean': sum(char_lengths) / len(char_lengths),
                    'median': sorted(char_lengths)[len(char_lengths) // 2]
                }
        
        return stats


class ConversationDataProcessor(DataProcessor):
    """Specialized processor for conversation/dialogue data"""
    
    def __init__(self, config):
        super().__init__(config)
        self.conversation_format = "dialogue"  # or "qa", "instruction"
    
    def set_conversation_format(self, format_type: str):
        """Set the conversation format"""
        self.format_type = format_type
    
    def _tokenize_causal(self, examples: Dict[str, List]) -> Dict[str, List]:
        """Tokenize conversation data for causal LM"""
        if self.format_type == "dialogue":
            return self._tokenize_dialogue(examples)
        elif self.format_type == "qa":
            return self._tokenize_qa(examples)
        elif self.format_type == "instruction":
            return self._tokenize_instruction(examples)
        else:
            return super()._tokenize_causal(examples)
    
    def _tokenize_dialogue(self, examples: Dict[str, List]) -> Dict[str, List]:
        """Tokenize dialogue format data"""
        # Format: "Human: ...\nAssistant: ...\nHuman: ...\nAssistant: ..."
        conversations = examples.get("conversation", examples.get("text", []))
        
        if isinstance(conversations, str):
            conversations = [conversations]
        
        # Add EOS token to each conversation
        formatted_conversations = []
        for conv in conversations:
            if not conv.endswith(self.tokenizer.eos_token):
                conv += self.tokenizer.eos_token
            formatted_conversations.append(conv)
        
        # Tokenize
        tokenized = self.tokenizer(
            formatted_conversations,
            truncation=True,
            padding=True,
            max_length=self.config.data.max_seq_length,
            return_tensors=None
        )
        
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    
    def _tokenize_qa(self, examples: Dict[str, List]) -> Dict[str, List]:
        """Tokenize Q&A format data"""
        questions = examples.get("question", [])
        answers = examples.get("answer", [])
        
        if isinstance(questions, str):
            questions = [questions]
        if isinstance(answers, str):
            answers = [answers]
        
        # Format: "Question: ...\nAnswer: ..."
        formatted_texts = []
        for q, a in zip(questions, answers):
            formatted = f"Question: {q}\nAnswer: {a}{self.tokenizer.eos_token}"
            formatted_texts.append(formatted)
        
        # Tokenize
        tokenized = self.tokenizer(
            formatted_texts,
            truncation=True,
            padding=True,
            max_length=self.config.data.max_seq_length,
            return_tensors=None
        )
        
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    
    def _tokenize_instruction(self, examples: Dict[str, List]) -> Dict[str, List]:
        """Tokenize instruction format data"""
        instructions = examples.get("instruction", examples.get("text", []))
        responses = examples.get("response", examples.get("output", []))
        
        if isinstance(instructions, str):
            instructions = [instructions]
        if isinstance(responses, str):
            responses = [responses]
        
        # Format: "### Instruction:\n...\n### Response:\n..."
        formatted_texts = []
        for inst, resp in zip(instructions, responses):
            formatted = f"### Instruction:\n{inst}\n### Response:\n{resp}{self.tokenizer.eos_token}"
            formatted_texts.append(formatted)
        
        # Tokenize
        tokenized = self.tokenizer(
            formatted_texts,
            truncation=True,
            padding=True,
            max_length=self.config.data.max_seq_length,
            return_tensors=None
        )
        
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
