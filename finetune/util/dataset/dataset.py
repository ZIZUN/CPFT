import re
from torch.utils.data import Dataset
import torch

from transformers import AutoTokenizer
from tqdm import tqdm

class IntentExample:
    def __init__(self, text, label, do_lower_case):
        self.original_text = text
        self.text = text
        self.label = label

        if do_lower_case:
            self.text = self.text.lower()

def load_intent_examples(file_path, do_lower_case=True):
    examples = []
    
    labels_li = []
    with open('{}/seq.in'.format(file_path), 'r', encoding="utf-8") as f_text, open('{}/label'.format(file_path), 'r', encoding="utf-8") as f_label:
        for text, label in zip(f_text, f_label):
            text = text.strip()
            label = label.strip()
            
            if label not in labels_li:
                labels_li.append(label)
            
            e = IntentExample(text, label, do_lower_case)
            examples.append(e)
    return examples, labels_li


class LoadDataset(Dataset):
    def __init__(self, corpus_path, labels_li, seq_len):
        self.seq_len = seq_len
        self.corpus_path = corpus_path

        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        self.start = self.tokenizer.bos_token_id
        self.sep = self.tokenizer.eos_token_id
        self.padding = self.tokenizer.pad_token_id

        self.dataset, _ = load_intent_examples(file_path=corpus_path)
        self.labels_li = labels_li
        self.dataset_len = len(self.dataset)
        
        self.processed_dataset = []

        for data in tqdm(self.dataset):
            text = data.text
            label = data.label
            label = self.labels_li.index(label)

            text = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))

            if len(text) <= self.seq_len - 2:
                text = [self.start] + text + [self.sep]
                pad_length = self.seq_len - len(text)

                attention_mask = (len(text) * [1]) + (pad_length * [0])
                text = text + (pad_length * [self.padding])
            else:
                text = text[:self.seq_len - 2]
                text = [self.start] + text + [self.sep]
                attention_mask = len(text) * [1]

            model_input = text
            model_label = int(label)
            
            self.processed_dataset.append({"input_ids": model_input, 'attention_mask': attention_mask, "labels": model_label})

    def __len__(self):
        return len(self.processed_dataset)

    def __getitem__(self, item):
        output = self.processed_dataset[item]
        return {key: torch.tensor(value) for key, value in output.items()}
    
    
    