from cmath import e
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
    def __init__(self,   seq_len, mode='train'):
        self.seq_len = seq_len

        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        self.start = self.tokenizer.bos_token_id
        self.sep = self.tokenizer.eos_token_id
        self.padding = self.tokenizer.pad_token_id
        self.mask = self.tokenizer.mask_token_id

        
        self.dataset = []

        self.dataset_1, _ = load_intent_examples(file_path='data/HWU64/train')
        self.dataset_2, _ = load_intent_examples(file_path='data/HWU64/valid')
        self.dataset += self.dataset_1 + self.dataset_2


        self.dataset_3, _ = load_intent_examples(file_path='data/CLINC150/train')
        self.dataset_4, _ = load_intent_examples(file_path='data/CLINC150/valid') 
        self.dataset += self.dataset_3 + self.dataset_4


        self.dataset_5, _ = load_intent_examples(file_path='data/BANKING77/train')
        self.dataset_6, _ = load_intent_examples(file_path='data/BANKING77/valid')                      
        self.dataset += self.dataset_5 + self.dataset_6

        self.dataset_7, _ = load_intent_examples(file_path='data/ATIS/train')
        self.dataset_8, _ = load_intent_examples(file_path='data/ATIS/valid')
        self.dataset_9, _ = load_intent_examples(file_path='data/SNIPS/train')
        self.dataset_10, _ = load_intent_examples(file_path='data/SNIPS/valid')
        self.dataset_11, _ = load_intent_examples(file_path='data/TOP')
        
        self.dataset +=  self.dataset_7 + self.dataset_8 + self.dataset_9+ self.dataset_10+ self.dataset_11


        import random
        random.shuffle(self.dataset)
        split_index = len(self.dataset) - 300
        if mode == 'train':
            self.dataset = self.dataset[:split_index]
        elif mode == 'test':
            self.dataset = self.dataset[split_index:]

        self.processed_dataset = []

        import random
        for data in tqdm(self.dataset[:100]):
            text = data.text
            label = data.label


            input_token_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
            masked_input_ids = input_token_ids
            if len(input_token_ids) < 5:
                continue            
            output_labels = [-100] * len(input_token_ids)
            
            
            
            masking_num = len(input_token_ids) * 15 // 100 + 1

            masked_indices = random.sample([i for i in range(len(input_token_ids))], masking_num)
            
            for index in masked_indices:
                check = random.randint(1,10)
                output_labels[index] = input_token_ids[index]
                if check == 1: # 10% - rand token
                    masked_input_ids[index] = random.randint(0, len(self.tokenizer) - 1)
                elif check == 2: # 10% - not change
                    masked_input_ids[index] = masked_input_ids[index]
                else: # 80% - masking token
                    masked_input_ids[index] = self.mask    
                # input_token_ids[index] = self.mask    
                
            if len(masked_input_ids) <= self.seq_len - 2:
                masked_input_ids = [self.start] + masked_input_ids + [self.sep]
                pad_length = self.seq_len - len(masked_input_ids)

                masked_input_ids = masked_input_ids + (pad_length * [self.padding])
            else:
                masked_input_ids = masked_input_ids[:self.seq_len - 2]
                masked_input_ids = [self.start] + masked_input_ids + [self.sep]

            if len(input_token_ids) <= self.seq_len - 2:
                input_token_ids = [self.start] + input_token_ids + [self.sep]
                pad_length = self.seq_len - len(input_token_ids)

                attention_mask = (len(input_token_ids) * [1]) + (pad_length * [0])
                input_token_ids = input_token_ids + (pad_length * [self.padding])
            else:
                input_token_ids = input_token_ids[:self.seq_len - 2]
                input_token_ids = [self.start] + input_token_ids + [self.sep]
                attention_mask = len(input_token_ids) * [1]

            if len(output_labels) <= self.seq_len - 2:
                output_labels = [self.start] + output_labels + [self.sep]
                pad_length = self.seq_len - len(output_labels)

                output_labels = output_labels + (pad_length * [self.padding])
            else:
                output_labels = output_labels[:self.seq_len - 2]
                output_labels = [self.start] + output_labels + [self.sep]

            
            self.processed_dataset.append({"masked_input_ids": masked_input_ids, 
                                           "input_ids": input_token_ids, 'attention_mask': attention_mask, "labels": output_labels})

    def __len__(self):
        return len(self.processed_dataset)

    def __getitem__(self, item):
        output = self.processed_dataset[item]
        return {key: torch.tensor(value) for key, value in output.items()}
    
    
    