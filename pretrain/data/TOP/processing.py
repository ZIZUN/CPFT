# read TSV files
import csv


all_lines = []
all_labels = []
with open('/home/leesm/Project/contrastive_training/pretrain/data/TOP/eval.tsv') as f:
    tr = csv.reader(f, delimiter='\t')
    for row in tr:  
        all_lines.append(row[0])
        all_labels.append(row[2])
     
with open('/home/leesm/Project/contrastive_training/pretrain/data/TOP/train.tsv') as f:
    tr = csv.reader(f, delimiter='\t')
    for row in tr:  
        all_lines.append(row[0])     
        all_labels.append(row[2])
        
        
with open('/home/leesm/Project/contrastive_training/pretrain/data/TOP/seq.in', 'a') as file:
    for line in all_lines:
        file.write(line + '\n')
with open('/home/leesm/Project/contrastive_training/pretrain/data/TOP/label', 'a') as file:
    for label in all_labels:
        file.write(label + '\n')