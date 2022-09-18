import argparse

from torch.utils.data import DataLoader

from util.model.RobertaForPretrain import RobertaPretrain_utterence
from util.trainer import Trainer
from util.dataset import LoadDataset, load_intent_examples
import torch
from transformers.models.roberta.modeling_roberta import RobertaConfig
from torch import distributed as dist
from torch.utils.data.distributed import DistributedSampler



parser = argparse.ArgumentParser()

parser.add_argument("-c", "--train_dataset", required=False, type=str, help="train dataset")
parser.add_argument("-t", "--test_dataset", type=str, default=None, help="test set for evaluate train set")
parser.add_argument("-o", "--output_path", required=False, type=str, help="ex)output/bert.model")

parser.add_argument("--model", type=str, required=True, help="model (base,trinity)")
parser.add_argument("--ddp", type=bool, default=False, help="for distrbuted data parrerel")
parser.add_argument("--local_rank", type=int, help="for distrbuted data parrerel")

parser.add_argument("--input_seq_len", required=True, type=int, default=512, help="maximum sequence input len")

parser.add_argument("-b", "--batch_size", type=int, default=8, help="number of batch_size")
parser.add_argument("-e", "--epochs", type=int, default=10, help="number of epochs")
parser.add_argument("-w", "--num_workers", type=int, default=5, help="dataloader worker size")

parser.add_argument("--with_cuda", type=bool, default=True, help="training with CUDA: true, or false")
parser.add_argument("--log_freq", type=int, default=1, help="printing loss every n iter: setting n")
parser.add_argument("--cuda_devices", type=int, nargs='+', default=None, help="CUDA device ids")
parser.add_argument("--on_memory", type=bool, default=True, help="Loading on memory: true or false")

parser.add_argument("--lr", type=float, default=1e-4, help="learning rate of adam")
parser.add_argument("--adam_weight_decay", type=float, default=0.01, help="weight_decay of adam")
parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam first beta value")

parser.add_argument("--accumulate", type=int, default=1, help="accumulation step")
parser.add_argument("--seed", type=int, default=42, help="seed")
parser.add_argument("--task", type=str, required=True, help="model (base,trinity)")
parser.add_argument("--gpu_num", type=str, default='0', help="gpu number")

args = parser.parse_args()



import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_num

os.environ["TOKENIZERS_PARALLELISM"] = "true"


if args.ddp:
    dist.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(args.local_rank)
import numpy as np
import random

os.environ["PL_GLOBAL_SEED"] = str(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)




print("Loading Train Dataset")
train_dataset = LoadDataset(seq_len=args.input_seq_len, mode='train')

print("Loading Test Dataset")
test_dataset = LoadDataset(seq_len=args.input_seq_len, mode='test')

if args.ddp:
    print("Creating Dataloader")
    train_sampler = DistributedSampler(train_dataset)
    train_data_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size, num_workers=args.num_workers)
else:
    print("Creating Dataloader")
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)

if args.ddp:
    print("Creating Dataloader")
    test_sampler = DistributedSampler(test_dataset)
    test_data_loader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size, num_workers=args.num_workers) \
        if test_dataset is not None else None
else:
    test_data_loader = DataLoader(test_dataset, batch_size=20, num_workers=args.num_workers) \
        if test_dataset is not None else None


model = RobertaPretrain_utterence(model_name="roberta")



print("Creating Trainer")
trainer = Trainer(task=args.task, model=model, train_dataloader=train_data_loader, test_dataloader=test_data_loader,
                  lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
                  with_cuda=args.with_cuda, cuda_devices=args.cuda_devices, log_freq=args.log_freq,
                  distributed = args.ddp, local_rank = args.local_rank, accum_iter= args.accumulate,
                  seed= args.seed, model_name=args.model)

print("Training Start")
for epoch in range(args.epochs):
    if args.ddp:
        train_sampler.set_epoch(epoch)
        trainer.train(epoch)
    else:
        trainer.train(epoch)

    train_dataset = LoadDataset(seq_len=args.input_seq_len, mode='train')
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    trainer.change_train_data(train_data_loader)        