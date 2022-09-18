

##   ex) bash cola.sh base 4
##   ex) bash cola.sh base 4 ddp 4
model=$1
bsz=$2
ddp=$3
ngpu_ddp=$4

cd..

# (DDP) or (nn.dataparallel, cpu)
if [ "${ddp}" = "ddp" ]
then
    cmd="${cmd}python -m torch.distributed.launch --nproc_per_node=${ngpu_ddp} --master_port=75128"
else
    cmd="${cmd}python"
fi

cmd="${cmd} finetune.py -c data/BANKING77/train_10 -v data/BANKING77/valid -t data/BANKING77/test --model=${model}\
            -o output/gpt2.model --batch_size ${bsz}  --epochs 10022222 --lr 5e-5 --seed 1000 --gpu_num 0
            --input_seq_len 50 --log_freq 100  --accumulate 1 --task BANKING77 --mode train
            --pretrained_model_path pretrained_checkpoint/71_17.67_1000_base"

if [ "${ddp}" = "ddp" ]
then
    cmd="${cmd} --ddp True"
fi

echo $cmd
$cmd
