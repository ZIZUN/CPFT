# CPFT
(unofficial implementation) Code for [Few-Shot Intent Detection via Contrastive Pre-Training and Fine-Tuning  (EMNLP 2021)](https://arxiv.org/pdf/2109.06349.pdf)


## Requirements
* [PyTorch](http://pytorch.org/) >= 1.7.1
* tokenizers==0.9.3
* sklearn
* transformers==4.10.2

## Process

1. Environment Setting
```console
pip install -r ./pretrain/requirements.txt
```

3. Pretraining (MLM Loss + Unsup_con Loss)
```console
cd pretrain
bash scripts/pretrain.sh base 64
```

3. Finetuning (Sup_con Loss + CLS Loss)
Before run, you need to check 'pretrained_model_path' on shell script
```console
cd finetune
bash scripts/finetune_HWU64_5shot.sh base 16
bash scripts/finetune_HWU64_10shot.sh base 16
bash scripts/finetune_CLINC150_10shot.sh base 16
bash scripts/finetune_CLINC150_10shot.sh base 16
bash scripts/finetune_BANKING77_10shot.sh base 16
bash scripts/finetune_BANKING77_10shot.sh base 16
```

## References
* [Supervised Contrastive Learning](https://arxiv.org/pdf/2004.11362.pdf)
* [Few-Shot Intent Detection via Contrastive Pre-Training and Fine-Tuning](https://arxiv.org/pdf/2109.06349.pdf)

## Q&A
If you encounter any problem, leave an issue in the github repo.
