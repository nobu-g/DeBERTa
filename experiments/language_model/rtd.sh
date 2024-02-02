#!/usr/bin/env bash

set -eo pipefail

SCRIPT=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT")
cd $SCRIPT_DIR

# cache_dir=/tmp/DeBERTa/RTD/

# max_seq_length=512
# data_dir=$cache_dir/wiki103/spm_$max_seq_length

# setup_wiki_data() {
#   task=$1
#   mkdir -p $cache_dir
#   if [[ ! -e $cache_dir/spm.model ]]; then
#     wget -q https://huggingface.co/microsoft/deberta-v3-base/resolve/main/spm.model -O $cache_dir/spm.model
#   fi

#   if [[ ! -e $data_dir/test.txt ]]; then
#     wget -q https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip -O $cache_dir/wiki103.zip
#     unzip -j $cache_dir/wiki103.zip -d $cache_dir/wiki103
#     mkdir -p $data_dir
#     python ./prepare_data.py -i $cache_dir/wiki103/wiki.train.tokens -o $data_dir/train.txt --max_seq_length $max_seq_length
#     python ./prepare_data.py -i $cache_dir/wiki103/wiki.valid.tokens -o $data_dir/valid.txt --max_seq_length $max_seq_length
#     python ./prepare_data.py -i $cache_dir/wiki103/wiki.test.tokens -o $data_dir/test.txt --max_seq_length $max_seq_length
#   fi
# }

# setup_wiki_data

max_seq_length=512
data_dir=${HOME}/work/DeBERTa/data
output_dir=${HOME}/work/DeBERTa/output

task=RTD

init=$1
tag="${init}-$(date -u +%Y-%m-%d-%H-%M)"
case ${init,,} in
deberta-v3-xsmall-continue)
  # wget https://huggingface.co/microsoft/deberta-v3-xsmall/resolve/main/pytorch_model.generator.bin
  # wget https://huggingface.co/microsoft/deberta-v3-xsmall/resolve/main/pytorch_model.bin
  parameters=" --num_train_epochs 1 \
  --model_config rtd_xsmall.json \
  --warmup 10000 \
  --num_training_steps 100000 \
  --learning_rate 5e-5 \
  --train_batch_size 256 \
  --init_generator <TODO: generator checkpoint> \
  --init_discriminator <TODO: discriminator checkpoint> \
  --decoupled_training True \
  --fp16 True "
  ;;
deberta-v3-xsmall)
  parameters=" --num_train_epochs 1 \
  --model_config rtd_xsmall.json \
  --warmup 10000 \
  --learning_rate 3e-4 \
  --epsilon 1e-6 \
  --adam_beta1 0.9 \
  --adam_beta2 0.98 \
  --train_batch_size 64 \
  --decoupled_training True \
  --fp16 True "
  ;;
deberta-v3-small-continue)
  # wget https://huggingface.co/microsoft/deberta-v3-small/resolve/main/pytorch_model.generator.bin
  # wget https://huggingface.co/microsoft/deberta-v3-small/resolve/main/pytorch_model.bin
  parameters=" --num_train_epochs 1 \
  --model_config rtd_small.json \
  --warmup 10000 \
  --num_training_steps 100000 \
  --learning_rate 5e-5 \
  --train_batch_size 256 \
  --init_generator <TODO: generator checkpoint> \
  --init_discriminator <TODO: discriminator checkpoint> \
  --decoupled_training True \
  --fp16 True "
  ;;
deberta-v3-base)
  parameters=" --num_train_epochs 2 \
  --model_config rtd_base.json \
  --warmup 10000 \
  --learning_rate 1e-4 \
  --epsilon 1e-6 \
  --adam_beta1 0.9 \
  --adam_beta2 0.999 \
  --max_grad_norm 0.5 \
  --train_batch_size 2400 \
  --eval_batch_size 64 \
  --accumulative_update 3 \
  --decoupled_training True \
  --fp16 True "
  ;;
deberta-v3-base-continue)
  parameters=" --num_train_epochs 2 \
  --model_config rtd_base.json \
  --warmup 10000 \
  --learning_rate 1e-4 \
  --epsilon 1e-6 \
  --adam_beta1 0.9 \
  --adam_beta2 0.999 \
  --max_grad_norm 0.5 \
  --train_batch_size 2400 \
  --eval_batch_size 64 \
  --init_generator ${output_dir}/deberta-v3-base2024-01-30-16-28/RTD/generator/pytorch.model-000500.bin \
  --init_discriminator ${output_dir}/deberta-v3-base2024-01-30-16-28/RTD/discriminator/pytorch.model-000500.bin \
  --init_resume_step 500 \
  --accumulative_update 3 \
  --decoupled_training True \
  --fp16 True "
  ;;
deberta-v3-large)
  parameters=" --num_train_epochs 1 \
  --model_config rtd_large.json \
  --warmup 10000 \
  --learning_rate 1e-4 \
  --epsilon 1e-6 \
  --adam_beta1 0.9 \
  --adam_beta2 0.98 \
  --train_batch_size 256 \
  --decoupled_training True \
  --fp16 True "
  ;;
*)
  echo "usage $0 <Pretrained model configuration>"
  echo "Supported configurations"
  echo "deberta-v3-xsmall - Pretrained DeBERTa v3 XSmall model with 9M backbone network parameters (12 layers, 256 hidden size) plus 32M embedding parameters(128k vocabulary size)"
  echo "deberta-v3-xsmall - Pretrained DeBERTa v3 Base model with 81M backbone network parameters (12 layers, 768 hidden size) plus 96M embedding parameters(128k vocabulary size)"
  echo "deberta-v3-xsmall - Pretrained DeBERTa v3 Large model with 288M backbone network parameters (24 layers, 1024 hidden size) plus 128M embedding parameters(128k vocabulary size)"
  exit 0
  ;;
esac

python -m DeBERTa.apps.run --model_config config.json \
  --tag "${tag}" \
  --do_train \
  --max_seq_len $max_seq_length \
  --dump 5000 \
  --task_name "${task}" \
  --data_dir "${data_dir}/full" \
  --vocab_path "${data_dir}/spm/code20K_en40K_ja60K.ver2.2.model" \
  --vocab_type spm \
  --world_size 1 \
  --workers 2 \
  --seed 42 \
  --output_dir "${output_dir}/${tag}/${task}" ${parameters}
