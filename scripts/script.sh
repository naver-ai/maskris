#!/bin/bash
uname -a
#date
#env
date

GPUS=0,1
DATASET=refcoco # refcoco, refcoco+, refcocog
DATA_PATH=YOUR_PATH_TO_COCO_DATASET
REFER_PATH=YOUR_PATH_TO_COCO_DATASET/refer
SWIN_PATH=YOUR_PATH_TO_SWIN_TRANSFORMER
BERT_TYPE=bert-base-uncased
OUTPUT_PATH=YOUR_PATH_TO_SAVE_LOGS
IMG_SIZE=448
now=$(date +"%Y%m%d_%H%M%S")

mkdir -p ${OUTPUT_PATH}
mkdir -p ${OUTPUT_PATH}/${DATASET}


# training code
CUDA_VISIBLE_DEVICES=${GPUS} python -m torch.distributed.launch --nproc_per_node 2 --master_port 12348 main.py \
        --dataset ${DATASET} --model_id ${DATASET} --batch-size 8 --pin_mem --print-freq 100 --workers 16 \
        --lr 1e-4 --wd 1e-2 --swin_type base \
        --warmup --warmup_ratio 1e-3 --warmup_iters 1500 --clip_grads --clip_value 1.0 \
        --pretrained_swin_weights ${SWIN_PATH} --epochs 50 --img_size ${IMG_SIZE} \
        --bert_tokenizer ${BERT_TYPE} --ck_bert ${BERT_TYPE} --output-dir ${OUTPUT_PATH} \
        --img_patch_size 32 --img_mask_ratio 0.75 \
        --txt_mask_ratio 0.15 --txt_mask_ratio_sub 0.8 0.1 \
        --refer_data_root ${DATA_PATH} --refer_root ${REFER_PATH} 2>&1 \
        | tee ${OUTPUT_PATH}'/'${DATASET}'/'train-${now}.txt

# evaluation code
SPLIT=val # val, testA, testB
RESUME_PATH=${OUTPUT_PATH}/model_best_${DATASET}.pth

CUDA_VISIBLE_DEVICES=${GPUS} python eval.py --swin_type base \
        --dataset ${DATASET} --split ${SPLIT} \
        --img_size ${IMG_SIZE} --resume ${RESUME_PATH} \
        --bert_tokenizer ${BERT_TYPE} --ck_bert ${BERT_TYPE} \
        --refer_data_root ${DATA_PATH} --refer_root ${REFER_PATH} 2>&1 | tee ${OUTPUT_PATH}/eval-${SPLIT}.txt
