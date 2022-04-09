GPUS=5,4,6,7
export CUDA_VISIBLE_DEVICES=$GPUS

IMAGE_DIR=/home/lishiping/cuhkpedes/imgs
BASE_ROOT=/home/lishiping
ANNO_DIR=$BASE_ROOT/cuhkpedes/processed_data
text_model='bert'
CKPT_DIR=$BASE_ROOT/checkpoints/data/model_data
LOG_DIR=$BASE_ROOT/data/logs
image_model='vit'
pretrain_dir=$BASE_ROOT/pretrained_models/imagenet21k+imagenet2012_ViT-B_16.npz
resnet50_dir=/home/lishiping/pretrained_models/resnet50-19c8e357.pth
lr=0.0003
num_epoches=60
batch_size=64
lr_decay_ratio=0.9
epoches_decay=20_30_40
diversity_lambda=0.2
Layer_ids=-1
num_classes=11003

python $BASE_ROOT/person_search/train.py \
    --CMPC \
    --CMPM \
    --img_model $image_model \
    --text_model $text_model \
    --pretrain_dir $pretrain_dir \
    --num_heads 10\
    --log_dir $LOG_DIR/10parts\
    --checkpoint_dir $CKPT_DIR/10parts\
    --image_dir $IMAGE_DIR \
    --anno_dir $ANNO_DIR \
    --batch_size $batch_size \
    --gpus $GPUS \
    --num_epoches $num_epoches \
    --lr $lr \
    --lambda_diversity $diversity_lambda \
    --lr_decay_ratio $lr_decay_ratio \
    --epoches_decay ${epoches_decay} \
    --num_classes ${num_classes} \
    --feature_size 768 \



