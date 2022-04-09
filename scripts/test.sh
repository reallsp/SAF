GPUS=4,5,6,7
export CUDA_VISIBLE_DEVICES=$GPUS

IMAGE_DIR=/home/lishiping/cuhkpedes/imgs
BASE_ROOT=/home/lishiping
ANNO_DIR=$BASE_ROOT/cuhkpedes/processed_data
text_model='bert'
CKPT_DIR=$BASE_ROOT/checkpoints/data/model_data
LOG_DIR=$BASE_ROOT/person_search/data/logs
image_model='vit'
pretrain_dir=$BASE_ROOT/pretrained_models/imagenet21k+imagenet2012_ViT-B_16.npz

lr=0.0003
num_epoches=45
batch_size=64
lr_decay_ratio=0.9
epoches_decay=20_30_40
diversity_lambda=0.3

python $BASE_ROOT/person_search/test.py \
    --log_dir $LOG_DIR/.. \
    --model_path $CKPT_DIR/.. \
    --image_dir $IMAGE_DIR \
    --img_model $image_model \
    --pretrain_dir $pretrain_dir \
    --anno_dir $ANNO_DIR \
    --text_model $text_model \
    --gpus $GPUS \
    --num_heads 10\
    --checkpoint_dir $CKPT_DIR \
    --feature_size 768 \
    --lambda_diversity $diversity_lambda \


