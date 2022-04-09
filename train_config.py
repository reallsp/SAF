import argparse
import os
import logging
from config import log_config, dir_config

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser(description='command for train on CUHK-PEDES')

    # Directory
    parser.add_argument('--image_dir', type=str, help='directory to store dataset')
    parser.add_argument('--anno_dir', type=str, help='directory to store anno file')
    parser.add_argument('--checkpoint_dir', type=str, help='directory to store checkpoint')
    parser.add_argument('--log_dir', type=str, help='directory to store log')
    parser.add_argument('--model_path', type=str, default = None, help='directory to pretrained model, whole model or just visual part')
    parser.add_argument('--pretrain_dir', type=str, default=None,
                        help='the path of vit parameters')
    parser.add_argument('--resnet50_dir', type=str, default=None,
                        help='the path of vit parameters')
    # Model setting

    parser.add_argument('--img_model', type=str,help='which model used to extract image features')
    parser.add_argument('--text_model', type=str,help='which model used to extract text features')
    parser.add_argument('--resume', action='store_true', help='whether or not to restore the pretrained whole model')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epoches', type=int, default=100)
    parser.add_argument('--ckpt_steps', type=int, default=5000, help='#steps to save checkpoint')
    parser.add_argument('--num_heads', type=int, default=12, help='#num of heads')
    parser.add_argument('--feature_size', type=int, default=512)
    parser.add_argument('--CMPM', action='store_true')
    parser.add_argument('--CMPC', action='store_true')
    parser.add_argument('--constraints_text', action='store_true')
    parser.add_argument('--constraints_images', action='store_true')
    parser.add_argument('--num_classes', type=int, default=11003)
    parser.add_argument('--pretrained', action='store_true', help='whether or not to restore the pretrained visual model')
    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14", "R50-ViT-B_16"],
                        default="ViT-B_16",
                        help="Which variant to use.")
    parser.add_argument('--embedding_size', type=int, default=512)
    parser.add_argument('--vocab_size', type=int, default=12000)
    parser.add_argument('--lstm_dropout_ratio', type=float, default=0.7)
    parser.add_argument('--max_length', type=int, default=100)
    # Optimization setting
    parser.add_argument('--optimizer', type=str, default='adam', help='one of "sgd", "adam", "rmsprop", "adadelta", or "adagrad"')
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--wd', type=float, default=0.00004)
    parser.add_argument('--lambda_diversity', type=float, default=0.5)
    parser.add_argument('--adam_alpha', type=float, default=0.9)
    parser.add_argument('--adam_beta', type=float, default=0.999)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--end_lr', type=float, default=0.0001, help='minimum end learning rate used by a polynomial decay learning rate')
    parser.add_argument('--lr_decay_type', type=str, default='exponential', help='One of "fixed" or "exponential"')
    parser.add_argument('--lr_decay_ratio', type=float, default=0.1)
    parser.add_argument('--epoches_decay', type=str, default='50,100', help='#epoches when learning rate decays')

    parser.add_argument('--nsave', type=str, default='')


    # Default setting
    parser.add_argument('--gpus', type=str, default='0')
    parser.add_argument('--layer_ids', type=str, default='-1')
    args = parser.parse_args()
    return args

def config():
    args = parse_args()
    dir_config(args)
    log_config(args,'train')
    return args
