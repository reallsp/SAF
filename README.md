This repo is the official implementation of "[Learning Semantic-Aligned Feature Representation for Text-based Person Search](https://arxiv.org/abs/2112.06714)" in PyTorch.
## Dependencies
* Python 3.7
* Pytorch 1.0.0 & torchvision 0.2.1
* numpy
* scipy 1.2.1 
* pytorch_transformers
## Usage

### Data Preparation

1. Please download [CUHK-PEDES dataset](https://github.com/ShuangLI59/Person-Search-with-Natural-Language-Description) .
2. Put reid_raw.json under project_directory/data/
3. run data.sh
2. Copy files **test_reid.json**, **train_reid.json** and **val_reid.json** under CUHK-PEDES/data/ to project_directory/cuhkpedes/processed_data/
3. Download [ViT-B_16](https://console.cloud.google.com/storage/vit_models/),  [bert-base-uncased model](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz) and [vocabulary](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt) to project_directory/pretrained_models/

