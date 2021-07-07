# recommended paddle.__version__ == 2.0.0
# CUDA_VISIBLE_DEVICES=0 python3 tools/train.py -c configs/rec/ch_ppocr_v2.0/rec_chinese_lite_train_v2.0.yml
# python3 -m paddle.distributed.launch --log_dir=./debug/ --gpus '0,1'  tools/train.py -c configs/rec/ch_ppocr_v2.0/rec_chinese_lite_train_v2.0.yml
# CUDA_VISIBLE_DEVICES=1 python3 tools/train.py -c  configs/rec/ch_ppocr_v2.0/rec_chinese_lite_train_v2.0_adam_lr0.01_warmup10.yml
# CUDA_VISIBLE_DEVICES=2 python3 tools/train.py -c  configs/rec/ch_ppocr_v2.0/rec_chinese_lite_train_v2.0_adam_lr0.001_warmup10_len160.yml
# CUDA_VISIBLE_DEVICES=0 python3 tools/train.py -c configs/resnet34_final.yml
CUDA_VISIBLE_DEVICES=0,1 python3 -m paddle.distributed.launch --log_dir=./debug/ --gpus '0,1'  tools/train.py -c configs/mobilenet_taller_cbam_swish.yml
