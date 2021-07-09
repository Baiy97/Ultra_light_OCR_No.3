# 注意数据存储位置
wget https://bj.bcebos.com/v1/ai-studio-online/8ad391b076894b449ca938ebe40cf6788ab000aa6e0d463586315734f20ceb20?responseContentDisposition=attachment%3B%20filename%3D%E8%AE%AD%E7%BB%83%E6%95%B0%E6%8D%AE%E9%9B%86.zip&authorization=bce-auth-v1%2F0ef6765c1e494918bc0d4c3ca3e5c6d1%2F2021-05-09T15%3A40%3A37Z%2F-1%2F%2Fd0127b44a66b9e5ef114252b5be3402d58604332f5e047055fb2c3073f949902


# 第一阶段训练
CUDA_VISIBLE_DEVICES=0 python3 tools/train.py -c configs/resnet34_final.yml

# 在训练集上测试
python3 tools/export_model.py -c configs/resnet34_final.yml -o Global.checkpoints=./output_resnet34_final/rec_owndict/latest
CUDA_VISIBLE_DEVICES=0 python3 tools/infer/predict_rec.py --image_dir="./data/train_data/TrainImages" --rec_model_dir="inference_models" --use_space_char=True --rec_batch_num=128

# 生成包含字符位置信息的文件
python char_seq_full_gt_gen.py


# 第二阶段训练
CUDA_VISIBLE_DEVICES=0,1 python3 -m paddle.distributed.launch --log_dir=./debug/ --gpus '0,1'  tools/train.py -c configs/mobilenet_taller_cbam_swish.yml

# 在测试集上测试
python3 tools/export_model.py -c configs/mobilenet_taller_cbam_swish.yml -o Global.checkpoints=./output_mobilenet_taller_cbam_swish/rec_owndict/latest
python3 tools/infer/predict_rec_ensemble.py --image_dir="./data/TestBImages" --rec_model_dir="inference_models" --use_space_char=True --rec_batch_num=128

# ensemble 生成 ensemble_res.txt
python ensemble.py
