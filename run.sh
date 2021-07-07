# 注意数据存储位置


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
