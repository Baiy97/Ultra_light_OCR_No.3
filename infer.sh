# 在测试集上测试
python3 tools/export_model.py -c configs/mobilenet_taller_cbam_swish.yml -o Global.checkpoints=./output_mobilenet_taller_cbam_swish/rec_owndict/latest
python3 tools/infer/predict_rec_ensemble.py --image_dir="./data/TestBImages" --rec_model_dir="inference_models" --use_space_char=True --rec_batch_num=128

# ensemble 生成 ensemble_res.txt
python ensemble.py
