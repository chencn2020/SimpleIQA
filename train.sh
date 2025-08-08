CUDA_VISIBLE_DEVICES=0,1 \
python -u train.py --dist-url 'tcp://localhost:12754'  --multiprocessing-distributed --world-size 1 --rank 0 \
    --dataset livec csiq kadid koniq10k live spaq_2 spaq_3 spaq_4 spaq_5 spaq_6 spaq_7 \
    --zero_shot_dataset bid tid2013 \
    --batch_size 24 --epochs 50 --seed 2024 \
    --random_flipping_rate 0.1 --random_scale_rate 0.5 \
    --model maniqa \
    --save_path ./MIX_Data_Training/MANIQA
