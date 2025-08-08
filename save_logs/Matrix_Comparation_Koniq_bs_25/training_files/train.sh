CUDA_VISIBLE_DEVICES=0,1 \
python train.py --dist-url 'tcp://localhost:12754' \
    --dataset bid livec \ 
    --multiprocessing-distributed --world-size 1 --rank 0 \
    --batch_size 44 --epochs 50 --seed 2024 \
    --random_flipping_rate 0.1 --random_scale_rate 0.5 \
    --model maniqa \
    --save_path ./MIX_Data_Training/maniqa
