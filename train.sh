CUDA_VISIBLE_DEVICES=0,1 \
python train.py --dist-url 'tcp://localhost:12754'  --multiprocessing-distributed --world-size 1 --rank 0 \
    --dataset bid livec \
    --batch_size 22 --epochs 50 --seed 2024 \
    --random_flipping_rate 0.1 --random_scale_rate 0.5 \
    --model maniqa \
    --save_path ./MIX_Data_Training/maniqa
