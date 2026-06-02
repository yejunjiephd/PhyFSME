


export CUDA_VISIBLE_DEVICES=0

model_name=PhyFSME

for pre_len in 96 # 192 336 720
do
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_$pre_len \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len $pre_len \
  --enc_in 7 \
  --hidden_size 128 \
  --use_norm 1 \
  --batch_size 256 \
  --learning_rate 0.001 \
  --dropout 0.2 \
  --heads 2 \
  --train_epochs 10 \
  --patience 3 \
  --loss MAE \
  --patch_sizes 96,48,24 \
  --initial_alphas 0.5,0.4,0.4 \
  --phys_loss_reduction sum \
  --beta1 0.01 \
  --beta2 0.1 \
  --beta3 0.1 \
  --learnable_beta 0 \
  --itr 1
done


