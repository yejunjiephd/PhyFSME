export CUDA_VISIBLE_DEVICES=0

model_name=PhyFSME

for pred_len in 96 #192 336 720
do
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path traffic.csv \
  --model_id traffic_$pred_len \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len $pred_len \
  --enc_in 862 \
  --use_norm 1 \
  --hidden_size 512 \
  --batch_size 16 \
  --learning_rate 0.001 \
  --lradj type3 \
  --train_epochs 20 \
  --patience 5 \
  --loss MAE \
  --dropout 0.2 \
  --heads 8 \
  --patch_sizes 96 \
  --initial_alphas 0.5 \
  --phys_loss_reduction sum \
  --learnable_beta 0 \
  --beta1 0.1 \
  --beta2 0.1 \
  --beta3 0.1 \
  --des 'Exp' \
  --itr 1
done
