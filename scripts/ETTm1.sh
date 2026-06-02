export CUDA_VISIBLE_DEVICES=0

model_name=PhyFSME

for pred_len in 96 #192 336 720
do
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_$pred_len \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len $pred_len \
  --enc_in 7 \
  --hidden_size 256 \
  --use_norm 1 \
  --batch_size 128 \
  --learning_rate 0.001 \
  --loss MAE \
  --dropout 0.5 \
  --heads 2 \
  --train_epochs 20 \
  --patience 5 \
  --patch_sizes 96,48,12 \
  --initial_alphas 0.5,0.5,0.5 \
  --phys_loss_reduction sum \
  --beta1 0.1 \
  --beta2 0.1 \
  --beta3 0.1 \
  --lradj type3 \
  --learnable_beta 0 \
  --itr 1
done
