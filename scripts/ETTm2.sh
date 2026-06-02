export CUDA_VISIBLE_DEVICES=0

model_name=PhyFSME

for pred_len in 96 #192 336 720
do
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_$pred_len \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len $pred_len \
  --enc_in 7 \
  --use_norm 1 \
  --hidden_size 128 \
  --batch_size 128 \
  --learning_rate 0.001 \
  --train_epochs 20 \
  --patience 5 \
  --loss MAE \
  --dropout 0.4 \
  --heads 4 \
  --patch_sizes 96,48 \
  --initial_alphas 0.5,0.5 \
  --phys_loss_reduction sum \
  --learnable_beta 0 \
  --beta1 0.01 \
  --beta2 0.1 \
  --beta3 0.1 \
  --lradj type3 \
  --des 'Exp' \
  --itr 1
done


