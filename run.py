import argparse
import os
import torch
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
import random
import numpy as np

if __name__ == '__main__':
    fix_seed = 2026
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser()

    # basic config
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--model_id', type=str, default='ETTm1', help='model id')
    parser.add_argument('--model', type=str, default='DLinear',
                        help='model name, options: [DLinear, Amplifier]')

    # data loader
    parser.add_argument('--data', type=str, default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='../../dataset/ETT-small', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTm1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')

    # Amplifier
    parser.add_argument('--use_norm', type=int, default=1, help='1: using use_norm Block, 1: not using use_norm Block for Amplifier model')
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--patch_sizes', type=str, default='',
                        help='comma-separated patch sizes for PhyFSME, e.g. 96,24,12; empty uses model default')
    parser.add_argument('--initial_alphas', type=str, default='',
                        help='comma-separated initial FRFT orders for PhyFSME experts; empty uses 0.5 for all')
    parser.add_argument('--phys_loss_reduction', type=str, default='sum', choices=['sum', 'mean'],
                        help='how to aggregate physical losses from multiple FRFT experts')

    # DLinear
    parser.add_argument('--individual', action='store_true', default=False,
                        help='DLinear: a linear layer for each variate(channel) individually')

    # optimization
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.02, help='optimizer learning rate')
    parser.add_argument('--alpha_learning_rate', type=float, default=0,
                        help='separate learning rate for FRFT alpha_raw parameters; 0 uses learning_rate')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='optimizer weight decay')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MAE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--alpha', type=float, default=0.8, help='fractional alpha')
    parser.add_argument('--keep_ratio', type=float, default=1, help='top_k_ratio')
    parser.add_argument('--heads', type=int, default=2, help='data loader num workers')

    parser.add_argument('--beta1', type=float, default=0.1, help='loss scale')
    parser.add_argument('--beta2', type=float, default=0.1, help='loss scale')
    parser.add_argument('--beta3', type=float, default=0.1, help='loss scale')
    parser.add_argument('--learnable_beta', type=int, default=0,
                        help='1: learn weights of the three physical losses; 0: use fixed beta1/beta2/beta3')
    parser.add_argument('--beta_mode', type=str, default='free', choices=['free', 'proportional'],
                        help='free: learn positive beta values; proportional: learn beta ratios with fixed total strength')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')



    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    args.use_gpu = True if torch.cuda.is_available() else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    Exp = Exp_Long_Term_Forecast

    if args.is_training:
        for ii in range(args.itr):

            exp = Exp(args)
            setting = '{}_{}_{}_sl{}_pl{}_hidden{}_use_norm{}_epochs{}_bc{}_lr{}_{}'.format(
                args.model_id,
                args.model,
                args.data,
                args.seq_len,
                args.pred_len,
                args.hidden_size,
                args.use_norm,
                args.train_epochs,
                args.batch_size,
                args.learning_rate,
                ii)

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_sl{}_pl{}_hidden{}_use_norm{}_epochs{}_bc{}_lr{}_{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.seq_len,
            args.pred_len,
            args.hidden_size,
            args.use_norm,
            args.train_epochs,
            args.batch_size,
            args.learning_rate,
            ii)

        exp = Exp(args)
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
