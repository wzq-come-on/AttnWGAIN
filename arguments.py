import argparse
import os

def AttnWGAIN_arguments():
    '''
    Network Parameter Definition
    Modification and definition of global network parameters.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=str, choices=['train', 'test', 'impute'], default='train', help="Model stage")
    parser.add_argument("--saving_impute_path", type=str, default='data/physio2012_37feats_01masked',
                        help="Path to save the imputed results")
    parser.add_argument('--data_path', type=str, choices=[
        'data/AirQuality_seqlen24_01masked',
        'data/Electricity_seqlen100_01masked',
        'data/physio2012_37feats_01masked',
        'data/ETTm1_seqlen24_01masked',
    ], default='data/ETTm1_seqlen24_01masked', help="Select the dataset")
    parser.add_argument("--seq_len", type=int, default=48, help="Time Series Length")
    parser.add_argument("--feature_num", type=int, default=37, help="Feature count in the dataset")
    parser.add_argument('--seed', type=int, default=2023, help='Missing rate of constructed missing data')
    parser.add_argument('--model', type=str, choices=['AttnWGAIN'],\
                        default='AttnWGAIN', help="Model Selection")

    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers for data loading")
    parser.add_argument("--MIT", type=bool, default=True, help="Whether to perform the masked imputation task")
    parser.add_argument("--model_type", type=str, default='AttentionGAN', help="Model type, which affects the data loading method")
    parser.add_argument("--device", type=str, default='cuda', help="Whether to use CUDA")

    parser.add_argument("--epochs", type=int, default=1000, help="Number of iterations")
    parser.add_argument("--epochs_D", type=int, default=1, help="Number of discriminator iterations")
    parser.add_argument("--epochs_G", type=int, default=1, help="Number of generator iterations")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.000682774550436755, help="Learning rate ")

    parser.add_argument("--optimizer_type", type=str, default='adam', help="Optimizer type")
    parser.add_argument("--weight_decay", type=float, default=0.00, help="Optimizer parameters")

    parser.add_argument("--n_groups", type=int, default=5, help="Number of EncoderLayer groups")
    parser.add_argument("--n_group_inner_layers", type=int, default=1, help="Number of layers per group in EncoderLayer")
    parser.add_argument("--d_model", type=int, default=256, help="Multi-head attention mechanism embedding mapped features")
    parser.add_argument("--d_inner", type=int, default=512, help="PositionWiseFeedForward parameter")
    parser.add_argument("--n_head", type=int, default=16, help="Number of heads parameter in the multi-head attention mechanism")
    parser.add_argument("--d_k", type=int, default=32, help="Dimension of q and k layers in the multi-head attention mechanism parameters")
    parser.add_argument("--d_v", type=int, default=32, help="Dimension of the v layer in the multi-head attention mechanism parameters")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate in the multi-head attention mechanism for overfitting prevention")
    parser.add_argument("--diagonal_attention_mask", type=bool, default=True, help="Diagonal masking option in the multi-head attention mechanism")

    parser.add_argument('--miss_rate', type=float, default=0.1, help='Missing rate of constructed missing data')
    parser.add_argument('--hint_rate', type=float, default=0.1, help='Hint probability, which determines the proportion of missing hints provided to the discriminator')
    parser.add_argument('--alpha', type=list, default=[100,100], help='Weighting ratio hyperparameter for loss function combination')
    parser.add_argument('--lambda_gp', type=int, default=10, help='Weighting ratio hyperparameter for the penalty term')

    parser.add_argument('--saving_model_path', type=str, default='./SavedModel', help='Directory for saving models')
    parser.add_argument('--best_imputation_MAE', type=float, default=1.0, help='Optimal strategy value for the model')
    parser.add_argument('--best_imputation_MAE_Threshold', type=float, default=0.5, help='Optimal policy saving threshold for the model')
    parser.add_argument('--best_imputation_RMSE', type=float, default=2.0, help='Optimal strategy value for the model')
    parser.add_argument('--best_imputation_RMSE_Threshold', type=float, default=2.0, help='Optimal policy saving threshold for the model')
    parser.add_argument('--best_imputation_MRE', type=float, default=2.0, help='Optimal strategy value for the model')
    parser.add_argument('--best_imputation_MRE_Threshold', type=float, default=2.0, help='Optimal policy saving threshold for the model')
    parser.add_argument('--min_mae_loss', type=float, default=0.5, help='Threshold for model validation')

    parser.add_argument('--log_saving', type=str, default='./logs', help='Directory for storing logs')

    args = parser.parse_args()
    if args.data_path=='data/physio2012_37feats_01masked':
        args.seq_len=48
        args.feature_num=37
        args.saving_model_path = os.path.join(args.saving_model_path, args.model)
        args.saving_model_path = os.path.join(args.saving_model_path, "physio2012")
        args.log_saving = os.path.join(args.log_saving, args.model)
        args.log_saving = os.path.join(args.log_saving, "physio2012")
        args.d_inner = 512
        args.d_k = 32
        args.d_model = 256
        args.d_v = 32
        args.n_group_inner_layers = 1
        args.n_groups = 5
        args.n_head = 8
        args.lr = 0.0005
        args.dropout = 0
        args.epochs = 1000

    if args.data_path=='data/AirQuality_seqlen24_01masked':
        args.seq_len=24
        args.feature_num=132
        args.saving_model_path = os.path.join(args.saving_model_path, args.model)
        args.saving_model_path = os.path.join(args.saving_model_path, "AirQuality")
        args.log_saving = os.path.join(args.log_saving, args.model)
        args.log_saving = os.path.join(args.log_saving, "AirQuality")
        args.d_inner = 512
        args.d_k = 128
        args.d_model = 256
        args.d_v = 64
        args.n_group_inner_layers = 1
        args.n_groups = 1
        args.n_head = 4
        args.lr = 0.0001
        args.dropout = 0.1
        args.miss_rate = 0.1
        args.epochs = 10000

    if args.data_path=='data/Electricity_seqlen100_01masked':
        args.seq_len=100
        args.feature_num=370
        args.saving_model_path = os.path.join(args.saving_model_path, args.model)
        args.saving_model_path = os.path.join(args.saving_model_path, "Electricity")
        args.log_saving = os.path.join(args.log_saving, args.model)
        args.log_saving = os.path.join(args.log_saving, "Electricity")
        args.d_inner = 128
        args.d_k = 128
        args.d_model = 2048
        args.d_v = 128
        args.n_group_inner_layers = 1
        args.n_groups = 1
        args.n_head = 8
        args.dropout=0.0
        args.lr = 0.0002
        args.miss_rate = 0.1
        args.epochs = 2000

    if args.data_path=='data/ETTm1_seqlen24_01masked':
        args.seq_len=24
        args.feature_num=7
        args.saving_model_path = os.path.join(args.saving_model_path, args.model)
        args.saving_model_path = os.path.join(args.saving_model_path, "ETT")
        args.log_saving = os.path.join(args.log_saving, args.model)
        args.log_saving = os.path.join(args.log_saving, "ETT")
        args.n_group_inner_layers = 5
        args.n_groups = 1
        args.lr = 0.001
        args.hint_rate = 0.1
        args.miss_rate = 0.1
        args.d_model = 256
        args.d_inner = 512
        args.n_head = 1
        args.d_k = 32
        args.d_v = 32
        args.epochs = 10000

    return args

if __name__ == '__main__':
    args = AttnWGAIN_arguments()

    import pandas as pd

    alllist = []
    column = ['one', 'two']


    print('--------args----------')
    for k in list(vars(args).keys()):
        list1=[k, vars(args)[k]]
        alllist.append(list1)
        print('%s: %s' % (k, vars(args)[k]))
    print('--------args----------\n')
    test = pd.DataFrame(columns=column, data=alllist)
    test.to_csv('test.csv')
