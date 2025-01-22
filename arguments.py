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
                        default='AttnWGAIN', help="模型选择")

    parser.add_argument("--num_workers", type=int, default=1, help="dataloader数据加载的子流程数")
    parser.add_argument("--MIT", type=bool, default=True, help="是否进行masked_imputation_task")
    parser.add_argument("--model_type", type=str, default='AttentionGAN', help="模型类型，影响数据读取方法")
    parser.add_argument("--device", type=str, default='cuda', help="使用cuda")

    parser.add_argument("--epochs", type=int, default=1000, help="迭代次数")
    parser.add_argument("--epochs_D", type=int, default=1, help="判别器迭代次数")
    parser.add_argument("--epochs_G", type=int, default=1, help="生成器迭代次数")
    parser.add_argument("--batch_size", type=int, default=128, help="小批次batch大小")
    parser.add_argument("--lr", type=float, default=0.000682774550436755, help="学习率 ")

    parser.add_argument("--optimizer_type", type=str, default='adam', help="优化器类型")
    parser.add_argument("--weight_decay", type=float, default=0.00, help="优化器参数")

    parser.add_argument("--n_groups", type=int, default=5, help="EncoderLayer组数")
    parser.add_argument("--n_group_inner_layers", type=int, default=1, help="EncoderLayer每组层数")
    parser.add_argument("--d_model", type=int, default=256, help="多头注意力机制embedding映射特征")
    parser.add_argument("--d_inner", type=int, default=512, help="PositionWiseFeedForward参数 2048")
    parser.add_argument("--n_head", type=int, default=16, help="多头注意力机制参数 头的数目")
    parser.add_argument("--d_k", type=int, default=32, help="多头注意力机制参数 q,k层的维度")
    parser.add_argument("--d_v", type=int, default=32, help="多头注意力机制参数 v层的维度 64")
    parser.add_argument("--dropout", type=float, default=0.0, help="多头注意力机制 dropout概率，防止过拟合")
    parser.add_argument("--diagonal_attention_mask", type=bool, default=True, help="多头注意力机制 是否对角掩码")

    parser.add_argument('--miss_rate', type=float, default=0.1, help='构造缺失数据的缺失率')
    parser.add_argument('--hint_rate', type=float, default=0.1, help='提示概率，决定了给判别器的缺失提示占比')
    parser.add_argument('--alpha', type=list, default=[100,100], help='超参数，损失函数结合占比')
    parser.add_argument('--lambda_gp', type=int, default=10, help='超参数，提出惩罚项占比')

    parser.add_argument('--saving_model_path', type=str, default='./SavedModel', help='模型保存目录')
    parser.add_argument('--best_imputation_MAE', type=float, default=1.0, help='模型最优策略值')
    parser.add_argument('--best_imputation_MAE_Threshold', type=float, default=0.5, help='模型保存最优策略阈值')
    parser.add_argument('--best_imputation_RMSE', type=float, default=2.0, help='模型最优策略值')
    parser.add_argument('--best_imputation_RMSE_Threshold', type=float, default=2.0, help='模型保存最优策略阈值')
    parser.add_argument('--best_imputation_MRE', type=float, default=2.0, help='模型最优策略值')
    parser.add_argument('--best_imputation_MRE_Threshold', type=float, default=2.0, help='模型保存最优策略阈值')
    parser.add_argument('--min_mae_loss', type=float, default=0.5, help='模型验证阈值')

    parser.add_argument('--log_saving', type=str, default='./logs', help='日志存放目录')

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
