from sklearn import logger
from arguments import *
from dataloader import UnifiedDataLoader,normalize_3d_array
args = AttnWGAIN_arguments()
if args.model == 'AttnWGAIN':
    from AttnWGAIN import Generator, Discriminator, G_loss, D_loss
import h5py
import torch
import torch.autograd as autograd
import logging
import numpy as np
import os
import copy

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from datetime import datetime
import logging

def seed_all(seed):
    if not seed:
        seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

def setup_logger(log_file_path, log_name, mode='a'):
    """set up log file
    mode : 'a'/'w' mean append/overwrite,
    """
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(log_file_path, mode=mode)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.propagate = False  # prevent the child logger from propagating log to the root logger (twice), not necessary
    return logger

def MAE_loss(Hat_X,X,mask):
    return torch.sum(torch.abs(X - Hat_X) * mask) / (torch.sum(mask) + 1e-9)

def masked_mse_cal(inputs, target, mask):
    return torch.sum(torch.square(inputs - target) * mask) / (torch.sum(mask) + 1e-9)

def masked_rmse_cal(inputs, target, mask):
    return torch.sqrt(masked_mse_cal(inputs, target, mask))

def masked_mre_cal(inputs, target, mask):
    return torch.sum(torch.abs(inputs - target) * mask) / (torch.sum(torch.abs(target * mask)) + 1e-9)

def compute_gradient_penalty(args,discriminator, real_samples, fake_samples, H):
    alpha = torch.rand(real_samples.size()).to(args.device)
    interpolates = alpha * real_samples + (1 - alpha) * fake_samples
    interpolates.requires_grad_(True)
    d_interpolates = discriminator(interpolates, H)
    gradients = autograd.grad(outputs=d_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(d_interpolates.size()).to(args.device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def validate(args, generator, discriminator, val_dataloader, logger,epoch):
    generator.eval()
    discriminator.eval()

    evalX_collector, evalMask_collector, imputations_collector = [], [], []
    with torch.no_grad():
        for idx, data in enumerate(val_dataloader):
            indices, X, missing_mask, H, deltaPre, X_holdout, indicating_mask = map(lambda x: x.to(args.device), data)

            evalX_collector.append(X_holdout)
            evalMask_collector.append(indicating_mask)


            # 随机噪声 转为tensor
            Z = np.random.uniform(0, 0.01, size=X.shape)
            Z = torch.from_numpy(Z.astype('float32')).to(args.device)

            # 缺失数据部分+随机噪声
            X = missing_mask * X + (1 - missing_mask) * Z

            deltaPre_cpu = deltaPre.cpu().numpy()
            deltaPre_normalized = normalize_3d_array(deltaPre_cpu)
            deltaPre_normalized = torch.tensor(deltaPre_normalized, dtype=torch.float32).to('cuda')
            G_sample = generator(X, missing_mask, deltaPre_normalized)
            # logger.info(G_sample)
            # 将缺失部分的生成数据+未缺失部分的真实数据
            Hat_X = missing_mask * X + (1 - missing_mask) * G_sample
            imputations_collector.append(Hat_X)

        evalX_collector = torch.cat(evalX_collector)
        evalMask_collector = torch.cat(evalMask_collector)
        imputations_collector = torch.cat(imputations_collector)
        imputation_MAE = MAE_loss(imputations_collector, evalX_collector, evalMask_collector)
        # 添加 rmse 和 mre
        imputation_RMSE = masked_rmse_cal(imputations_collector, evalX_collector, evalMask_collector)
        imputation_MRE = masked_mre_cal(imputations_collector, evalX_collector, evalMask_collector)

        # mae
        logger.info(
            "[Epoch %d/%d] [validating MAE loss: %f]"
            % (epoch, args.epochs, imputation_MAE)  # loss_G.item()
        )
        ## 保存模型
        if imputation_MAE < args.best_imputation_MAE_Threshold:
            saving_path = os.path.join(args.saving_model_path, 'model_imputationMAE_{:.4f}'.format(imputation_MAE))
            torch.save({'model': generator.state_dict()}, saving_path)
            logger.info(
                "Save Model [model_imputation_MAE_%.4f] "
                % (imputation_MAE)  # loss_G.item()
            )
            args.best_imputation_MAE_Threshold = imputation_MAE
        ## 更新最优值
        if imputation_MAE < args.best_imputation_MAE:
            args.best_imputation_MAE = imputation_MAE

        # rmse
        logger.info(
            "[Epoch %d/%d] [validating RMSE loss: %f]"
            % (epoch, args.epochs, imputation_RMSE)  # loss_G.item()
        )
        ## 保存模型
        if imputation_RMSE < args.best_imputation_RMSE_Threshold:
            saving_path = os.path.join(args.saving_model_path, 'model_imputation_RMSE_{:.4f}'.format(imputation_RMSE))
            torch.save({'model': generator.state_dict()}, saving_path)
            args.best_imputation_RMSE_Threshold = imputation_RMSE
        ## 更新最优值
        if imputation_RMSE < args.best_imputation_RMSE:
            args.best_imputation_RMSE = imputation_RMSE

        # mre
        logger.info(
            "[Epoch %d/%d] [validating MRE loss: %f]"
            % (epoch, args.epochs, imputation_MRE)  # loss_G.item()
        )
        ## 保存模型
        if imputation_MRE < args.best_imputation_MRE_Threshold:
            saving_path = os.path.join(args.saving_model_path, 'model_imputationMRE_{:.4f}'.format(imputation_MRE))
            torch.save({'model': generator.state_dict()}, saving_path)
            args.best_imputation_MRE_Threshold = imputation_MRE
        ## 更新最优值
        if imputation_MRE < args.best_imputation_MRE:
            args.best_imputation_MRE = imputation_MRE
        logger.info(
            "---------------分割线-----------------"
        )

def impute_missing_data(model, train_data, val_data, test_data, logger):
    logger.info(f"Start imputing all missing data in all train/val/test sets...")
    model.eval()
    imputed_data_dict = {}
    with torch.no_grad():
        for dataloader, set_name in zip(
            [train_data, val_data, test_data], ["train", "val", "test"]
        ):
            indices_collector, imputations_collector = [], []
            for idx, data in enumerate(dataloader):
                if args.stage=='impute': 
                    # indices, X, missing_mask = map(lambda x: x.to(args.device), data)
                    indices, X, missing_mask, H, deltaPre, X_holdout, indicating_mask = map(lambda x: x.to(args.device), data)
                    inputs = {"indices": indices, "X": X, "missing_mask": missing_mask}
                    deltaPre_cpu = deltaPre.cpu().numpy()
                    deltaPre_normalized = normalize_3d_array(deltaPre_cpu)
                    deltaPre_normalized = torch.tensor(deltaPre_normalized, dtype=torch.float32).to('cuda')
                imputed_data = model(X, missing_mask,deltaPre_normalized)
                indices_collector.append(indices)
                imputations_collector.append(imputed_data)
            
            indices_collector = torch.cat(indices_collector)
            indices = indices_collector.cpu().numpy().reshape(-1)
            imputations_collector = torch.cat(imputations_collector)
            imputations = imputations_collector.data.cpu().numpy()
            ordered = imputations[np.argsort(indices)]  # to ensure the order of samples
            imputed_data_dict[set_name] = ordered
    imputation_saving_path = os.path.join(args.saving_impute_path, "imputations.h5")
    with h5py.File(imputation_saving_path, "w") as hf:
        hf.create_dataset("imputed_train_set", data=imputed_data_dict["train"])
        hf.create_dataset("imputed_val_set", data=imputed_data_dict["val"])
        hf.create_dataset("imputed_test_set", data=imputed_data_dict["test"])
    logger.info(f"Done saving all imputed data into {imputation_saving_path}.")

def auc(args,train_dataloader, val_dataloader,test_dataloader,logger=None):
    models = Generator(args.n_groups, args.n_group_inner_layers, args.seq_len, args.feature_num, args.d_model, args.d_inner, args.n_head, args.d_k, args.d_v, args.dropout, args.diagonal_attention_mask, args.device)
    saving_paths = []
    for file in os.listdir(args.saving_model_path):
        saving_paths.append(os.path.join(args.saving_model_path, file))
    try:
        for saving_path in saving_paths:
            state_dict = torch.load(saving_path)
            models.load_state_dict(state_dict['model'])
            models.to(args.device)
            models.eval()
            with torch.no_grad():             
                if args.stage=='impute':
                    impute_missing_data(models, train_dataloader, val_dataloader, test_dataloader,logger)
                    logger.info(f"save done!")
    except Exception:
        if args.stage == 'train':
            logger.info(Exception)
            logger.info("Abandon>0.22! The best_imputation_MAE:[%f]" % (args.best_imputation_MAE))
        else:
            print(Exception)

def test(args, test_dataloader, logger=None , train_data=None, val_data=None, test_data=None):

    generator = Generator(args.n_groups, args.n_group_inner_layers, args.seq_len, args.feature_num, args.d_model, args.d_inner, args.n_head, args.d_k, args.d_v, args.dropout, args.diagonal_attention_mask, args.device)

    saving_paths = []
    if args.stage == 'train':
        saving_paths.append(os.path.join(args.saving_model_path,'model_imputationMAE_{:.4f}'.format(args.best_imputation_MAE)))
    else:
        for file in os.listdir(args.saving_model_path):
            saving_paths.append(os.path.join(args.saving_model_path, file))
    try:

        for saving_path in saving_paths:
            state_dict = torch.load(saving_path)
            generator.load_state_dict(state_dict['model'])
            generator.to(args.device)
            generator.eval()
            evalX_collector, evalMask_collector, imputations_collector = [], [], []
            with torch.no_grad():
                for idx, data in enumerate(test_dataloader):
                    indices, X, missing_mask, H, deltaPre, X_holdout, indicating_mask = map(lambda x: x.to(args.device), data)

                    evalX_collector.append(X_holdout)
                    evalMask_collector.append(indicating_mask)


                    Z = np.random.uniform(0, 0.01, size=X.shape)
                    Z = torch.from_numpy(Z.astype('float32')).to(args.device)


                    X = missing_mask * X + (1 - missing_mask) * Z

                    deltaPre_cpu = deltaPre.cpu().numpy()
                    deltaPre_normalized = normalize_3d_array(deltaPre_cpu)
                    deltaPre_normalized = torch.tensor(deltaPre_normalized, dtype=torch.float32).to('cuda')
                    G_sample = generator(X, missing_mask, deltaPre_normalized)
                    Hat_X = missing_mask * X + (1 - missing_mask) * G_sample
                    imputations_collector.append(Hat_X)

                evalX_collector = torch.cat(evalX_collector)
                evalMask_collector = torch.cat(evalMask_collector)
                imputations_collector = torch.cat(imputations_collector)

                imputation_MAE = MAE_loss(imputations_collector, evalX_collector, evalMask_collector)
                imputation_RMSE = masked_rmse_cal(imputations_collector, evalX_collector, evalMask_collector)
                imputation_MRE = masked_mre_cal(imputations_collector, evalX_collector, evalMask_collector)
                if args.stage=='train':
                    logger.info(
                        "[saving_path,%s testing MAE loss: %f]"
                        % (saving_path,imputation_MAE)
                    )
                else:
                    logger.info(
                        "[saving_path,%s testing MAE loss: %f;RMSE loss: %f ;MRE loss: %f]"
                        % (saving_path, imputation_MAE, imputation_RMSE, imputation_MRE)
                    )
                    if imputation_MAE < args.best_imputation_MAE_Threshold:
                        logger.info(
                            "-----------------Save Model [best_testing MAE loss: %f;RMSE loss: %f ;MRE loss: %f] ------------------------"
                            % (imputation_MAE, imputation_RMSE, imputation_MRE)
                        )
                        args.best_imputation_MAE_Threshold = imputation_MAE
    except Exception:
        if args.stage == 'train':
            logger.info(Exception)
            logger.info("Abandon>0.22! The best_imputation_MAE:[%f]" % (args.best_imputation_MAE))

        else:
            print(Exception)

def trainAndValidate(args,train_dataloader, val_dataloader, logger):
    generator = Generator(args.n_groups, args.n_group_inner_layers, args.seq_len, args.feature_num, args.d_model, args.d_inner, args.n_head, args.d_k, args.d_v, args.dropout, args.diagonal_attention_mask, args.device)
    discriminator = Discriminator(args.n_groups, args.n_group_inner_layers, args.seq_len, args.feature_num, args.d_model, args.d_inner, args.n_head, args.d_k, args.d_v, args.dropout, args.diagonal_attention_mask, args.device)
    g_loss = G_loss()
    d_loss = D_loss()
    OPTIMIZER = {'adam': torch.optim.Adam, 'adamw': torch.optim.AdamW}

    optimizer_G = OPTIMIZER[args.optimizer_type](generator.parameters(), lr=args.lr,
                                                 weight_decay=args.weight_decay)
    optimizer_D = OPTIMIZER[args.optimizer_type](discriminator.parameters(), lr=args.lr,
                                                 weight_decay=args.weight_decay)

    if torch.cuda.is_available():
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        g_loss = g_loss.cuda()
        d_loss = d_loss.cuda()

    generator.train()
    discriminator.train()
    for epoch in range(args.epochs):
        for idx, data in enumerate(train_dataloader):
            indices, X, missing_mask, H, deltaPre, X_holdout, indicating_mask = map(lambda x: x.to(args.device), data)
            for j in range(args.epochs_D):
                Z = np.random.uniform(0, 0.01, size=X.shape)
                Z = torch.from_numpy(Z.astype('float32')).to(args.device)

                X_copy = copy.deepcopy(X)
                X = missing_mask * X + (1 - missing_mask) * Z
  
                deltaPre_cpu = deltaPre.cpu().numpy()
                deltaPre_normalized = normalize_3d_array(deltaPre_cpu)
                deltaPre_normalized = torch.tensor(deltaPre_normalized, dtype=torch.float32).to('cuda')
                G_sample = generator(X, missing_mask, deltaPre_normalized).detach()
                Hat_X = missing_mask * X + (1 - missing_mask) * G_sample
                D_prob = discriminator(Hat_X, H)
                loss_D = d_loss(X, missing_mask, G_sample, D_prob, X_holdout, indicating_mask, args.alpha)
                gradient_penalty = compute_gradient_penalty(args, discriminator, X_copy, G_sample, H)
                optimizer_D.zero_grad()
                (loss_D++ args.lambda_gp * gradient_penalty).backward()
                optimizer_D.step()
            for j in range(args.epochs_G):
                Z = np.random.uniform(0, 0.01, size=X.shape)
                Z = torch.from_numpy(Z.astype('float32')).to(args.device)
                X = missing_mask * X + (1 - missing_mask) * Z

                deltaPre_cpu = deltaPre.cpu().numpy()
                deltaPre_normalized = normalize_3d_array(deltaPre_cpu)
                deltaPre_normalized = torch.tensor(deltaPre_normalized, dtype=torch.float32).to('cuda')
                G_sample  = generator(X, missing_mask, deltaPre_normalized)
                Hat_X = missing_mask * X + (1 - missing_mask) * G_sample
                D_prob = discriminator(Hat_X, H).detach()
                loss_G = g_loss(X, missing_mask, G_sample, D_prob, X_holdout, indicating_mask, args.alpha)
                optimizer_G.zero_grad()
                loss_G.backward()
                optimizer_G.step()
            if idx % 10==9:
                logger.info(
                    "[Epoch %d/%d] [Batch %d] [D loss: %f] [G loss: %f]"
                    % (epoch, args.epochs, idx,  loss_D.item(), loss_G.item())  # loss_G.item()
                )
        validate(args, generator, discriminator, val_dataloader, logger,epoch)

def main():
    args = AttnWGAIN_arguments()

    unified_dataloader = UnifiedDataLoader(args.data_path, args.seq_len, args.feature_num, args.model_type,
                                           args.hint_rate,
                                           args.miss_rate, args.batch_size, args.num_workers, args.MIT)

    epochs_Gs = [1]
    hint_rates=[0]
    alphas=[[100,100]]
    for epochs_G in epochs_Gs:
        for hint_rate in hint_rates:
            for alpha in alphas:
                seed_all(args.seed)

                args.epochs_G = epochs_G
                args.hint_rate=hint_rate
                args.alpha=alpha

                args.best_imputation_MAE=1.0
                args.best_imputation_MAE_Threshold=0.9


                if args.stage == 'train':
                    time_now = datetime.now().__format__('%Y-%m-%d_T%H_%M_%S')
                    logger = setup_logger(os.path.join(args.log_saving, time_now+'_'+str(args.epochs_G)+'_'+str(args.hint_rate)+'_'+str(args.alpha)),time_now+'_'+str(args.lr)+'_'+str(args.hint_rate)+'_'+str(args.alpha), 'w')

                    logger.info(args)

                    train_dataloader, val_dataloader = unified_dataloader.get_train_val_dataloader()

                    trainAndValidate(args,train_dataloader, val_dataloader, logger)

                    test_dataloader = unified_dataloader.get_test_dataloader()
                    test(args, test_dataloader, logger)

                elif args.stage == 'test':
                    time_now = datetime.now().__format__('%Y-%m-%d_T%H_%M_%S')
                    logger = setup_logger(os.path.join(args.log_saving, time_now+'_'+str(args.epochs_G)+'_'+str(args.hint_rate)+'_'+str(args.alpha)),time_now+'_'+str(args.lr)+'_'+str(args.hint_rate)+'_'+str(args.alpha), 'w')
                    logger.info(args)
                    test_dataloader = unified_dataloader.get_test_dataloader()
                    test(args, test_dataloader, logger)

                elif args.stage == 'impute':
                    time_now = datetime.now().__format__('%Y-%m-%d_T%H_%M_%S')
                    logger = setup_logger(os.path.join(args.log_saving, time_now+'_'+str(args.epochs_G)+'_'+str(args.hint_rate)+'_'+str(args.alpha)),time_now+'_'+str(args.lr)+'_'+str(args.hint_rate)+'_'+str(args.alpha), 'w')
                    logger.info(args)
                    test_dataloader = unified_dataloader.get_test_dataloader()
                    train_data, val_data = unified_dataloader.get_train_val_dataloader()
                    test_data = unified_dataloader.get_test_dataloader()
                    auc(args, train_data, val_data ,test_data, logger)
                    


if __name__ == '__main__':
    main()
