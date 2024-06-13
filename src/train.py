import torch

torch.manual_seed(1)
from model import *
from model_LSTM import *
from model_CNN import *
from data_process.data_processing import *
from data_process.loader import *
from torch.utils.data import DataLoader
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from utils.logger import init_logger
import datetime
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from model import GCU_Transformer



class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))


def Training(opt):
    PATH = opt.path + "/" + opt.dataset
    logsavepath = opt.logsave_path + "/" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") # Path of saved log
    logger = init_logger(logsavepath, opt, True)

    WRITER = SummaryWriter(log_dir=logsavepath)

    ##------load parameters--------##
    dataset = opt.dataset
    num_epochs = opt.epoch  # Number of training epochs
    d_model = opt.dim_en  # dimension in encoder
    heads = opt.head  # number of heads in multi-head attention
    N = opt.num_enc_layers  # number of encoder layers
    m = opt.num_features  # number of features

    batch_size = opt.batch_size
    train_seq_len = opt.train_seq_len
    test_seq_len = opt.test_seq_len

    patch_size = opt.patch_size
    LR = opt.LR
    smooth_param = opt.smooth_param
    ##------Model to CUDA------##

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    ##------load dataset--------##
    #  group_train: processed data, group by unit
    #  y_test: standard RUL for test dataset
    #  group_test&X_test: processed test dataset, group by unit
    group_train, y_test, group_test, X_test = data_processing(dataset, smooth_param)
    print("data processed")

    ##------sequence dataset--------##
    #   patch_size: generated size of sequence, the size of windows
    train_dataset = SequenceDataset(mode='train', group=group_train, sequence_train=train_seq_len,
                                    patch_size=train_seq_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    print("train loaded")

    test_dataset = SequenceDataset(mode='test', group=group_test, y_label=y_test, sequence_train=train_seq_len,
                                   patch_size=train_seq_len)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, drop_last=False)
    print("test loaded")

    ##------SAVE PATH--------##
    if opt.path == '':
        PATH = "train-model-" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".pth"
    else:
        PATH = PATH + "/" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if not os.path.exists(PATH):
        os.makedirs(PATH)

    logger.cprint("------Train-------")
    logger.cprint("------" + PATH + "-------")

    ##------model define--------##
    # setting of model is written in train.sh
    train_model = GCU_Transformer(seq_size=train_seq_len, patch_size=patch_size, in_chans=m,
                                   embed_dim=d_model, depth=N, num_heads=heads,
                                   decoder_embed_dim=d_model, decoder_depth=N, decoder_num_heads=heads,
                                   norm_layer=nn.LayerNorm, batch_size=batch_size)
    # train_model = LSTM1(group_train.get_group(1).shape[1] - 3, 96, 4, 1)
    # train_model = CNN1(m)
    print(train_model)

    # ------put model to GPU------#
    if torch.cuda.is_available():
        train_model = train_model.to(device)

    for p in train_model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    # criterion对应深度学习中的损失函数，optimization对应于优化器，用于梯度下降以更新参数
    criterion = torch.nn.MSELoss(reduction="mean")
    optimization = torch.optim.Adam(filter(lambda p: p.requires_grad, train_model.parameters()), lr=LR,
                                    weight_decay=opt.weight_decay)

    for p in train_model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)  # 服从均匀分布的glorot初始化器

    best_test_rmse = 10000
    for epoch in range(num_epochs):
        train_model.train()
        train_epoch_loss = 0

        iter_num = 0

        for X, y in train_loader:
            iter_num += 1

            if torch.cuda.is_available():
                X = X.cuda()
                y = y.cuda()

            y_pred = train_model.forward(X)

            loss = criterion(y_pred.reshape(y_pred.shape[0]), y)  # mse loss
            optimization.zero_grad()
            loss.backward()
            optimization.step()

            train_epoch_loss = train_epoch_loss + loss.item()

        train_epoch_loss = np.sqrt(train_epoch_loss / len(train_loader))

        WRITER.add_scalar('Train RMSE', train_epoch_loss, epoch)

        train_model.eval()
        with torch.no_grad():
            test_epoch_loss = 0
            res = 0
            for X, y in test_loader:
                if torch.cuda.is_available():
                    X = X.cuda()
                    y = y.cuda()

                y_hat_recons = train_model.forward(X)
                y_hat_unscale = y_hat_recons * 125

                subs = y_hat_unscale.reshape(y_hat_recons.shape[0]) - y
                subs = subs.cpu().detach().numpy()
                # calculate score
                if subs[0] < 0:
                    res = res + np.exp(-subs / 13)[0] - 1
                else:
                    res = res + np.exp(subs / 10)[0] - 1
                # calculate loss
                loss = criterion(y_hat_unscale.reshape(y_hat_recons.shape[0]), y)
                test_epoch_loss = test_epoch_loss + loss

            test_loss = torch.sqrt(test_epoch_loss / len(test_loader))
            WRITER.add_scalar('Test loss', test_loss, epoch)
            if epoch >= 10 and test_loss < best_test_rmse:
                best_test_rmse = test_loss
                best_score = res
                cur_best = train_model.state_dict()
                best_model_path = PATH + "/" + "new_best.pth"
                torch.save(cur_best, best_model_path)
                logger.cprint(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                logger.cprint("========New Best Test Loss Updata: %1.5f Best Score: %1.5f========" % (best_test_rmse, best_score))
                logger.cprint(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

        logger.cprint("Epoch: %d, training loss: %1.5f, testing rmse: %1.5f, score: %1.5f" % (
        epoch + 1, train_epoch_loss, test_loss, res))
        logger.cprint("------------------------------------------------------------")
    logger.cprint(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    logger.cprint("========Last Best Test Loss Updata: %1.5f Best Score: %1.5f========" % (best_test_rmse, best_score))
    logger.cprint(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    return
