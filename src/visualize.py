import matplotlib.pyplot as plt
import os
import datetime
import torch


def visualize(result, rmse, datasetName, epoch, LR):
    # get current date and time
    # rmse_float = rmse * 1000
    # rmse_float = torch.round(rmse_float)
    # rmse_float /= 1000
    # print(rmse_float)
    now = datetime.datetime.now()
    formatted_date = now.strftime("%Y-%m-%d_%H-%M-%S")  # 格式化日期时间
    # format path of savepath
    # savefigpath = './Pics/{}/Transformer({})-epoch({})-LR({})_{}.png'.format(datasetName, rmse, epoch, LR, formatted_date)
    savefigpath = './Pics/{}/LSTM({})-epoch({})-LR({})_{}.png'.format(datasetName, rmse, epoch, LR, formatted_date)
    # make sure dir exists
    os.makedirs(os.path.dirname(savefigpath), exist_ok=True)

    # set rul_last
    if datasetName in ['FD001', 'FD003']:
        rul_last = 100
    elif datasetName == 'FD002':
        rul_last = 259
    else :
        rul_last = 248

    # the true remaining useful life of the testing samples
    true_rul = result.iloc[:, 0:1].to_numpy()
    # the predicted remaining useful life of the testing samples
    pred_rul = result.iloc[:, 1:].to_numpy()
    # config fonts of pic

    plt.figure(figsize=(10, 6))
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.axvline(x=rul_last, c='r', linestyle='--')
    plt.plot(true_rul, label='Actual Data', linewidth=2.0)
    plt.plot(pred_rul, label='Predicted Data(RMSE={})'.format(rmse), linewidth=2.0)
    # plt.title('RUL Prediction with Transformer on CMAPSS-{}(epoch={},LR={})'.format(datasetName, epoch, LR), fontsize=14)
    plt.title('RUL Prediction with LSTM on CMAPSS-{}(epoch={},LR={})'.format(datasetName, epoch, LR), fontsize=14)
    plt.legend(fontsize=12)
    plt.xlabel("Samples", fontsize=14)
    plt.ylabel("Remaining Useful Life", fontsize=14)
    plt.savefig(savefigpath, dpi=600, bbox_inches='tight', pad_inches=0.0)
    plt.grid()
    plt.show()
