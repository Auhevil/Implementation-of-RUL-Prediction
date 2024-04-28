import matplotlib.pyplot as plt
import os
import datetime


def visualize(result, rmse, datasetName, epoch):
    # get current date and time
    now = datetime.datetime.now()
    formatted_date = now.strftime("%Y-%m-%d_%H-%M-%S")  # 格式化日期时间
    # format path of savepath
    savefigpath = './Pics/{}/Transformer({})_{}-epoch_{}.png'.format(datasetName, rmse, formatted_date, epoch)
    # make sure dir exists
    os.makedirs(os.path.dirname(savefigpath), exist_ok=True)

    # set rul_last
    if datasetName in ['FD002', 'FD004']:
        rul_last = 250
    else:
        rul_last = 100

    # the true remaining useful life of the testing samples
    true_rul = result.iloc[:, 0:1].to_numpy()
    # the predicted remaining useful life of the testing samples
    pred_rul = result.iloc[:, 1:].to_numpy()

    plt.figure(figsize=(10, 6))
    plt.axvline(x=rul_last, c='r', linestyle='--')
    plt.plot(true_rul, label='Actual Data')
    plt.plot(pred_rul, label='Predicted Data')
    plt.title('RUL Prediction on CMAPSS - {}(epoch={})'.format(datasetName, epoch))
    plt.legend()
    plt.xlabel("Samples")
    plt.ylabel("Remaining Useful Life")
    plt.savefig(savefigpath)
    plt.grid()
    plt.show()
