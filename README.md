# Statement of Project

## GCU-Transformer for RUL Prediction (C-MAPSS)

An implementation with GCU-Transformer with PyTorch for remaining useful life prediction on C-MAPSS.
_Author: Haoren Guo, National University of Singapore_

## Quick Run

1. type the command `bash train.sh` to run this project
2. `train.sh` contains:
   1. **Hyperparameters** of model
   2. **Dataset** for trainning
   3. **Path** of models and logs to save
   4. **Path** of imgs to save
   5. **Mode** switching between train and test

## Testing

Change `MODES='Train'` to `MODES='test'` and change the `MODEL_PATH` to the model you saved.

## File Structure

```  text
.
|-- CMAPSSData          训练所用的C-MAPSS数据集，包含了四个子数据集:FD001、FD002、FD003、FD004
|-- Pics                模型测试结果图片
|   |-- FD001
|   |-- FD002
|   |-- FD003
|   `-- FD004
|-- __pycache__
|-- saved_model         训练好的模型(pytorch导出)
|   |-- log
|   `-- model
|-- src  模型源码
|   |-- __pycache__
|   |-- data_process    数据处理函数
|   |-- transformer     transformer模型
|   `-- utils           一些工具包
`-- venv                模型所用的虚拟环境
   |-- Lib
   `-- Scripts
```

## Environment Details

``` python
python==3.8.8
numpy==1.20.1
pandas==1.2.4
matplotlib==3.3.4
pytorch==1.8.1
```

## Credit

This work is inpired by Mo, Y., Wu, Q., Li, X., & Huang, B. (2021). Remaining useful life estimation via transformer encoder enhanced by a gated convolutional unit. Journal of Intelligent Manufacturing, 1-10.

## License

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
