
import os
import random
import glob
import pandas as pd
from sklearn.model_selection import train_test_split

import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

import deepmatcher as dm

#数据处理：加载和处理标记的训练、验证和测试 CSV 数据。
train, validation, test = dm.data.process(
    path=r'G:\PycharmProject\deepmatcher-master\exp_datasets\1-amazon_google',
    train='train.csv',
    validation='valid.csv',
    test='test.csv'
)
#模型定义：指定神经网络架构。使用内置混合动力 模型（如我们论文的第 4.4 节所述）默认。能 根据您的内心愿望进行定制。
model = dm.MatchingModel(attr_summarizer='hybrid')
#模型训练：训练神经网络。
model.run_train(train,
                validation,
                epochs=10,
                batch_size=16,
                best_save_path='hybrid_model.pth',
                pos_neg_ratio=2
                )
#应用：在测试集上评估模型并应用于未标记的数据。

model.run_eval(test)

unlabeled = dm.data.process_unlabeled(path=r'G:\PycharmProject\deepmatcher-master\exp_datasets\1-amazon_google\unlabeled.csv', trained_model=model)
model.run_prediction(unlabeled)

