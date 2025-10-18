model.run_eval(test)

unlabeled = dm.data.process_unlabeled(path=r'G:\PycharmProject\deepmatcher-master\exp_datasets\1-amazon_google\unlabeled.csv', trained_model=model)
model.run_prediction(unlabeled)