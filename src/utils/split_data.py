import pandas as pd

def split_data(dataset: pd.DataFrame, split_ratio: float):
    """
    Splits the dataset in train and test
    :param dataset: pd.DataFrame Object data
    :param split_ratio: Proportion of Test and Train data
    :return: pd.DataFrame train dataset, pd.DataFrame test dataset
    """
    test_size = int(len(dataset.index) * (1 - split_ratio))
    return dataset.iloc[:test_size, :].iloc[:, 1].values, dataset.iloc[:test_size, :].iloc[:, 3].values, dataset.iloc[test_size+1:, :].iloc[:, 1].values, dataset.iloc[test_size+1:, :].iloc[:, 3].values