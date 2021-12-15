import pandas as pd


def load_dataset(path: str):
    """
    Funció que donada la ruta del fitxer, retorna el dataset carregat en un objecte pandas
    :param path: String amb la ruta al fitxer
    :return: DataFrame amb les dades
    """
    try:
        dataset = pd.read_csv(path, sep=';')
    except:
        print("::-> ERROR : llegeix_taulell_doc()...")
    else:
        return dataset


def preprocessing(dataset: pd.DataFrame, delate_nan=True, shuffle=True):
    """
    Given a dataset, process the columns by transforming the raw data to be more suitable for the estimators.
    :param dataset: DataFrame data object
    :param delate_nan: Boolean indicates if NaN is removed from the dataset
    :param shuffle: Boolean indicates if shuffle the data
    :return: Processed dataset
    """
    assert not dataset.empty, 'preprocessing() : DataFrame empty'

    if delate_nan:
       dataset = dataset.dropna()
    if shuffle:
        dataset = dataset.sample(frac=1).reset_index(drop=True)
    return dataset
