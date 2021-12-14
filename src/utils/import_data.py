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


def preprocessing(dataset, delate_nan=True, shufle=True):
    if delate_nan:
       dataset = dataset.dropna()
    if shufle:
        dataset = dataset.sample(frac=1).reset_index(drop=True)
    return dataset
