import pandas as pd
from src.utils.preprocessing import process_tweets


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
