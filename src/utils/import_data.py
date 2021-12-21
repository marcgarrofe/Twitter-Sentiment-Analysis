import pandas as pd
import os


def load_dataset(path: str):
    """
    Funció que donada la ruta del fitxer, retorna el dataset carregat en un objecte pandas
    :param path: String amb la ruta al fitxer
    :return: DataFrame amb les dades
    """
    assert os.path.isfile(path), "load_dataset() File doesn't exist"
    try:
        dataset = pd.read_csv(path, sep=';')
    except:
        print("ERROR : llegeix_taulell_doc()...")
    else:
        return dataset
