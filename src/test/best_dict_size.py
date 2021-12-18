from src.models.naive_bayes import NB
from src.utils.import_data import load_dataset
from src.utils.preprocessing import preprocessing
from src.utils.split_data import split_data
from src.utils.analytics import confusion_matrix, accuracy, recall, precision


DATA_PATH = '../../data/FinalStemmedSentimentAnalysisDataset.csv'
DATASET_SIZE = -1
dataset = load_dataset(DATA_PATH)
print("::-> Dataset loaded")

dataset = preprocessing(dataset, delate_nan=True, shuffle=True, clean_text=False)

if DATASET_SIZE != -1:
    dataset = dataset.iloc[0:DATASET_SIZE, :]

dataset = preprocessing(dataset, delate_nan=False, shuffle=False, clean_text=True)

MINIMUM_NUMBER_OF_APPEARANCES = [1, 2, 4, 5, 10, 15, 20, 25, 50]

# for minimum_appearances in MINIMUM_NUMBER_OF_APPEARANCES:
