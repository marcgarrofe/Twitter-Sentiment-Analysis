from src.models.naive_bayes import NB
from src.utils.import_data import load_dataset
from src.utils.preprocessing import preprocessing
from src.utils.split_data import split_data
from src.utils.analytics import confusion_matrix, accuracy, recall, precision

import random
def delete_random_elems(input_list, n):
    to_delete = set(random.sample(range(len(input_list)), n))
    return [x for i,x in enumerate(input_list) if not i in to_delete]

DATA_PATH = '../../data/FinalStemmedSentimentAnalysisDataset.csv'
DATASET_SIZE = -1
SPLIT_RATIO = 0.2
DICT_SIZE = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

dataset = load_dataset(DATA_PATH)
print("::-> Dataset loaded")

dataset = preprocessing(dataset, delate_nan=True, shuffle=True, clean_text=False)

if DATASET_SIZE != -1:
    dataset = dataset.iloc[0:DATASET_SIZE, :]

dataset = preprocessing(dataset, delate_nan=False, shuffle=False, clean_text=True)

for size in DICT_SIZE:
    X_train, y_train, X_test, y_test = split_data(dataset, split_ratio=SPLIT_RATIO)
    model = NB(alpha=1.5)
    model.fit(X_train, y_train, verbose=0)
    dict_elements = model.vocabulary.items()
    dict_elements = delete_random_elems(dict_elements, int(len(dict_elements) * size))
    model.vocabulary = dict(dict_elements)
    classification = model.predict(X_test, verbose=0)
    conf_matrix = confusion_matrix(y_test, classification)
    print("\n----------------------------------------------")
    print("::-> Size = ", size)
    print("Confusion Matrix : \n", conf_matrix, "\n")
    print("Accuracy  = ", accuracy(conf_matrix))
    print("Recall    = ", recall(conf_matrix))
    print("Precision = ", precision(conf_matrix))
    print("----------------------------------------------")
