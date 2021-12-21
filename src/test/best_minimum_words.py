from src.models.naive_bayes import NB
from src.utils.import_data import load_dataset
from src.utils.preprocessing import preprocessing
from src.utils.split_data import split_data
from src.utils.analytics import confusion_matrix, accuracy, recall, precision


DATA_PATH = '../../data/FinalStemmedSentimentAnalysisDataset.csv'
DATASET_SIZE = -1
SPLIT_RATIO = 0.2

dataset = load_dataset(DATA_PATH)
print("::-> Dataset loaded")

dataset = preprocessing(dataset, delate_nan=True, shuffle=True, clean_text=False)

if DATASET_SIZE != -1:
    dataset = dataset.iloc[0:DATASET_SIZE, :]

dataset = preprocessing(dataset, delate_nan=False, shuffle=False, clean_text=True)

MINIMUM_NUMBER_OF_APPEARANCES = [10000, 20000, 30000, 40000, 50000]
X_train, y_train, X_test, y_test = split_data(dataset, split_ratio=SPLIT_RATIO)

# for minimum_appearances in MINIMUM_NUMBER_OF_APPEARANCES:
list_accuracy = list()
list_recall = list()
list_precision = list()

for minimum_appearances in MINIMUM_NUMBER_OF_APPEARANCES:
    model = NB(alpha=1.5)
    model.minimum_appearances = minimum_appearances
    model.fit(X_train, y_train, verbose=0)

    classification = model.predict(X_test, verbose=0)

    conf_matrix = confusion_matrix(y_test, classification)
    """
    print("\n----------------------------------------------")
    print("::-> Dict minimum Size = ", minimum_appearances)
    print("Confusion Matrix : \n", conf_matrix)
    print("Accuracy  = ", accuracy(conf_matrix))
    print("Recall    = ", recall(conf_matrix))
    print("Precision = ", precision(conf_matrix))
    print("----------------------------------------------")
    """
    # print(minimum_appearances, accuracy(conf_matrix), recall(conf_matrix), precision(conf_matrix))
    list_accuracy.append(accuracy(conf_matrix))
    list_recall.append(recall(conf_matrix))
    list_precision.append(precision(conf_matrix))

for i in list_accuracy:
    print(i)

print(" ")

for i in list_recall:
    print(i)

print(" ")

for i in list_precision:
    print(i)

