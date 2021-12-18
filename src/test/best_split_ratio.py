from sklearn.model_selection import GridSearchCV
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

SPLIT_RATIO = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
for split_ratio in SPLIT_RATIO:
    X_train, y_train, X_test, y_test = split_data(dataset, split_ratio=split_ratio)
    model = NB(alpha=1.5)
    model.fit(X_train, y_train, verbose=0)
    classification = model.predict(X_test, verbose=0)
    conf_matrix = confusion_matrix(y_test, classification)
    print("\n----------------------------------------------")
    print("::-> Split ratio = ", split_ratio)
    print("Confusion Matrix : \n", conf_matrix, "\n")
    print("Accuracy  = ", accuracy(conf_matrix))
    print("Recall    = ", recall(conf_matrix))
    print("Precision = ", precision(conf_matrix))
    print("----------------------------------------------")
