from src.utils.import_data import load_dataset
from src.utils.preprocessing import preprocessing
from src.utils.analytics import confusion_matrix, accuracy, recall, precision, count_class_type
from src.utils.split_data import split_data
from src.models.naive_bayes import NB
import os


# Small or Large
DATASET_SMALL = False
# If -1, no reduction applied
DATASET_SIZE = -1
SPLIT_RATIO = 0.2
ALPHA = 1.7

sys_path = os.path.dirname(os.path.abspath(__file__))

if DATASET_SMALL:
    print("\n::-> Testing with Small DataSet")
    DATA_PATH = sys_path + '/data/FinalStemmedSentimentAnalysisDatasetSmall.csv'
else:
    print("\n::-> Testing with Large DataSet")
    DATA_PATH = sys_path + '/data/FinalStemmedSentimentAnalysisDataset.csv'

if DATASET_SIZE != -1:
    print("\n::-> Testing with ", DATASET_SIZE, " rows")
else:
    print("\n::-> Testing with ALL the rows")

print("     Alpha = ", ALPHA)
print("     Split Ratio = ", SPLIT_RATIO, "\n")

dataset = load_dataset(DATA_PATH)

dataset = preprocessing(dataset, delate_nan=True, shuffle=True, clean_text=True)

if DATASET_SIZE != -1:
    dataset = dataset.iloc[0:DATASET_SIZE, :]

count_class_type(dataset, 'sentimentLabel')

X_train, y_train, X_test, y_test = split_data(dataset, split_ratio=SPLIT_RATIO)

model = NB(alpha=ALPHA)
model.fit(X_train, y_train, verbose=1)

classification = model.predict(X_test, verbose=1)

conf_matrix = confusion_matrix(y_test, classification)
print("\nConfusion Matrix : \n", conf_matrix, "\n")
print("Accuracy  = ", accuracy(conf_matrix))
print("Recall    = ", recall(conf_matrix))
print("Precision = ", precision(conf_matrix))
