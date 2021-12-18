from utils.import_data import load_dataset
from utils.preprocessing import preprocessing
from utils.analytics import confusion_matrix, accuracy, recall, precision, count_class_type
from utils.split_data import split_data
from models.naive_bayes import NB

# Small or Large
DATASET_SMALL = False
# If -1, no reduction applied
DATASET_SIZE = 1000
SPLIT_RATIO = 0.2


if DATASET_SMALL:
    print("::-> Testing with Small DataSet")
    DATA_PATH = '../data/FinalStemmedSentimentAnalysisDatasetSmall.csv'
else:
    print("::-> Testing with Large DataSet")
    DATA_PATH = '../data/FinalStemmedSentimentAnalysisDataset.csv'


dataset = load_dataset(DATA_PATH)

dataset = preprocessing(dataset, delate_nan=True, shuffle=True, clean_text=True)

if DATASET_SIZE != -1:
    dataset = dataset.iloc[0:DATASET_SIZE, :]

count_class_type(dataset, 'sentimentLabel')

X_train, y_train, X_test, y_test = split_data(dataset, split_ratio=SPLIT_RATIO)

model = NB(alpha=1.5)
model.fit(X_train, y_train, verbose=1)

classification = model.predict(X_test, verbose=1)

conf_matrix = confusion_matrix(y_test, classification)
print("\nConfusion Matrix : \n", conf_matrix, "\n")
print("Accuracy  = ", accuracy(conf_matrix))
print("Recall    = ", recall(conf_matrix))
print("Precision = ", precision(conf_matrix))


"""
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X_train, y_train, cv=5)
print(scores)
"""
