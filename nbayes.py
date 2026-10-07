# #pip install kagglehub

# import kagglehub

# # Download latest version
# path = kagglehub.dataset_download("reihanenamdari/breast-cancer")

# print("Path to dataset files:", path)
#Implementing Naive Bayes From Scratch Using Breast Cancer Dataset (https://www.kaggle.com/datasets/reihanenamdari/breast-cancer/data)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder




# ----- IMPORT AND PROCESS DATASET -----
# Read dataset
data = pd.read_csv("Breast_Cancer.csv",header=0)

# Turn all feature values into ordinal encoding (i.e. T1 -> 1, T2 -> 2, T3 -> 3); nominal encoding would be overcomplicating since we're just counting the data
# Encoding tumor size according to range
label_encoder = LabelEncoder()
print(data.columns)
data["T Stage "] = label_encoder.fit_transform(data["T Stage "])
data["N Stage"] = label_encoder.fit_transform(data["N Stage"])
data["differentiate"] = label_encoder.fit_transform(data["differentiate"])
data["Grade"] = label_encoder.fit_transform(data["Grade"])
data["A Stage"] = label_encoder.fit_transform(data["A Stage"])
tumor_bins = [0, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99]
tumor_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
data["Tumor Size"] = pd.cut(data["Tumor Size"], bins = tumor_bins, labels = tumor_labels, include_lowest=True)
data["Estrogen Status"] = label_encoder.fit_transform(data["Estrogen Status"])
data["Progesterone Status"] = label_encoder.fit_transform(data["Progesterone Status"])
# Regional Node Examined does not seem pertinent to classification, however we instead use "Reginol Node Positive" / "Regional Node Examined"
# We could argue that this inflates the impact of "Reginol Node Positive" on the learning, but the ratio of positive nodes to nodes observed may prove to be a useful statistic on top of gross # of positive nodes
data["Regional Node Examined"] = data["Reginol Node Positive"] / data["Regional Node Examined"]
data["Regional Node Examined"] = pd.cut(data["Regional Node Examined"], bins=20, labels=False, include_lowest=True)
# Reginol Node Positive we can leave as is
months_bins = [0, 11, 23, 35, 47, 59, 71, 83, 95, 107]
months_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8]
data["Survival Months"] = pd.cut(data["Survival Months"], bins = months_bins, labels = months_labels, include_lowest=True)


# ----- FEATURE SELECTION -----
# Separate our features and our classifications; drop features that do not directly contribute to the classification of cancer stage

X = data.drop(columns=["Age", "Race", "Marital Status", "Status", "6th Stage"], axis=1)
y = data["6th Stage"]

print(data)
# print(X.iloc[:,0])
print(X)
print(y)


# Split data into training and testing 8:2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Print commands for checking training and testing X and y arrays
# print("X_train has " + str(len(X_train)) + " rows.")
# print(X_train)
# print("X_test has " + str(len(X_test)) + " rows.")
# print(X_test)
# print("y_train has " + str(len(y_train)) + " rows.")
# print(y_train)
# print("y_test has " + str(len(y_test)) + " rows.")
# print(y_test)




# ----- IMPLEMENT NAIVE BAYES LEARNING ----- 
def learnnb(X, Y):

    # rows (# of entries) and columns (# of features)
    m, n = X.shape

    
    # Prior probabilities; we should have 5 because there are six stages we can categorize them into (IIA, IIB, IIIA, IIIB, IIIC)
    priorIIA, priorIIB, priorIIIA, priorIIIB, priorIIIC = (y=="IIA").sum(), (y=="IIB").sum(), (y=="IIIA").sum(), (y=="IIIB").sum(), (y=="IIIC").sum()
    priorIIA, priorIIB, priorIIIA, priorIIIB, priorIIIC = priorIIA / m, priorIIB / m, priorIIIA / m, priorIIIB / m, priorIIIC / m

    # print(priorIIA, priorIIB, priorIIIA, priorIIIB, priorIIIC)


    #


# learnnb(X_train,y_train)

