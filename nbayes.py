# #pip install kagglehub

# import kagglehub

# # Download latest version
# path = kagglehub.dataset_download("reihanenamdari/breast-cancer")

# print("Path to dataset files:", path)

#Implementing Multinomial and Multiclass Naive Bayes From Scratch Using Breast Cancer Dataset (https://www.kaggle.com/datasets/reihanenamdari/breast-cancer/data)

# region ----- LIBRARIES / PACKAGES -----
import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
# endregion

# region ----- IMPORT AND PROCESS DATASET -----
# Read dataset (using pandas simply for exposure's sake, using numpy would be simpler for my purposes)
data = pd.read_csv("Breast_Cancer.csv",header=0)

# Turn all feature values into ordinal encoding (i.e. T1 -> 1, T2 -> 2, T3 -> 3); nominal encoding would be overcomplicating since we're just counting the data
# Encoding tumor size according to range
label_encoder = LabelEncoder()
# print(data.columns)
data["T Stage "] = label_encoder.fit_transform(data["T Stage "])
data["N Stage"] = label_encoder.fit_transform(data["N Stage"])
data["differentiate"] = label_encoder.fit_transform(data["differentiate"])
data["Grade"] = label_encoder.fit_transform(data["Grade"])
data["A Stage"] = label_encoder.fit_transform(data["A Stage"])
tumor_bins = [0, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99, 109, 119, 129, 139, 149]
tumor_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
data["Tumor Size"] = pd.cut(data["Tumor Size"], bins = tumor_bins, labels = tumor_labels, include_lowest=True)
data["Estrogen Status"] = label_encoder.fit_transform(data["Estrogen Status"])
data["Progesterone Status"] = label_encoder.fit_transform(data["Progesterone Status"])
# Regional Node Examined does not seem pertinent to classification, however we instead use "Reginol Node Positive" / "Regional Node Examined"
# We cannot use Regional Node Positive in conjunction with this relation as we would be violating the assumption of feature independence
data["Regional Node Examined"] = data["Reginol Node Positive"] / data["Regional Node Examined"]
# Scale of 1-100%, split into groups by 5% buckets
data["Regional Node Examined"] = pd.cut(data["Regional Node Examined"], bins=20, labels=False, include_lowest=True)
data.rename(columns={"Regional Node Examined" : "% Positive/Examined Regional Nodes"}, inplace=True)
months_bins = [0, 11, 23, 35, 47, 59, 71, 83, 95, 107]
months_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8]
data["Survival Months"] = pd.cut(data["Survival Months"], bins = months_bins, labels = months_labels, include_lowest=True)
# endregion



# region ----- FEATURE SELECTION -----
# Separate our features and our classifications; drop features that do not directly contribute to the classification of cancer stage

X = data.drop(columns=["Age", "Race", "Marital Status", "Status", "6th Stage", "Reginol Node Positive"], axis=1)
y = data["6th Stage"]
y = label_encoder.fit_transform(data["6th Stage"])

# Convert to numpy for easier data manipulation
X = X.to_numpy(dtype=int)

# Split data into training and testing 8:2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# endregion



# region ----- Print commands for checking training and testing X and y arrays -----

# np.set_printoptions(threshold=sys.maxsize, suppress=True)

# print(data)

# pd.set_option('display.max_rows', None)
# print(X)
# print(y)

# print("X_train has " + str(len(X_train)) + " rows.")
# print(X_train)
# print("X_test has " + str(len(X_test)) + " rows.")
# print(X_test)

# print("y_train has " + str(len(y_train)) + " rows.")
# print(y_train)
# print("y_test has " + str(len(y_test)) + " rows.")
# print(y_test)
# endregion




# region ----- IMPLEMENT NAIVE BAYES LEARNING ----- 
# Essentially trying to return prior probabilities and conditional probabilties calculated from the training set; then given a row from testing set, we try to predict what class it would be
def learnnb(X, Y):

    # rows (# of entries) and columns (# of features)
    m, n = X.shape

    
    # Prior probabilities; we should have 5 because there are 5 stages we can categorize them into (IIA, IIB, IIIA, IIIB, IIIC)
    priorIIA, priorIIB, priorIIIA, priorIIIB, priorIIIC = (y==0).sum(), (y==1).sum(), (y==2).sum(), (y==3).sum(), (y==4).sum()
    # priorIIA, priorIIB, priorIIIA, priorIIIB, priorIIIC = priorIIA / m, priorIIB / m, priorIIIA / m, priorIIIB / m, priorIIIC / m

    #priorp: 5x1 (prior probabilities)
    
    priorp = np.array([priorIIA / m, priorIIB / m, priorIIIA / m, priorIIIB / m, priorIIIC / m])

    # Conditional probabilties; will be a df of size classes x features x unique features
    # Counts all numbers of instances of feature

    # Laplace smoothing
    condp = np.ones((5, len(X[0]), X.max()+1))
    # print(condp.shape)


    # Counting all the
    # for each entry in csv
    for i in range(len(X)):
        # per feature
        for j in range(len(X[0])):
            # y[i] = class at i'th row, j = feature column, X[i][j] = unique feature value
            condp[Y[i]][j][X[i][j]] += 1

    # print(X.shape)
    # print(condp.sum())

    condp[0] /= priorIIA
    condp[1] /= priorIIB
    condp[2] /= priorIIIA
    condp[3] /= priorIIIB
    condp[4] /= priorIIIC


    # print(priorIIA, priorIIB, priorIIIA, priorIIIB, priorIIIC)

    return (priorp, condp)
    
# endregion




# region ----- IMPLEMENT NAIVE BAYES PREDICTION -----
def predictnb(X_test, model):
    # print(X_test)
    priorp, condp = model

    predict = np.empty((len(X_test)), dtype=int)

    for i in range(len(X_test)):
        
        postp = np.array([condp[0][0][X_test[i][0]], condp[1][0][X_test[i][0]], condp[2][0][X_test[i][0]], condp[3][0][X_test[i][0]], condp[4][0][X_test[i][0]]])
        # per feature
        for j in range(1, len(X_test[0])):
            postp = postp * np.array([condp[0][j][X_test[i][j]], condp[1][j][X_test[i][j]], condp[2][j][X_test[i][j]], condp[3][j][X_test[i][j]], condp[4][j][X_test[i][j]]])
        
        # [postIIA, postIIB, postIIIA, postIIIB, postIIIC]
        # print(postp)
        postp = postp * priorp
        normalize = sum(postp)
        postp /= normalize
        # print(postp)
        predict[i] = np.argmax(postp)
        # print(predict[i])
    return predict
# endregion




# region ----- IMPLEMENT NAIVE BAYES TESTING -----
def testnb(predict, y_test):
    # print(predict[0:50])
    # print(y_test[0:50])
    return (predict!=y_test).mean()
# endregion




# region ----- TEST NAIVE BAYES PREDICTION -----

# Run Naive Bayes implementation
model = learnnb(X_train,y_train)
predict = predictnb(X_test, model)

# Testing sklearn module's multinomial naive bayes w/ one-count smoothing compared to functions above
sk_model = MultinomialNB(alpha=1.0)
sk_model.fit(X_train, y_train)
sk_predict = sk_model.predict(X_test)


print("Naive Bayes error rate: " + str(f"{testnb(predict,y_test) * 100:.3g}") + "%")
print("Scikit Learn error rate: " + str(f"{testnb(sk_predict, y_test) * 100:.3g}") + "%")
