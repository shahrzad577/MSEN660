
"Assignmnet 2: Classification using wrapper feature selection"

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import scipy
from sklearn import metrics
import itertools
import matplotlib.pyplot as plt

project_path = "H:\\tamu\\courses\\MSEN660\\Homeworks\\ProblemSet_2\\project_2\\"
file_path = project_path + 'Stacking_Fault_Energy_Dataset.txt'
original_data  = pd.read_table(file_path)

# pre-process the data
num_all_samples = original_data.shape[0]                                       # original number of training points
features = original_data.columns[:-1]                                    # original features
feature_sum_percentage = np.sum(original_data.iloc[:, :-1] > 0) / num_all_samples   # fraction of nonzero components for each feature
features_to_drop = features[feature_sum_percentage<0.6]            # features with less than 60% nonzero components
data_1 = original_data.drop (features_to_drop, axis=1)                # drop those features
sample_min = data_1.min(axis=1)                                  # finding samples with zero values
data_2 = data_1[sample_min != 0]                                # drop sample points with any zero values
SFE_data = data_2[(data_2.SFE < 35) | (data_2.SFE > 45)]          # get a subset of dataframe with condition

Y = SFE_data.SFE > 40

# Assignmnet 1- a:

split = int(0.2 * SFE_data.shape[0])
train_data , test_data = SFE_data[:split], SFE_data[split:]

X_train = train_data.iloc[:,:-1]
Y_train = train_data.SFE>40   # High SFE will be labled as true (1) and low SFE will be labeled as false (0)
# High SFE = class 1 / low SFE = class 0

X_test = test_data.iloc[:, :-1]
Y_test = test_data.SFE>40


## Assignmnet 2: Classification using wrapper feature selection

# 1) exhausive search (for 1 to 5 variables)

def get_combination(iterable, k_features):
    comb_list = []

    for subset in itertools.combinations(iterable, k_features):
        comb_list.append(subset)
    comb_feature = [list(comb_list[i]) for i in range(len(comb_list))]
    return comb_feature


def get_classifier(train_data, feature):
    xtrain = train_data[feature]
    Y_train = train_data.SFE > 40
    ytrain = Y_train.astype(int)
    classifier = LDA()
    classifier.fit(xtrain, ytrain)
    return classifier

def get_accuracy(data, feature, classifier):
    x = data[feature]
    y1 = data.SFE > 40
    y_org = y1.astype(int)
    y_pred = classifier.predict(x)
    acc = metrics.accuracy_score(y_org, y_pred)
    return acc


def ExhaustiveSearchFeatureSelection(train_data, test_data, k_features):
    df_acc = pd.DataFrame(columns=['features', 'accuracy_train', 'accuracy_test'])

    X_train = train_data.iloc[:, :-1]
    iterable = X_train.columns.values
    comb_feature = get_combination(iterable, k_features)

    for feature in comb_feature:
        classifier = get_classifier(train_data, feature)
        acc_train = get_accuracy(train_data, feature, classifier)
        #acc = np.round(acc, 4)
        df_acc = df_acc.append({'features': feature, 'accuracy_train': acc_train}, ignore_index=True)

    train_acc = max(df_acc.accuracy_train)
    max_acc_features = df_acc.loc[df_acc['accuracy_train'] == train_acc, 'features'].tolist()
    selected_features = max_acc_features[0]

    #getting accuracy on test using the classifier of selected features
    cls = get_classifier(train_data, selected_features)
    test_acc = get_accuracy(test_data,selected_features,cls)

    return train_acc, test_acc, selected_features


# get lists of the results
def get_featuresets(train_data, min_feature, max_feature):
    train_acc_list = []
    test_acc_list =[]
    featuresets_list = []

    for i in range(min_feature, max_feature + 1):
        train_acc, test_acc, selected_features = ExhaustiveSearchFeatureSelection(train_data, test_data, k_features=i)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        featuresets_list.append(selected_features)

    return train_acc_list, test_acc_list, featuresets_list


train_acc_list, test_acc_list, featuresets_list = get_featuresets(train_data, min_feature=1, max_feature=5)

print(train_acc_list)
print (test_acc_list)
print(featuresets_list)

efs_cols = {'EFS/n_variable':[1,2,3,4,5], 'selected_features': featuresets_list,
            'error on train': [1-i for i in train_acc_list], 'error on test':[1- i for i in test_acc_list]}
efs_DataFrame = pd.DataFrame(efs_cols)
print(efs_DataFrame)

def SequentialForwardSearch(train_data, test_data, max_feat):
    feat_list = []
    train_acc_list = []
    test_acc_list = []
    iterable = X_train.columns.values

    efs= ExhaustiveSearchFeatureSelection(train_data, test_data, k_features=1)
    train_acc_list.append(efs[0])
    test_acc_list.append(efs[1])
    selected_features = efs[2]

    feat_list.append(selected_features)
    initial_list = selected_features  # selected_features will be overwritten in the while loop

    counter = 0
    while (counter < max_feat-1):
        comb_feature = get_combination([i for i in iterable if not i in initial_list], k_features=1)
        new_set = [initial_list + i for i in comb_feature]

        df_acc = pd.DataFrame(columns=['features', 'accuracy'])
        for set in new_set:
            classifier = get_classifier(train_data, feature= set)
            acc = get_accuracy(train_data, feature=set, classifier = classifier)
            acc = np.round(acc, 4)
            df_acc = df_acc.append({'features': set, 'accuracy': acc}, ignore_index=True)

        acc_max = max(df_acc.accuracy)
        max_acc_features = df_acc.loc[df_acc['accuracy'] == acc_max, 'features'].tolist()
        selected_features = max_acc_features[0]
        feat_list.append(selected_features)
        train_acc_list.append(acc_max)
        initial_list = selected_features

        # getting accuracy on test using the classifier of selected features
        cls = get_classifier(train_data, selected_features)
        test_acc = get_accuracy(test_data, selected_features, cls)
        test_acc_list.append(test_acc)

        counter += 1

    return train_acc_list, test_acc_list, feat_list

train_acc_list, test_acc_list, feat_list = SequentialForwardSearch (train_data, test_data, max_feat=5)
print('================================')
print(train_acc_list)
print(test_acc_list)
print(feat_list)

sfs_cols = {'SFS/n_variable':[1,2,3,4,5], 'selected_features': feat_list,
            'error on train': [1-i for i in train_acc_list], 'error on test':[1- i for i in test_acc_list]}
sfs_DataFrame = pd.DataFrame(sfs_cols)
print(sfs_DataFrame)







