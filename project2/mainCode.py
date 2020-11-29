
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn import svm
import itertools
from sklearn.preprocessing import LabelEncoder
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC
from sklearn import metrics


file_path = 'H:\\tamu\\courses\\MSEN660\\Homeworks\\Computer_project2\\DataSet\\CMU-UHCS_Dataset\\'

def get_SVM_classifier(X_train, Y_train, cv):
    clf = svm.SVC(kernel='rbf', C=1, gamma='scale')
    clf.fit(X_train, Y_train)
    scores = cross_val_score(clf, X_train, Y_train, cv=cv)  #returns accuracy on model with k_fold cross validation
    mean_error = 1 - scores.mean()
    return clf, mean_error

def get_combination(iterable, k_features):
    comb_list = []
    for subset in itertools.combinations(iterable, k_features):
        comb_list.append(subset)
    comb_feature = [list(comb_list[i]) for i in range(len(comb_list))]
    return comb_feature

## call this function for train data since the array in .csv are read as string
def read_array_from_string(str):
    spaces_removed = ' '.join(str.split())
    brackets_removed = spaces_removed[1:-1].strip()  # .strip to trim the last space in the array
    numbers = brackets_removed.split(' ')
    arrr = []
    for s in numbers:
        arrr.append(float(s))
    return arrr


def get_key_with_min_second_value(dic):
    best_key = ''
    min_err = 10
    for k, v in dic.items():
        if v[1] < min_err:
            min_err = v[1]
            best_key = k

    return best_key

mic_label = ['spheroidite', 'network', 'pearlite', 'spheroidite+widmanstatten']

comb_label = get_combination(mic_label, k_features=2)  # get pairwise combination
models = ['feature_64_b1', 'feature_128_b2', 'feature_256_b3', 'feature_512_b4', 'feature_512_b5']
lb_make = LabelEncoder()  # for encoding labels from categorical to numerical)

#using pairwise model to get 6 pair-wise clasisifiers
def calc_pairwise_classifier(df_train, models):

    solution = {}
    for l in comb_label:
        solution[str(l)] = []

    for l in comb_label:

        train_data = df_train[(df_train['primary_microconstituent'] == l[0]) | (df_train['primary_microconstituent'] == l[1])]
        Y_train = train_data['labels']

        models_mean_err = {}    # 'model' -> [clf, err]
        for m in models:
            X = train_data[m].values
            X_train = [read_array_from_string(s) for s in X]
            clf, mean_error = get_SVM_classifier(X_train, Y_train, cv=10)
            models_mean_err[m] = [clf, mean_error]

        best_key = get_key_with_min_second_value(models_mean_err)
        solution[str(l)].append(best_key)  # best feature
        solution[str(l)].append(models_mean_err[best_key][0])  #best classifier
        solution[str(l)].append(models_mean_err[best_key][1])  # best error

    return solution


def get_multilabel_classifier(df_train, feature):

    train_data = df_train
    X = train_data[feature].values
    X_train = [read_array_from_string(s) for s in X]
    Y_train = train_data['labels']

    multilabel_cls = OneVsOneClassifier(LinearSVC(random_state=0)).fit(X_train, Y_train)

    return multilabel_cls


# for new image the selected 6 pairwise classifier and 1 multi_label classifier
# Given a new image, each of the six classifers is
# applied and then a vote is taken to achieve a consensus for the most often predicted label.
def test_error_pairwise_cls(df_test, best_features, best_cls, comb_label):

    most_predicted_label = []    ## but I need 6 list for each combination .. dictionary?
    predicted_labels = {{comb_label[0]: [], comb_label[1]: [], comb_label[2]: [],
                         comb_label[3]: [], comb_label[4]: [], comb_label[5]:[]}}

    for l in comb_label:
        i =0
        test_data = df_test[(df_test['primary_microconstituent'] == l[0]) | (df_train['primary_microconstituent'] == l[1])]
        X = test_data[best_features[i]].values
        X_test = [read_array_from_string(s) for s in X]
        Y_test = test_data['labels']
        clf = best_cls[i]

        pred_labels = []
        for t in X_test:
            y_pred_test = clf.predict(t)
            pred_labels.append(y_pred_test)

        most_frequent_label = max(set(pred_labels), key=pred_labels.count)
        most_predicted_label.append(predicted_labels[l])

    return most_predicted_label



def test_error_multilabel_cls (df_test, ml_cls, feature):

    test_data = df_test
    X = test_data[feature].values
    X_test = [read_array_from_string(s) for s in X]
    Y_test = lb_make.fit_transform(test_data["primary_microconstituent"])

    y_pred_test_ml = ml_cls.predict(X_test)
    error_test_ml = np.mean(y_pred_test_ml != Y_test)

    return error_test_ml


# read the trained data
df_train = pd.read_csv(file_path + "train_data.csv")

solution = calc_pairwise_classifier(df_train, models)
# the best feature in all cases except 1 is 'feature_512_b5'
multilabel_cls = get_multilabel_classifier(df_train, feature='feature_512_b5')

print("solution= ", solution)

# read the test_data
df_test = pd.read_csv(file_path + "test_data.csv")

#pairwise_test_error = test_error_pairwise_cls(df_test, best_features, best_cls)
#print('pairwise_test_error= ', pairwise_test_error)

error_test_ml = test_error_multilabel_cls(df_test, multilabel_cls, feature='feature_512_b5' )
print('error_test_ml= ', error_test_ml)


