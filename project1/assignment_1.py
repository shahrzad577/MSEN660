"Assignmnet 1: Classification using filter feature selection"

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import scipy
import matplotlib.pyplot as plt

project_path = "H:\\tamu\\courses\\MSEN660\\Homeworks\\ProblemSet_2\\project_2\\"
file_path = project_path + 'Stacking_Fault_Energy_Dataset.txt'
original_data  = pd.read_table(file_path)
# read_table is read_csv with sep=',' replaced by sep='\t'

# pre-process the data
num_all_samples = original_data.shape[0]                                       # original number of training points
features = original_data.columns[:-1]                                    # original features
feature_sum_percentage = np.sum(original_data.iloc[:, :-1] > 0) / num_all_samples   # fraction of nonzero components for each feature
features_to_drop = features[feature_sum_percentage<0.6]            # features with less than 60% nonzero components
data_1 = original_data.drop(features_to_drop, axis=1)                # drop those features
sample_min = data_1.min(axis=1)                                  # finding samples with zero values
data_2 = data_1[sample_min != 0]                                # drop sample points with any zero values
SFE_data = data_2[(data_2.SFE < 35) | (data_2.SFE > 45)]          # get a subset of dataframe with condition


Y = SFE_data.SFE > 40    # if SFE > 40, Y =true , else Y=false
# so Y == True and ~Y==False which represent the label

plt.style.use('seaborn')
fig,ax = plt.subplots(1,3,figsize=(16,4),dpi=150)
fig.subplots_adjust(wspace=0.3)
plt.sca(ax[0])
for idx, element in enumerate(['Cr', 'Fe', 'Ni']):
    plt.sca(ax[idx])
    plt.title(element, fontsize=16)
    SFE_data[element][Y].hist(bins=8, rwidth=0.9, color='red', density=True, alpha=0.2)
    SFE_data[element][~Y].hist(bins=8, rwidth=0.9, color='blue', density=True, alpha=0.2)
    SFE_data[element][Y].plot(kind='kde', style='r-')
    SFE_data[element][~Y].plot(kind='kde', style='b-')
    bottom,top = ax[idx].get_ylim()
    ax[idx].yaxis.set_ticks(np.arange(bottom,top,0.05))
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('percent weight',fontsize=14)
    plt.ylabel('histogram/density',fontsize=14)    
plt.savefig(project_path + 'c01_matex-a.png', bbox_inches='tight')
#plt.show()


# X[Y,0] returns values of Ni where SFE>40
# X[Y,1] returns values of Fe where SFE>40

# Plot results
def Plot_Classifier(X, Y, var1, var2, a, b, str):
    plt.style.use('seaborn')
    fig = plt.figure(figsize=(4,4),dpi=150)   # (8,8)
    plt.axis('equal')
    plt.scatter(X[var1][~Y],X[var2][~Y],c='blue',s=32,label='Low SFE') # Low SFE daya (Y==False or ~Y)
    plt.scatter(X[var1][Y],X[var2][Y],c='orange',s=32,label='High SFE')
    left,right = plt.xlim()
    bottom,top = plt.ylim()
    # drawing the classifier line based on aTx + b =0  (T means Transpose)
    plt.plot([left,right],[-left*a[0]/a[1]-b/a[1],-right*a[0]/a[1]-b/a[1]],'k',linewidth=2)
    plt.xlim(left,right)
    plt.ylim(bottom,top)
    plt.xticks(fontsize=10) # 14
    plt.yticks(fontsize=10)
    plt.xlabel(var1, fontsize=14)  # 18
    plt.ylabel(var2, fontsize=14)
    plt.legend(fontsize=10,loc="lower left",markerfirst=False,markerscale=1.5,handletextpad=0.1)
    plt.savefig(project_path + var1 +'_' + var2 + '_' + str, bbox_inches='tight')
    plt.show()

# Select variables for LDA
var1 = 'Ni'
var2 = 'Fe'

# Feature data
X = SFE_data[[var1,var2]]

num_samples = int(SFE_data.shape[0])
class1_prior = np.sum(Y)/num_samples        # the prior probablity of class_1
class0_prior = np.sum(~Y)/num_samples    # the prior probablity of class_0

# Apply LDA and get line coefficients
classifier = LDA(priors=(class1_prior, class0_prior))     #  0.51, 0.49
classifier.fit(X, Y.astype(int))    # Y is true/false --> Y.astype will be 1/0

#retreiving a and b value in the classifier formula
a = classifier.coef_[0]
b = classifier.intercept_[0]

Plot_Classifier (X, Y, var1, var2, a, b, 'train')

# Assignmnet 1- a:  the first 20% samples for training and rest for testing

split = int(0.2 * SFE_data.shape[0])
train_data , test_data = SFE_data[:split], SFE_data[split:]

X_train = train_data.iloc[:,:-1]
Y_train = train_data.SFE>40   # High SFE will be labled as true (1) and low SFE will be labeled as false (0)
# High SFE = class 1 / low SFE = class 0

X_test = test_data.iloc[:, :-1]
Y_test = test_data.SFE>40

# X_train[Y_train] returns X_train where SFE>40 is true
# X_train[~Y_train] returns X_train where SFE>40 is false


# Assignmnet 1- b:

ttest = scipy.stats.ttest_ind(X_train[Y_train], X_train[~Y_train])
ttest_Value = [abs(ele) for ele in ttest]   #note that P_value always >0

cols = {'t_statistics': ttest_Value[0], 'p_value': ttest_Value[1]}

df = pd.DataFrame(cols, columns=['t_statistics','p_value'],
                  index=['C', 'N',  'Mn', 'Si', 'Cr', 'Ni' , 'Fe'])

df_org = pd.DataFrame(cols, columns=['t_statistics','p_value'],
                  index=['C', 'N',  'Mn', 'Si', 'Cr', 'Ni', 'Fe'])   # is used for assignmnet2 ( not sorted)

df.sort_values(by=['t_statistics'], inplace=True, ascending=False)
#print(df)

# Assignmnet 1- c:

var1 = 'Cr'
var2 = 'Mn'

Xtrain_var1_var2 = train_data [[var1,var2]]
classifier_2 = LDA()
classifier_2.fit(Xtrain_var1_var2, Y_train.astype(int))

a_2 = classifier_2.coef_[0]
b_2 = classifier_2.intercept_[0]

# plot classifier
Plot_Classifier (Xtrain_var1_var2, Y_train, var1, var2, a_2, b_2, 'train')

#  Plot the testing data with the sumperimposed previously-obtained LDA decision boundary
Xtest_var1_var2 = X_test [[var1,var2]]
#Plot_Classifier (Xtest_var1_var2, Y_test, var1, var2, a_2, b_2, 'test')

# Estimate the classification error using the training and test data

y_pred_train = classifier_2.predict(Xtrain_var1_var2)
error_train = np.mean(y_pred_train != Y_train.astype(int))

y_pred_test = classifier_2.predict(Xtest_var1_var2)
y_test = Y_test.astype(int).values

error_test = np.mean(y_pred_test != y_test)

# 4) Repeat for the top three, four, and five predictors
# generallize the method to a function that compute classifier and classification error

def get_classifier(train_data, Y_train, feature):
    xtrain = train_data[feature]
    ytrain = Y_train.astype(int)
    classifier = LDA()
    classifier.fit(xtrain, ytrain)
    return classifier

def get_classification_error(train_data, Y_train, X_test, Y_test):
    idx = 2
    test_error_list = []
    train_error_list = []
    columns = df.index

    while (idx <= 5):  # idx <= len(columns) if you desire to reach top 7 (all features)
        feature = columns[:idx]

        classifier = get_classifier(train_data, Y_train, feature)

        xtrain = train_data[feature]
        y_pred_train = classifier.predict(xtrain)
        ytrain = Y_train.astype(int)
        error_train = np.mean(y_pred_train != ytrain)
        train_error_list.append(error_train)

        xtest = X_test[feature]
        y_pred_test = classifier.predict(xtest)
        ytest = Y_test.astype(int)
        error_test = np.mean(y_pred_test != ytest)
        test_error_list.append(error_test)
        idx += 1

    return train_error_list, test_error_list

train_error_list, test_error_list = get_classification_error (train_data, Y_train, X_test, Y_test)

print('train_error_list= ', np.round(train_error_list, 4))
print('test_error_list= ', np.round(test_error_list, 4))






