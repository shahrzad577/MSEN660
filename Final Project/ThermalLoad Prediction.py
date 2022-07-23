import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import seaborn as sns
from scipy.stats import spearmanr
import timeit


## This file is exactly the same as 04_test However for all data removing shape column completey

base_path = "H:\\tamu\\courses\\paper\\KES_SDM\\"
csv_path = base_path + 'ExcelData\\DataFiltered.csv'

df = pd.read_csv(csv_path)


# Second condition: considering one model for cubic form and one model for L,T, and U
df_Cube = df[df['Shape'] == 'Cubic']
df_OtherShapes = df[df['Shape'] != 'Cubic']

def get_train_test(df):
    X = pd.concat([df.iloc[:, 0:5], df.iloc[:, 6:10]], axis=1, join='inner')
    y = df["ThermalLoads"]
    np.random.seed(10)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7)

    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = get_train_test(df_OtherShapes)  # df   / df_Cube   /  df_OtherShapes


# since I want to use Shape column as hue for plot I need to split all data based on randomization and then remove
# shape column for X_train, X_test, y_train and y_test.
shape_series = X_test['Shape']
# series need to be convert to array to make the indices start from zero otherwise in the result of concat is wrong!
shape_arr = np.array(shape_series)
shape_df = pd.DataFrame(shape_arr, columns=['Shape'])

# Removing the shape column
X_train = X_train.iloc[:, 1:]
X_test = X_test.iloc[:, 1:]


# Regression models:
def get_LR_model(X_train, y_train):
    lr = LinearRegression(fit_intercept=True, normalize=False, copy_X=True)
    lr.fit(X_train, y_train)
    rsquared = lr.score(X_train, y_train)

    return lr, rsquared

lr , rsquared = get_LR_model(X_train, y_train)

print("lr model, R squared of train data:", rsquared)

y_pred_lr = lr.predict(X_test)
test_rSquared = lr.score(X_test, y_test)
print('lr2 model  R squared of test data:', test_rSquared)

# Getting the correlation of y_pred and y_test. +_1 means very correlated, 0 means not correlated at all
corrLR, _ = spearmanr(y_pred_lr, y_test)
print('correlation between y_pred_linearRegression and y_test: ', corrLR)

print('coefficient of lr model: ', lr.coef_)
print('intercept of lr model', lr.intercept_)


# Polynomial regression
def get_PolynomialLinearModel(X_train,y_train, degree):
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X_train)
    poly.fit(X_poly, y_train)
    lrPoly = LinearRegression()
    lrPoly.fit(X_poly, y_train)
    poly_rsquared_train = lrPoly.score(X_poly, y_train)

    return lrPoly, poly, poly_rsquared_train


# monitoring time
start = timeit.default_timer()
lrPoly, poly, poly_rsquared_train = get_PolynomialLinearModel(X_train, y_train, degree=4)
stop = timeit.default_timer()
print('Time: ', stop - start)


Xtest_poly = poly.fit_transform(X_test)
poly_rsquared_test = lrPoly.score(Xtest_poly, y_test)


print("Poly model, R squared of train data:", poly_rsquared_train)
print("Poly model, R squared of test data:", poly_rsquared_test)


y_pred_poly = lrPoly.predict(Xtest_poly)
# Getting the correlation of y_pred and y_test. +_1 means very correlated, 0 means not correlated at all
corr, _ = spearmanr(y_pred_poly, y_test)
print('correlation between y_pred_poly and y_test: ', corr)

print('coefficient of lrPoly model: ', lrPoly.coef_)
print('intercept of lrPoly model', lrPoly.intercept_)


y_pred_poly_df = pd.DataFrame(y_pred_poly, columns=['y_pred_poly'])
y_pred_lr_df = pd.DataFrame(y_pred_lr, columns=['y_pred_LR'])


y_test_arr = np.array(y_test)
y_test_df = pd.DataFrame(y_test_arr, columns=['y_test'])

# concat y_test, y_pred and shape as one DataFrame to use for plot
data_PolyPred = pd.concat([y_pred_poly_df, y_test_df,shape_df], axis=1, join='inner')
data_LRPred = pd.concat([y_pred_lr_df, y_test_df, shape_df], axis=1, join='inner')

shapes = ["UShape", "LShape", "Cubic", "TShape"]
# shapes2 = ["UShape", "LShape", "Cubic", "TShape"]

# sorting the data frame based on original Shapes list so the color matches the other plots
def sortDataFramebyShape(data, shapes):
    data.Shape = pd.Categorical(data.iloc[:,2],
                          categories=shapes,
                          ordered=True)

# get sorted data based on shape
sortDataFramebyShape(data_PolyPred, shapes)
sortDataFramebyShape(data_LRPred, shapes)



csfont = {'fontname':'calibri'}   #calibri   Times New Roman
labelfont = 12
titlefont = 16
ticksize = 9

def get_scatterbyHue(data, title, plotname):
    groups = data.groupby('Shape')
    for name, group in groups:
        plt.plot(group.iloc[:,0], group.iloc[:,1], marker='o',
                 mfc='none', linestyle='', markersize=6, label=name)  # mfc='none' unfill marker
    plt.gcf().subplots_adjust(top=0.8, bottom=0.2, left=0.2, right=0.8)
    #plt.legend(prop={'size': 10})
    plt.xlabel('predicted energy (kWh/m2)', fontsize=labelfont, **csfont)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.ylabel('simulated energy (kWh/m2)', fontsize=labelfont, **csfont)
    plt.title(title, fontsize=titlefont, **csfont)
    plt.grid(linestyle='dotted')
    plt.savefig(base_path + 'Plots\\' + plotname + '.jpg')
    plt.show()



####### for all data
get_scatterbyHue(data_PolyPred, 'Predicted vs. simulated values \n Polynomial model, degree=4',
                  'all_data degree4')
get_scatterbyHue(data_LRPred, 'Predicted vs. simulated values \n Regression model, degree=1',
                'all_data degree1')

###### for df_Cube
get_scatterbyHue(data_PolyPred, 'Predicted vs. simulated values \n Polynomial model, degree=4',
                'square degree4')
get_scatterbyHue(data_LRPred, 'Predicted vs. simulated values \n Regression model, degree=1',
                'square degree1')

##### for df_OtherShapes
get_scatterbyHue(data_PolyPred, 'Predicted vs. simulated values \n Polynomial model, degree=4',
                 'TUL_shape degree4')
get_scatterbyHue(data_LRPred, 'Predicted vs. simulated values \n Regression model, degree=1',
                 'TUL_shape degree1')


### Plot for cubic model or U.L,T model
def get_scatterPlotGray(y_pred, y_test, plotname):
     plt.figure(figsize=(8,8))
     plt.scatter(x=y_pred, y=y_test, c='white', edgecolors='black')
     plt.xlabel('predicted values')
     plt.ylabel('simulated values')
     plt.title(plotname)
     plt.grid(linestyle='dotted')
     plt.savefig(base_path + 'Plots\\' + plotname + '.jpg')

get_scatterPlotGray(y_pred_poly, y_test, 'polynomial model_all data')
get_scatterPlotGray(y_pred_lr, y_test, 'Regression model_all data')
