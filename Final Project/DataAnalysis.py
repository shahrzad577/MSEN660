import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler as ssc

# Feature selection using PCA and Exhausive search
# Note that PCA calculation does not consider response variable (thermalLoad)

base_path = "H:\\tamu\\courses\\paper\\KES_SDM\\"
csv_path = base_path + 'ExcelData\\DataFiltered.csv'
screePlot_path = base_path + 'ScreePlots\\'
LoadingMatrix_path = base_path + 'LoadingMatrix\\'

df = pd.read_csv(csv_path)


# for PCA we consider the data of only one geometry type at a time to make sure if different features
# have different impact considering the geometry. For example does rotation weight is different considering
# different geometry?


def get_XandFeaturename(df,shape,bool):

    df = df[df['Shape'] == shape]
    ## Considering wall material
    if bool:
        features_name = df.columns[1:5].append(df.columns[6:10])
        X = pd.concat([df.iloc[:, 1:5],df.iloc[:, 6:10]], axis=1, join='inner')
        components = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8']

    ## Not considering wall material
    else:
        features_name = df.columns[1:5]
        X = df.iloc[:, 1:5]
        components = ['PC1', 'PC2', 'PC3', 'PC4']

    return features_name, X, components


#........................ Feature selection using PCA .......................
def compute_PCA(X):
    X_norm = ssc().fit_transform(X)
    pca = PCA().fit(X_norm)
    variance = pca.explained_variance_ratio_
    cumsum = np.cumsum(variance)

    return pca, variance, cumsum


csfont = {'fontname':'Times New Roman'}
labelfont = 16
titlefont = 19
ticksize = 8

# Scree plot
def get_scree_plot(data, componnets ,ltype, xlabel, ylabel, figname, shape):
    plt.figure(figsize=(8, 5))
    x = np.arange(start=0, stop=len(componnets), step=1)
    plt.xticks(x, componnets)
    plt.plot(100 * data, ltype , linewidth=2)  # 'ro-'
    plt.xlabel(xlabel, fontsize=labelfont, **csfont) #'PC'
    plt.ylabel(ylabel+shape, fontsize=labelfont, **csfont)    #'Percentage of Variance'
    plt.title('scree plot', fontsize=titlefont, **csfont)
    plt.grid(linestyle='dotted')
    figpath = screePlot_path + shape + figname + '.png'    # 'Variance'
    plt.savefig(figpath, dpi=150)


# loading matrix
def get_Loading_Matrix(pca, cols, index, shape, str):
    loading = pca.components_.T *np.sqrt (pca.explained_variance_)
    loading_matrix = pd.DataFrame(loading, columns=cols, index=index)  #columns=components, index=features_name
    loading_matrix.to_csv(LoadingMatrix_path + shape +'_loading_matrix_' + str + '.csv')


# features_name, X, components = get_XandFeaturename(df, shape='Cubic', bool=True)



'''
shapes = ["UShape", "LShape", "Cubic", "TShape"]

# In the first ForLoop the bool is True meaning we are Considering wall material in PCA calculation and plots
for shape in shapes:
    features_name, X, components = get_XandFeaturename(df, shape=shape, bool=True)
    pca, variance, cumsum = compute_PCA(X)
    get_Loading_Matrix(pca, cols=components, index=features_name, shape=shape, str='all')
    # plots
    get_scree_plot(variance, components, 'ro-', 'PC', '_Percentage of Variance_all_', 'Variance_all', shape)
    get_scree_plot(cumsum, components, 'go--', 'PC', '_Cumulative Percentage of Variance_all_', 'Cumsum_all', shape)

# In the second ForLoop the bool is False meaning we are NOT Considering wall material in PCA calculation and plots
for shape in shapes:
    features_name, X, components = get_XandFeaturename(df, shape=shape, bool=False)
    pca, variance, cumsum = compute_PCA(X)
    get_Loading_Matrix(pca, cols=components, index=features_name, shape=shape, str= 'NoWallMtrl')
    # plots
    get_scree_plot(variance, components, 'ro-', 'PC', '_Percentage of Variance_NoWallMtrl', 'Variance_NoWallMtrl', shape)
    get_scree_plot(cumsum, components, 'go--', 'PC', '_Cumulative Percentage of Variance_NoWallMtrl', 'Cumsum_NoWallMtrl', shape)

'''

df = pd.read_csv(csv_path)
## Splitting data to 2 modes (samples with mt1 and samples with mt2 or mt3)
df_mt1 = df[df['Wall_Mtrl'] == 'str_TIP_Insulation']
df_mt2_3 = df[df['Wall_Mtrl'] != 'str_TIP_Insulation']

df = df_mt2_3

X = pd.concat([df.iloc[:, 1:5],df.iloc[:, 6:10]], axis=1, join='inner')
features_name = df.columns[1:5].append(df.columns[6:10])
components = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8']

pca, variance, cumsum = compute_PCA(X)
get_Loading_Matrix(pca, cols=components, index=features_name, shape='allshape_mt2_3', str='all')
# plots
# get_scree_plot(variance, components, 'ro-', 'PC', '_Percentage of Variance_mt2_mt3', 'Variance_mt2_mt3', 'allForms')
# get_scree_plot(cumsum, components, 'go--', 'PC', '_Cumulative Percentage of Variance_mt2_mt3', 'Cumsum_mt2_mt3', 'allForms')