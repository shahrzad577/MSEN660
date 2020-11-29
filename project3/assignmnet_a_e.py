import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler as ssc
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage


file_path = 'H:\\tamu\\courses\\MSEN660\\Homeworks\\computer_project3\\'
SMA_org = pd.read_csv(file_path + 'Soft_Magnetic_Alloy_Dataset.csv')

# data pre processing
features_org = SMA_org.columns[0:26]
feature_val_org = SMA_org.values[:, 0:26]
response_org = SMA_org['Coercivity (A/m)']   # select response

num_samples = feature_val_org.shape[0]
feature_value_sum = np.sum(feature_val_org > 0, axis=0) / num_samples
is_selected = feature_value_sum > 0.05      # true/false  --> condition on column

# filtering data
feature_val_filt = feature_val_org[:, is_selected]      # get features with more than 5% feature val sum
no_NAN = np.invert(np.isnan(response_org))  # true/false --> all rows with real value of response --> condition on row
feature_val = feature_val_filt[no_NAN, :]
response = response_org[no_NAN]
features_name = features_org[is_selected]

# add random perturbation to the features: adding zero mean Gaussian noise to feature value
np.random.seed(0)
std = 2
mean = 0
n, d = feature_val.shape
noise = np.random.normal(mean, std, [n, d])
feature_val_noisy = feature_val + noise
feature_val_noisy = (feature_val_noisy + abs(feature_val_noisy)) / 2    # clamp values at zero (?????)

# normalize data
feature_val_normalized = ssc().fit_transform(feature_val_noisy)

# compute PCA
pca = PCA().fit(feature_val_normalized)
variance = pca.explained_variance_ratio_
cumsum = np.cumsum(variance)

components = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10', 'PC11', 'PC12']

# scree plot
def get_scree_plot(data, componnets ,ltype, xlabel, ylabel, figname):
    plt.figure(figsize=(8, 5))
    x = np.arange(start=0, stop=len(componnets), step=1)
    plt.xticks(x, componnets)
    plt.plot(100 * data, ltype , linewidth=2)  # 'ro-'
    plt.xlabel(xlabel) #'PC'
    plt.ylabel(ylabel)    #'Percentage of Variance'
    plt.title('scree plot')
    plt.grid(linestyle='dotted')
    figpath = file_path + figname + '.png'    # 'Variance'
    plt.savefig(figpath, dpi=150)


get_scree_plot(variance, components, 'ro-', 'PC', 'Percentage of Variance', 'Variance')

#get_scree_plot(cumsum, componnets,'go--', 'PC', 'Cumulative Percentage of Variance', 'Cumsum')


## plot cumsum
plt.figure(figsize=(8, 5))
x = np.arange(start=0, stop=len(components), step=1)     #for the line 2 point weere enough though..
y = [95 for i in np.zeros((12,))]
plt.xticks(x, components)
plt.plot(100 * cumsum, 'go--', linewidth=2, label='CumSum')
plt.plot(x,y,color='red',label='95% variance')
plt.xlabel('PC') #'PC'
plt.ylabel('Cumulative Percentage of Variance')    #'Percentage of Variance'
plt.title('scree plot')
plt.grid(linestyle='dotted')
plt.legend(framealpha=1, frameon=True)
figpath = file_path + 'Cumulative Percentage of Variance' + '.png'    # 'Variance'
plt.savefig(figpath, dpi=150)

# part_d : PCA plots
x_pca = PCA().fit_transform(feature_val_normalized)    # data transfomed by PCA()

low_ind = response <= 2
medium_ind = (response > 2) & (response < 8)
high_ind = response >= 8

def get_scatter_plot(X_pca, components, xlabel, ylabel):
    FirstPC = components.index(xlabel)
    SecondPC = components.index(ylabel)

    plt.figure()
    plt.scatter(X_pca[high_ind, FirstPC], X_pca[high_ind, SecondPC], alpha=0.8, c='red', edgecolors='none', s=16, label='high')
    plt.scatter(X_pca[medium_ind, FirstPC], X_pca[medium_ind, SecondPC], alpha=0.8, c='green', edgecolors='none', s=16,label='medium')
    plt.scatter(X_pca[low_ind, FirstPC], X_pca[low_ind, SecondPC], alpha=0.8, c='blue', edgecolors='none', s=16, label='low')
    plt.legend(fontsize=14, facecolor='white', markerscale=2, markerfirst=False, handletextpad=0)
    plt.grid(linestyle='dotted')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title('Principal Components')
    plt.xticks(size='medium')
    plt.yticks(size='medium')
    figname = xlabel + '_vs_' + ylabel
    plt.savefig(file_path + figname + '.png',dpi=150)


components = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10', 'PC11', 'PC12']
get_scatter_plot(x_pca, components, 'PC1', 'PC2')
get_scatter_plot(x_pca, components, 'PC1', 'PC3')
get_scatter_plot(x_pca, components, 'PC2', 'PC3')


# part_e: loading matrix
#note that pc.components --> principal axes in feature space representing the direction of maximum variance in data sorted by explained_variance
#The first principal component PC1 represents the component that retains the maximum variance of the data
# w1 corresponds to an eigenvector of the covariance matrix
# The elements of the eigenvector (w1j) are also known as loadings
# PCA loadings are the coefficients of the linear combination of the original variables from which the principal components (PCs) are constructed

loading = pca.components_.T *np.sqrt (pca.explained_variance_)
loading_matrix = pd.DataFrame (loading, columns=components, index=features_name)
# loading_matrix.to_csv(file_path + 'loading_matrix.csv')

get_scatter_plot (feature_val_normalized, features_name.tolist(), 'Fe', 'Si')


print(loading_matrix)

# Dendogram plot:
def get_dendogram_plot(fetaure_vector,ft, n):
    arr = np.zeros((ft, n))
    for i in range(ft):
        row = i * ft
        arr[i, 0] = fetaure_vector[row, 0]
        arr[i, 1] = fetaure_vector[row, 11]

    complete = linkage(arr, 'complete')   # 'single' or 'average' for other linkages

    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    dendrogram(
        complete,
        leaf_rotation=90.,  # rotates the x axis labels
        leaf_font_size=8.,  # font size for the x axis labels
    )
    plt.show()

#get_dendogram_plot (feature_val_normalized, 12,2)  # ft=total number of components , 2 is for 2 branch


