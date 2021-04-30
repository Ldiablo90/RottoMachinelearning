import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.DataFrame(columns=['calory','breakfast','lunch','dinner','exercise','body_shape'])

df.loc[0] = [1200, 1, 0, 0, 2, 'Skinny']
df.loc[1] = [2800, 1, 1, 1, 1, 'Normal']
df.loc[2] = [3500, 2, 2, 1, 0, 'Fat']
df.loc[3] = [1400, 0, 1, 0, 3, 'Skinny']
df.loc[4] = [5000, 2, 2, 2, 0, 'Fat']
df.loc[5] = [1300, 0, 0, 1, 2, 'Skinny']
df.loc[6] = [3000, 1, 0, 1, 1, 'Normal']
df.loc[7] = [4000, 2, 2, 2, 0, 'Fat']
df.loc[8] = [2600, 0, 2, 0, 0, 'Normal']
df.loc[9] = [3000, 1, 2, 1, 1, 'Fat']

print('df is \n',df)

x = df[['calory','breakfast','lunch','dinner','exercise']]
print('x is \n', x)

y = df[['body_shape']]
print('y is \n', y)

x_std = StandardScaler().fit_transform(x)
print('x_std is \n', x_std)

features = x_std.T
covariance_matrix = np.cov(features)
print('covariance_matrix is \n',covariance_matrix)

eig_vals, eig_vecs = np.linalg.eig(covariance_matrix)
print('eig_vals is\n',eig_vals)

projected_x = x_std.dot(eig_vecs.T[0])

result = pd.DataFrame(projected_x, columns=['PC1'])
result['y-axis'] = 0.0
result['label'] = y

print('result is\n',result)

sns.lmplot(x='PC1',y='y-axis', data=result, fit_reg=False, scatter_kws={'s':50}, hue='label')
plt.title('PCA result')

plt.show()

# ********************************************************************************

from sklearn import decomposition

pca = decomposition.PCA(n_components=1)
sklearn_pca_x = pca.fit_transform(x_s
td)

sklearn_result = pd.DataFrame(sklearn_pca_x, columns=['PC1'])

sklearn_result['y-axis'] = 0.0
sklearn_result['label'] = y
sns.lmplot(x='PC1',y='y-axis', data=result, fit_reg=False, scatter_kws={'s':50}, hue='label')