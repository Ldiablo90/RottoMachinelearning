import pandas as pd
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.DataFrame(np.random.randint(low=-3, high=3, size=(3,3)))
df.columns=['move_1','move_2','move_3']
print(df.head())

movememts = df.values

linalg.norm(movememts, ord=1,axis=1)

df = pd.DataFrame(np.random.randint(low=-1, high=10, size=(3,2)))
df.columns=['x','y']
print(df.head())

sns.lmplot(x='x',
           y='y',
           data=df,
           fit_reg=False,
           scatter_kws={"s":200})
plt.title('data point visualization')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

data_points = df.values

linalg.norm(data_points, ord=2, axis=1)
