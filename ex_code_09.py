import pandas as pd
from IPython.display import Image
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
Image("images/titanic-disaster.jpg")

train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/test.csv')

train.head()
test.head()

train.shape
test.shape

train.info()
test.info()

train.isnull().sum()
test.isnull().sum()

def bar_chart(feature):
    survived = train[train['Survived']==1][feature].value_counts()
    dead = train[train['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived,dead])
    df.index = ['Survived','Dead']
    df.plot(kind='bar',stacked=True, figsize=(10,5))

bar_chart('Sex')
bar_chart('Pclass')
bar_chart('SibSp')
bar_chart('Parch')
bar_chart('Embarked')

plt.show()
