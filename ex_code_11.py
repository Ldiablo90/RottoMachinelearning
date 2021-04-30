import pandas as pd
from IPython.display import Image
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv('./input/train.csv')
test = pd.read_csv('./input/test.csv')

train_test_data = [train, test]

def bar_chart(feature):
    survived = train[train['Survived']==1][feature].value_counts()
    dead = train[train['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived,dead])
    df.index = ['Survived','Dead']
    df.plot(kind='bar',stacked=True, figsize=(10,5))

for dataset in train_test_data:
    dataset['Title'] = dataset['Name'].str.extract('([A-Za-z]+)\.', expand=False)

train['Title'].value_counts()
test['Title'].value_counts()

title_mapping = {"Mr":0,
                 "Miss":1,
                 "Mrs":2,
                 "Master":3,
                 "Dr":3,
                 "Rev":3,
                 "Col":3,
                 "Major":3,
                 "Mlle":3,
                 "Countess":3,
                 "Ms":3,
                 "Lady":3,
                 "Jonkheer":3,
                 "Don":3,
                 "Dona":3,
                 "Mme":3,
                 "Capt":3,
                 "Sir":3,
            }
for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].map(title_mapping)
print(train.head())
bar_chart('Title')
train.drop("Name", axis=1, inplace=True)
test.drop("Name", axis=1, inplace=True)

sex_mapping = {"male":0, "female":1}
for dataset in train_test_data:
    dataset["Sex"] = dataset['Sex'].map(sex_mapping)
print(train.head())
bar_chart('Sex')

train['Age'].fillna(train.groupby('Title')['Age'].transform('median'),inplace=True)
test['Age'].fillna(train.groupby('Title')['Age'].transform('median'),inplace=True)

facet = sns.FacetGrid(train, hue="Survived",aspect=4)
facet.map(sns.kdeplot, 'Age',shade=True)
facet.set(xlim=(0, train['Age'].max()))
facet.add_legend()
plt.xlim(0,20)

age_mapping = {
    "child":0,
    "young":1,
    "adult":2,
    "mid-age":3,
    "senior":4
}
print(train_test_data[0].head(100))
"""
for dataset in train_test_data:
    dataset.loc[dataset['Age'] <= 16, 'Age'] = 0,
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <=26),'Age'] = 1,
    dataset.loc[(dataset['Age'] > 26) & (dataset['Age'] <=36),'Age'] = 2,
    dataset.loc[(dataset['Age'] > 36) & (dataset['Age'] <=62),'Age'] = 3,
    dataset.loc[dataset['Age'] > 62,'Age'] = 4

bar_chart('Age')

Pclass1 = train[train['Pclass'] == 1]['Embarked'].value_counts()
Pclass2 = train[train['Pclass'] == 2]['Embarked'].value_counts()
Pclass3 = train[train['Pclass'] == 3]['Embarked'].value_counts()

df = pd.DataFrame([Pclass1,Pclass2,Pclass3])
df.index = ['1st class','2an class','3rd class']
df.plot(kind='bar', stacked=True, figsize=(10,5))
 
plt.show()
"""