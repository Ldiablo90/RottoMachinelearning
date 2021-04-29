import pandas as pd
from IPython.display import Image
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/test.csv')

train_test_data = [train,test]

for dataset in train_test_data:
    dataset['Title'] = dataset['Name'].str.extract('([A-Za-z]+)\.',expand=False)


print(train['Title'].value_counts())
print(test['Title'].value_counts())

title_mapping = {
