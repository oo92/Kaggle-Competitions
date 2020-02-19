import pandas as pd

train_one = pd.read_csv("https://raw.githubusercontent.com/oo92/blue-book-bulldozers-kaggle/master/TrainOne.csv", parse_dates=['saledate'])
train_two = pd.read_csv("https://raw.githubusercontent.com/oo92/blue-book-bulldozers-kaggle/master/TrainTwo.csv")
test = pd.read_csv("https://raw.githubusercontent.com/oo92/blue-book-bulldozers-kaggle/master/Test.csv", parse_dates=['saledate'])
validate = pd.read_csv("https://raw.githubusercontent.com/oo92/blue-book-bulldozers-kaggle/master/Valid.csv", parse_dates=['saledate'])

train = pd.concat([train_one, train_two], axis=1)

train_independent = train.drop('SalePrice', axis=1)
train_dependent = train['SalePrice']

print(train_dependent)