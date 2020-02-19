import pandas as pd
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=FutureWarning)

# Importing the datasets
train = pd.read_csv("https://raw.githubusercontent.com/oo92/Titanic-Kaggle/master/train.csv")
test = pd.read_csv("https://raw.githubusercontent.com/oo92/Titanic-Kaggle/master/test.csv")

# The Function is responsible for selecting, engineering and returning the predictive features
def featureSelectionAndEngineering(df):
    df['Sex'] = df['Sex'].replace(['female', 'male'], [0, 1])
    
    df['Embarked'] = df['Embarked'].replace(['C', 'Q', 'S'], [1, 2, 3])
    # Integer encoding the Embarked column
    
    df['Age'].fillna(df.groupby('Sex')['Age'].transform("median"), inplace=True)
    df['HasCabin'] = df['Cabin'].notnull().astype(int)
    df['Relatives'] = df['SibSp'] + df['Parch']
    df['Fare'].fillna(df.groupby('Sex')['Age'].transform("median"), inplace=True)
    return train[['Pclass', 'Sex', 'Relatives', 'Fare', 'Age', 'Embarked', 'HasCabin']]

# Using the train/test split method to validate the accuracy of the model
x_train, x_validate, y_train, y_validate = train_test_split(featureSelectionAndEngineering(train), train['Survived'], test_size=0.22, random_state=0)

logReg = LogisticRegression()

# Training the model with the training data set and the Logistic Regression algorithm
logReg.fit(x_train, y_train)

# Assigning the accuracy of the model to the variable "accuracy"
accuracy = logReg.score(x_validate, y_validate)

# Predicting for the data in the test set
predictions = logReg.predict(featureSelectionAndEngineering(test))

# Writing the predictions to a new CSV file
submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': predictions})
filename = 'Titanic-Submission.csv'
submission.to_csv(filename, index=False)

print(accuracy*100, "%")

# The model yields an accuracy score of 83.76% with the given data set
