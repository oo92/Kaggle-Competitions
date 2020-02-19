import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

train = pd.read_csv("https://raw.githubusercontent.com/oo92/Boston-Kaggle/master/train.csv")
test = pd.read_csv("https://raw.githubusercontent.com/oo92/Boston-Kaggle/master/test.csv")

# Iterates through the columns and fixes any NaNs
def refilling_nan_values(dataFrame):
    for col in dataFrame.columns:
        # If there are any NaN values in this column
        if pd.isna(dataFrame[col]).any():
            # Replace NaN in object columns with 'N/A'
            if dataFrame[col].dtypes == 'object':
                dataFrame[col].fillna('XX', inplace=True)
            # Replace NaN in float columns with 0
            elif dataFrame[col].dtypes == 'float64':
                dataFrame[col].fillna(dataFrame[col].mean(), inplace=True)
    return dataFrame


def train_OHE(dataFrame, cat_cols):
    categorical_df = dataFrame[cat_cols]
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    return encoder.fit(categorical_df)

# Getting every categorical column in the dataframe
def get_cat_cols(dataFrame):
    cat_cols = []
    for col in dataFrame.columns:
        if not (dataFrame[col].dtypes == 'float64' or dataFrame[col].dtypes == 'int64'):
            cat_cols.append(col)

    return cat_cols

def OHE(dataFrame, cat_cols, encoder):
    # Get dataframe with only categorical columns
    categorical_dataFrame = dataFrame[cat_cols]
    # Get one hot encoding
    OHE_dataFrame = pd.DataFrame(encoder.transform(categorical_dataFrame), columns = encoder.get_feature_names())
    # Get float columns
    float_dataFrame = dataFrame.drop(cat_cols, axis=1)
    # Return the combined array
    return pd.concat([float_dataFrame, OHE_dataFrame], axis = 1)

# Transforms a dataset to be ready for input into a model
def feature_engineering(dataFrame, encoder = None):
    dataFrame = refilling_nan_values(dataFrame)
    categorical_columns = get_cat_cols(dataFrame)
    # If there is no encoder, train one
    if encoder == None:
        encoder = train_OHE(dataFrame, categorical_columns)
    # Encode Data
    dataFrame = OHE(dataFrame, categorical_columns, encoder)
    # Return the encoded data AND encoder
    return dataFrame, encoder

labels = train['SalePrice']
train_id = train['Id']
test_id = test['Id']
train.drop('SalePrice', axis = 1, inplace = True)
train.drop('Id', axis = 1, inplace = True)
test.drop('Id', axis = 1, inplace = True)

# Dividing the training dataset into train/test sets with the test size being 22% of the overall dataset.
x_train, x_test, y_train, y_test = train_test_split(train, labels, test_size = 0.22, random_state = 42)

# Reset indices after splitting
x_train.reset_index(drop = True, inplace = True)
x_test.reset_index(drop = True, inplace = True)
y_train.reset_index(drop = True, inplace = True)
y_test.reset_index(drop = True, inplace = True)

# Encode train and save encoder
x_train, encoder = feature_engineering(x_train)

# Invoking the Random Forest Classifier with a 1.25x the mean threshold to select correlating features
sel = SelectFromModel(RandomForestClassifier(n_estimators = 100), threshold = '1.25*mean')
sel.fit(x_train, y_train)
selected = x_train.columns[sel.get_support()]

# Defining the Gradient Boosting Regressor algorithm
gradientBoostingRegressor = GradientBoostingRegressor(n_estimators=100, max_depth=4)

# Transform the split test set using the same encoder
x_test, _ = feature_engineering(x_test, encoder)

train, encoder = feature_engineering(train)
test, _ = feature_engineering(test, encoder)

# Fitting the columns into the Gradient Boosting Regressor
gradientBoostingRegressor.fit(train[selected], labels)

# Predicting the outcomes
predictions = gradientBoostingRegressor.predict(test[selected])

# Writing the predictions to a new CSV file
submission = pd.DataFrame({'Id': test_id, 'SalePrice': predictions})
filename = 'Boston-Submission.csv'
submission.to_csv(filename, index=False)