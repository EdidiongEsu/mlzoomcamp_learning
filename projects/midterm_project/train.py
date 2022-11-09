# Predicting Job change of data scientists

# Loading the dataset and Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier

import bentoml

# Load dataset
df = pd.read_csv("Data/churn.csv")

# Drop unwanted columns
df = df.drop(columns=["Unnamed: 0", "referral_id",
             "security_no", "joining_date"])

# Categorical and Numerical Columns
categorical_columns = list(df.dtypes[df.dtypes == "object"].index)
numerical_columns = [
    col for col in df.columns if col not in categorical_columns and col != 'churn_risk_score']

# cleaning up spaces in categorical varibles
for col in categorical_columns:
    df[col] = df[col].str.lower().str.replace(" ", "_")

# Cleaning up `Unknown` datapoints
# joined_through_referral and medium_of_operation have ? in their values. Change to null values
for col in ['medium_of_operation', 'joined_through_referral']:
    df[col] = df[col].replace('?', np.NaN)

# <b>`Gender`<b>. Unknown will bereplaced with Null values
df["gender"] = df["gender"].replace("unknown", np.NaN)

# <b>`days_since_last_login`<b>. -999 looks weird so that will be changed to a null value
df["days_since_last_login"] = df["days_since_last_login"].replace(-999, np.NaN)

# <b>`Average time spent` and `Points in Wallet`<b>
# There are values lower than 0 in the both columns which does not make sense based on data context
for col in ['avg_time_spent', 'points_in_wallet']:
    df[col] = df[col].apply(lambda x: x if x >= 0 else np.nan)

# <b>`Average Frequency Login days`<b>
df['avg_frequency_login_days'] = df['avg_frequency_login_days'].apply(
    lambda x: x if x != 'error' else -1)
df['avg_frequency_login_days'] = df['avg_frequency_login_days'].astype('float')
df['avg_frequency_login_days'] = df['avg_frequency_login_days'].apply(
    lambda x: x if x >= 0 else np.nan)

df["avg_frequency_login_days"] = pd.to_numeric(
    df["avg_frequency_login_days"], errors='coerce')

# updating categorical values
categorical_columns = list(df.dtypes[df.dtypes == "object"].index)
numerical_columns = [
    col for col in df.columns if col not in categorical_columns and col != 'churn_risk_score']

for feature in numerical_columns:
    df[feature] = pd.to_numeric(df[feature], errors='coerce')


def train_function():
    # ## Separate dataset into train and test

    df_full_train, df_test = train_test_split(
        df, test_size=0.2, random_state=1)
    df_train, df_val = train_test_split(
        df_full_train, test_size=0.25, random_state=1)

    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    y_train = df_train.churn_risk_score
    y_val = df_val.churn_risk_score
    y_test = df_test.churn_risk_score

    del df_train["churn_risk_score"]
    del df_val["churn_risk_score"]
    del df_test["churn_risk_score"]

    # Feature Engineering
    # Replacing Null Values

    # <b>`Categorical columns with missing data`<b>

    cat_col_na = [
        col for col in categorical_columns if df[col].isnull().sum() > 0]

    # variables to impute with the string missing
    with_string_missing = [
        col for col in cat_col_na if df[col].isnull().mean() > 0.1]

    # variables to impute with the most frequent category
    with_frequent_category = [
        col for col in cat_col_na if df[col].isnull().mean() < 0.1]

    # replace missing values with new label: "missing"

    df_full_train[with_string_missing] = df_full_train[with_string_missing].fillna(
        'missing')
    df_train[with_string_missing] = df_train[with_string_missing].fillna(
        'missing')
    df_val[with_string_missing] = df_val[with_string_missing].fillna('missing')
    df_test[with_string_missing] = df_test[with_string_missing].fillna(
        'missing')

    # Mode is only gotten from training dataset. Replace with the mode (first value) if there are multiple modes
    for col in with_frequent_category:

        mode = df_train[col].mode()[0]

        df_full_train[col].fillna(mode, inplace=True)
        df_train[col].fillna(mode, inplace=True)
        df_val[col].fillna(mode, inplace=True)
        df_test[col].fillna(mode, inplace=True)

    # <b>`Numerical columns with missing data`<b>

    num_with_na = [
        col for col in numerical_columns if df[col].isnull().sum() > 0]

    for col in numerical_columns:

        median = df_train[col].median()

        df_full_train[col].fillna(median, inplace=True)
        df_train[col].fillna(median, inplace=True)
        df_val[col].fillna(median, inplace=True)
        df_test[col].fillna(median, inplace=True)

    # ## Using the best model - Decision Tree

    # The project_EDA file contains comparing the different models that can be used.
    # ### Training the model
    dv = DictVectorizer(sparse=False)

    train_dicts = df_train[categorical_columns +
                           numerical_columns].to_dict(orient="records")
    X_train = dv.fit_transform(train_dicts)

    val_dicts = df_val.to_dict(orient='records')
    X_val = dv.transform(val_dicts)

    test_dicts = df_test.to_dict(orient='records')
    X_test = dv.transform(test_dicts)

    dt = DecisionTreeClassifier(max_depth=10, min_samples_leaf=10)
    dt.fit(X_train, y_train)

    y_pred = dt.predict_proba(X_val)[:, 1]
    roc_score = roc_auc_score(y_val, y_pred)

    return dt, dv


dt, dv = train_function()


def create_bentoml():
    tag = bentoml.sklearn.save_model('decision_tree', dt,
                                     custom_objects={
                                         "dictVectorizer": dv
                                     }, signatures={
                                         "predict_proba": {
                                             "batchable": True,
                                             "batch_dim": 0
                                         }})
    print(tag)
    print(f'Model tag created in path: {tag.path}')

    return tag


# run bentoml funtion
create_bentoml()
