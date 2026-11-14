# EMPLOYEE CHURN - MLOPS CLASS ASSIGNMENT
# Created by Muhammad Farzam Khan

import pandas as pd
import numpy as np
import pickle
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent

# 1) IMPORTING DATA

data = pd.read_csv(BASE_DIR / 'Employee(2).csv')
print('Data shape:', data.shape)
print(data.head())
print('\nTarget Distribution:\n', data.LeaveOrNot.value_counts(normalize=True))
print('\nMissing Values:\n', data.isna().sum())

# 2) FEATURES AND TARGET
X = data.drop(columns=['LeaveOrNot'])
y = data['LeaveOrNot']

# PaymentTier is a category: 1, 2 and 3 represent separate tier groups.
categorical_features = list(X.select_dtypes(include=[object]).columns) + ['PaymentTier']
numerical_features = list(X.select_dtypes(include=[np.float64, np.int64]).columns.difference(['PaymentTier']))

print('\nCategorical Features:\n', categorical_features)
print('\nNumerical Features:\n', numerical_features)

# 3) TRAIN / TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4) EXPORTING FEATURE INPUT METADATA (SAME FORMAT AS CLASS)
def summarize_cat(data, categorical_features):
    results = []
    for column in data[categorical_features]:
        members = data[column].unique().tolist()
        results.append([column, members])
    return pd.DataFrame(results, columns=['Column Name', 'Members'])

my_feature_dict = {
    'CATEGORICAL': summarize_cat(X_train, categorical_features).to_dict(),
    'NUMERICAL': {
        'Column Name': numerical_features,
        'Default': {col: int(X_train[col].median()) for col in numerical_features},
        'Min': {col: int(X_train[col].min()) for col in numerical_features},
        'Max': {col: int(X_train[col].max()) for col in numerical_features},
    }
}

with open(BASE_DIR / 'my_feature_dict.pkl', 'wb') as fp:
    pickle.dump(my_feature_dict, fp)
print('\nFeature schema saved as my_feature_dict.pkl')

# 5) PREPROCESSING AND FITTING PIPELINE
pipeline_num = Pipeline(steps=[
    ('simple_imputer1', SimpleImputer(strategy='median')),
    ('scale_data', StandardScaler()),
])

pipeline_cat = Pipeline(steps=[
    ('OneHotEncode', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor_stage_2 = ColumnTransformer(
    transformers=[
        ('cat', pipeline_cat, categorical_features),
        ('num', pipeline_num, numerical_features),
    ], remainder='drop'
)

preprocessor_stack = Pipeline(steps=[
    ('preprocessor_stage_2', preprocessor_stage_2)
])

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor_stack),
    ('classifier', RandomForestClassifier(random_state=42))
])

pipeline.fit(X_train, y_train)

# 6) EVALUATION
y_train_pred = pipeline.predict(X_train)
y_test_pred = pipeline.predict(X_test)
print('\nTraining Accuracy:', accuracy_score(y_train, y_train_pred))
print('Test Accuracy:', accuracy_score(y_test, y_test_pred))
print('\nClassification Report (Test):\n', classification_report(y_test, y_test_pred))
print('\nConfusion Matrix (Test):\n', confusion_matrix(y_test, y_test_pred))

# 7) EXPORT AND RELOAD FITTED MODEL
# The lecture used dill because the Telco pipeline included a lambda.
# Our employee pipeline contains no lambda, so standard pickle is sufficient.
with open(BASE_DIR / 'pipeline.pkl', 'wb') as fp:
    pickle.dump(pipeline, fp)
print('\nPipeline saved as pipeline.pkl')

with open(BASE_DIR / 'pipeline.pkl', 'rb') as fp:
    loaded_pipeline = pickle.load(fp)

my_pred_array = X_test.iloc[15:16]
print('\nSample employee input:\n', my_pred_array.to_string(index=False))
print('Prediction using reloaded pipeline:', loaded_pipeline.predict(my_pred_array))
