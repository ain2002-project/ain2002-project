import random
import os
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress tensorflow warnings

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor

import tensorflow as tf
from tensorflow import keras  # for building neural networks using TensorFlow
from sklearn.metrics import roc_auc_score  # calculates ROC AUC score
from sklearn.tree import DecisionTreeRegressor # for building decision tree regressors
from sklearn.preprocessing import OneHotEncoder  # for one-hot encoding categorical features
from sklearn.neighbors import LocalOutlierFactor  # detecting outliers
from sklearn.model_selection import GridSearchCV  # hyperparameter tuning
from sklearn.model_selection import train_test_split  # splitting data
from sklearn.ensemble import GradientBoostingClassifier  # for building gradient boosting classifiers
from sklearn.feature_selection import mutual_info_regression  # for performing mutual information feature selection

from lightgbm import LGBMClassifier


seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
tf.keras.utils.set_random_seed(seed)




def fill_missing_bmi(df: pd.DataFrame, seed: int=42):
    '''
    Fill missing bmi values in the data frame using a decision tree regressor
    '''
    df = df.copy() # Make a copy of the data frame
    desicion_tree_regressor = DecisionTreeRegressor(random_state=seed)

    # Create a new data frame with columns for age, gender, and bmi
    X = df[['age','gender','bmi']].copy()

    # Convert gender to a binary value (0 for male and 1 for female)
    X.gender = X.gender.replace({'Male':0,'Female':1}).astype(np.uint8)

    # Identify rows where bmi is missing and store in missing_rows
    nan_bmi_rows = X.bmi.isna()
    missing_rows = X[nan_bmi_rows]

    # Remove rows with missing bmi values from X and store remaining values in y
    X = X[~nan_bmi_rows]
    y = X.pop('bmi')

    desicion_tree_regressor.fit(X,y) # fitting
    predictions = desicion_tree_regressor.predict(missing_rows[['age','gender']]) # predict missing bmi values

    # Store predicted bmi values in a pandas Series with the same index as missing_rows
    predicted_bmi = pd.Series(predictions, index=missing_rows.index)
    df.loc[missing_rows.index,'bmi'] = predicted_bmi # Update the df data frame with the predicted bmi values

    return df

def calculate_risk(x):
    glucose_risk = 1 if x.glucose_levels in [2,3] else 0  # Assign glucose risk value based on glucose level category
    bmi_risk = 1 if x.bmi in [2,3,4] else 0  # Assign BMI risk value based on BMI category
    age_risk = 1 if x.age in [2,3] else 0  # Assign age risk value based on age category
    hypertension_risk = 1 if x.hypertension == 1 else 0  # Assign hypertension risk value based on hypertension status
    heart_disease_risk = 1 if x.heart_disease == 1 else 0  # Assign heart disease risk value based on heart disease status
    smoking_status_risk = 1 if x.smoking_status in ['formerly smoked', 'smokes'] else 0  # Assign smoking status risk value based on smoking status
    
    # let's calculate the total risk factor
    total_risk = glucose_risk + bmi_risk + age_risk + hypertension_risk + heart_disease_risk + smoking_status_risk
    return total_risk

def generate_features(df): 
    # Create three new columns in the DataFrame by performing arithmetic operations on existing columns
    df['age/bmi'] = df.age / df.bmi
    df['age*bmi'] = df.age * df.bmi
    df['bmi/prime'] = df.bmi / 25
#   df['obesity'] = df.avg_glucose_level * df.bmi / 1000
    df['blood_heart']= df.hypertension*df.heart_disease
    return df

def compute_outliers(data,target,df, feature, threshold=6):
    mean, std = np.mean(df), np.std(df)  # Calculate the mean and standard deviation of the data
    z_score = np.abs((df-mean) / std)   # Calculate the z-score for each data point
    good = z_score < threshold # Identify data points that are more than the threshold number of standard deviations from the mean
    
    data,target = data[good],target[good] # Remove the rejected points from the input and target data
    bad_indexes = np.where(~good)[0] # Record the indexes of the rejected points
    return data, target, bad_indexes

def apply_feature_engineering(df, test_df):
    df = df.copy()
    test_df = test_df.copy()

    # Modify the 'gender' column of the df and test_df data frames
    df['gender'] = np.where(df['gender']=='Other', np.random.choice(['Male', 'Female']), df['gender'])
    test_df['gender'] = np.where(test_df['gender']=='Other', np.random.choice(['Male', 'Female']), test_df['gender'])

    df = fill_missing_bmi(df)

    # glucose_levels
    bins = [0, 100, 126, 300 ]
    labels=['normal','pre-diabetic','diabetic']
    df['glucose_levels'] = pd.cut(df['avg_glucose_level'], bins, labels=labels)
    test_df['glucose_levels'] = pd.cut(test_df['avg_glucose_level'], bins, labels=labels)

    # bmi_category
    bins = [0, 25, 30, 40, 100]
    labels=['normal','overweight','obese', 'obese1']
    df['bmi_category'] = pd.cut(df['bmi'], bins, labels=labels)
    test_df['bmi_category'] = pd.cut(test_df['bmi'], bins, labels=labels)

    # risk factor
    df['risk factor'] = df.apply(calculate_risk, axis=1)
    test_df['risk factor'] = test_df.apply(calculate_risk, axis=1)

    df = generate_features(df)
    test_df = generate_features(test_df)

    cat_features = [
        "gender",
        "ever_married",
        "work_type",
        "Residence_type",
        "smoking_status",
        "bmi_category",
        "glucose_levels",
    ]

    df = pd.get_dummies(data=df, columns=cat_features)
    test_df = pd.get_dummies(data=test_df, columns=cat_features)


    return df, test_df

def preprocess(df, test_df):
    df = df.copy()
    test_df = test_df.copy()

    cont_FEATURES = [
        "age",
        "avg_glucose_level",
        "age/bmi",
        "bmi",
        "age*bmi",
        "bmi/prime",
    ]

    X = df.drop(["id", "stroke"], axis=1) # Remove the "id" and "stroke" columns from the DataFrame to create the input data
    y = df["stroke"] # Extract the "stroke" column as the target variable

    bad_indexes = [] # Initialize a list to record the indexes of rejected points for all features

    # Loop over all continuous features and plot outliers for each one
    for feature in cont_FEATURES:
        X,y, bad_idx = compute_outliers(X,y,X[feature], feature, threshold=6)
        bad_indexes.extend(bad_idx)


    X=X.reset_index(drop=True)
    y=y.reset_index(drop=True)

    train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.2, random_state=42)
    test_df.drop(["id"], axis=1,inplace=True) # drop the ID column

    return train_X, valid_X, train_y, valid_y, test_df


def main():
    df = pd.read_csv("data/playground-series-s3e2/train.csv")
    original = pd.read_csv('data/healthcare-dataset-stroke-data.csv')
    test_df = pd.read_csv("data/playground-series-s3e2/test.csv")
    df = pd.concat([df, original], axis=0)

    df, test_df = apply_feature_engineering(df, test_df)    
    train_X, valid_X, train_y, valid_y, test_df = preprocess(df, test_df)
    
    input_shape = train_X.shape[1] # Get the number of features in the training data

    # Define a neural network model using the Keras Sequential API
    model = keras.Sequential([
        keras.Input(shape=input_shape),  # Add an input layer with the same number of features as the training data
        keras.layers.Dense(256, activation='relu'),  # Add a fully connected dense layer with 256 neurons and ReLU activation function
        keras.layers.Dropout(0.1),  # Add a dropout layer to prevent overfitting
        keras.layers.Dense(128, activation='relu'),  # Add another fully connected dense layer with 128 neurons and ReLU activation function
        keras.layers.Dropout(0.1),  # Add another dropout layer
        keras.layers.Dense(64, activation='relu'),  # Add another fully connected dense layer with 64 neurons and ReLU activation function
        keras.layers.Dropout(0.1),  # Add another dropout layer
        keras.layers.Dense(1, activation='sigmoid', name='output')  # Add an output layer with a single neuron and sigmoid activation function
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )

    callbacks_list = [
        keras.callbacks.EarlyStopping(
            monitor='val_auc',
            patience=10,
            mode='max'
        ),
        keras.callbacks.ModelCheckpoint(
            filepath='models/best_model.h5',
            monitor='val_auc',
            save_best_only=True,
            mode='max'
        ),
    ]

    history = model.fit(
        train_X, train_y,
        validation_data=(valid_X, valid_y),
        epochs=100,
        batch_size=32,
        callbacks=callbacks_list,
        verbose=2,
    )
    nn_model = keras.models.load_model('models/best_model.h5')


    best_parameters_gboost = {'ccp_alpha': 0.0, 'learning_rate': 0.05, 'n_estimators': 100, 'n_iter_no_change': 100, 'tol': 0.0001, 'validation_fraction': 0.2}

    gboost_model = GradientBoostingClassifier(random_state=seed, **best_parameters_gboost)
    gboost_model.fit(train_X, train_y)

    # pickle the gboost model
    with open('models/gboost_model.pkl', 'wb') as f:
        pickle.dump(gboost_model, f)

    lgbm_hyperparameters = {
        'n_estimators': [100, 300, 500],
        'learning_rate': [0.01, 0.05, 0.1, 0.2,],
        'num_leaves': [2, 6, 10, 14, 18, 20],
        'max_depth': [2, 6, 8,  12,  16,  20],
        'min_child_samples': [2, 4,  8, 12, 16,  20],
        'subsample': [0.5, 0.6, 0.8, 1],
    }

    best_params_lgbm = {'learning_rate': 0.05, 'max_depth': 6, 'min_child_samples': 12, 'n_estimators': 100, 'num_leaves': 6, 'subsample': 0.5}
    lgbm_model = LGBMClassifier(random_state=seed, n_jobs=-1, **best_params_lgbm)
    lgbm_model.fit(train_X, train_y)

    with open('models/lgbm_model.pkl', 'wb') as f:
        pickle.dump(lgbm_model, f)



if __name__ == "__main__":
    main()
