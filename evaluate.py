
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
import lightgbm as lgbm

from train import preprocess, apply_feature_engineering

seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
tf.keras.utils.set_random_seed(seed)



def main():
    df = pd.read_csv("data/playground-series-s3e2/train.csv")
    original = pd.read_csv('data/healthcare-dataset-stroke-data.csv')
    test_df = pd.read_csv("data/playground-series-s3e2/test.csv")
    df = pd.concat([df, original], axis=0)

    df, test_df = apply_feature_engineering(df, test_df)    
    train_X, valid_X, train_y, valid_y, test_df = preprocess(df, test_df)
    
    with open('models/gboost_model.pkl', 'rb') as f:
        gboost_model = pickle.load(f)

    nn_model = keras.models.load_model('models/best_model.h5')

    with open('models/lgbm_model.pkl', 'rb') as f:
        lgbm_model = pickle.load(f)


    # Predictions
    predictions = [
        gboost_model.predict_proba(test_df)[:, 1],
        nn_model.predict(test_df).flatten(),
        lgbm_model.predict_proba(test_df)[:, 1],
    ]
    ensemble_predictions = np.mean(predictions, axis=0)

    submission = pd.read_csv("data/playground-series-s3e2/sample_submission.csv")
    submission["stroke"] = ensemble_predictions
    submission.to_csv("submission.csv", index=False)



if __name__ == "__main__":
    main()