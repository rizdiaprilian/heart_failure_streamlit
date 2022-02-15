import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, auc
from sklearn.metrics import make_scorer, recall_score, precision_score
from sklearn.metrics import roc_curve, precision_recall_curve, matthews_corrcoef

## In heart failure, 0 defines survive and 1 represents confirmed death
specificity = make_scorer(precision_score, pos_label=0)
npv = make_scorer(recall_score, pos_label=0)

# from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split

import xgboost
from xgboost import XGBClassifier

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
import pickle


def load_data(dataset):
    """
    Importing csv data, segregate categorical data from numerical formats
    Preparing train and test data 
    Data : 'normalized_data.csv'
    """
    df = pd.read_csv(dataset, sep=',')
    cat_cols = list(filter(lambda x: x if len(df[x].unique()) <= 3 else None, df.columns))
    df[cat_cols] = df[cat_cols].astype('category')

    X = df.loc[:,:"time"]
    y = df.loc[:,["DEATH_EVENT"]]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    return X_train, X_test, y_train, y_test


def pipeline(X_train, X_test, y_train, y_test):
    """
    Machine learning pipeline that transforms both train and test data
    that preventing from data leakage.
    """

    cat_columns = ['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking']
    num_columns = ['age', 'creatinine_phosphokinase', 'ejection_fraction', 
                    'platelets', 'serum_creatinine', 'serum_sodium', 'time']

    ## Transformer Pipeline
    num_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    cat_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(drop='first', dtype=np.int))
    ])

    ## Column Transformer
    preprocessor = ColumnTransformer([
        ('numeric', num_transformer, num_columns),
        ('categoric', cat_transformer, cat_columns),
    ])

    ## Apply Column Transformer
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    ## Label Encoding
    y_transformer = LabelEncoder()
    y_train = y_transformer.fit_transform(y_train).ravel()
    y_test = y_transformer.transform(y_test).ravel()

    return X_train, X_test, y_train, y_test


def training(X_train, X_test, y_train):
    lr_model = LogisticRegression(penalty = 'l2', solver = 'liblinear',
                              C = 0.25, class_weight="balanced", 
                              n_jobs=-1, max_iter=200, verbose=1)
    clf_lr = lr_model.fit(X_train, y_train)
    lr_predict = clf_lr.predict(X_test)

    rbf_svc_model = SVC(C=0.5, max_iter=200, kernel='rbf', probability=True, 
                        class_weight="balanced", random_state=42, verbose=1)
    clf_rbf_svm = rbf_svc_model.fit(X_train, y_train)
    rbf_svm_predict = clf_rbf_svm.predict(X_test)

    rf_model = RandomForestClassifier(oob_score=True, random_state=42, 
                            class_weight="balanced", n_jobs=-1, 
                            n_estimators = 200, max_depth=7, max_samples=0.8, verbose=1)
    clf_rf = rf_model.fit(X_train, y_train)
    rf_model_predict = clf_rf.predict(X_test)

    # return clf_lr, clf_rbf_svm, clf_rf
    return lr_model, rbf_svc_model, rf_model


def saving_model(lr_model, rbf_svc_model, rf_model, X_train, y_train, X_test, y_test):
    base_models = {'lr' : lr_model,
               'rbf_svc' : rbf_svc_model,
               'rf' : rf_model}
    training_splits = {'X_train' : X_train, 
                     'y_train' : y_train, 
                     'X_test' : X_test, 
                     'y_test' : y_test}
    pickle.dump(training_splits, open('./Pickle/training_splits.p', 'wb'))
    pickle.dump(base_models, open('./Pickle/base_models.p', 'wb'))


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_data('normalized_data.csv')
    X_train_enh, X_test_enh, y_train_enh, y_test_enh = pipeline(X_train, X_test, y_train, y_test)
    lr_model, rbf_svc_model, rf_model = training(X_train_enh, X_test_enh, y_train_enh)
    saving_model = saving_model(lr_model, rbf_svc_model, rf_model, X_train_enh, y_train_enh, X_test_enh, y_test_enh)
    


