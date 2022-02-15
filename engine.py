from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, recall_score, precision_score
from sklearn.metrics import roc_curve
import streamlit as st
from dataclasses import dataclass
from typing import List
from typing import Optional

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from enum import Enum


### Instantenate Learner name
class LearnerName(str, Enum):
    logreg = "Logistic Regression"
    svm = "Support Vector Machine"
    randomforest = "Random Forest"

### Learning information
@dataclass
class LearnerInfo:
    learner: Optional[str] = None

### class learning LogisticRegression
@dataclass
class Logistic_Regression(LearnerInfo):
    """Learning data and make prediction with logit method, Logistic Regression"""
    C: Optional[float] = None
    max_iter: Optional[int] = None
    penalty: Optional[str] = None

    def create_algorithm(self, C, max_iter, penalty):
        return LogisticRegression(penalty = penalty, solver = 'liblinear',
                              C = C, class_weight="balanced", n_jobs=-1, 
                              max_iter= max_iter)

### class learning RandomForest
@dataclass
class Random_Forest(LearnerInfo):
    """Learning data and make prediction with tree-based method, Random Forest"""

    n_estimators: Optional[int] = None
    max_depth: Optional[int] = None
    max_samples: Optional[float] = None

    def create_algorithm(self, max_depth, max_samples, n_estimators):
        return RandomForestClassifier(oob_score=True, random_state=42, 
                                class_weight="balanced", n_jobs=-1, 
                                n_estimators=n_estimators, max_depth=max_depth, 
                                max_samples=max_samples)
### class learning SVM 
@dataclass
class SVM(LearnerInfo):
    """Learning data and make prediction with kernel-based method, SVM"""

    C: Optional[float] = None
    kernel: Optional[str] = None
    max_iter: Optional[int] = None

    def create_algorithm(self, C, kernel, max_iter):
        return SVC(C=C, max_iter=max_iter, kernel=kernel, 
                 probability=True, class_weight="balanced", random_state=42)

### Adding user control on parameter setting for learner
def classification(classifier, X_train, y_train, X_test, y_test):
    if classifier == "Logistic Regression":
        model = Logistic_Regression(LearnerName.logreg)
        C = st.sidebar.number_input("Regularization parameter", 0.01, 10.0, step=0.01, key='C')
        max_iter = st.sidebar.slider("Max_iter", value=100, min_value=50, max_value=500, step=25)
        penalty = st.sidebar.radio("Penalty", ("l1", "l2"), key="penalty")
        algo_lr = model.create_algorithm(C, max_iter, penalty)
        algo_lr.fit(X_train, y_train)
        y_predict = algo_lr.predict(X_test)
        y_prob = algo_lr.predict_proba(X_test)

    elif classifier == LearnerName.svm:
        model = SVM("Support Vector Machine")
        C = st.sidebar.number_input("Regularization parameter (C)", 0.01, 10.0, step=0.01, key='C')
        kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key="kernel")
        max_iter = st.sidebar.slider("Max_iter", value=100, min_value=50, max_value=500, step=25, key="Maximum Iteration")
        algo_svm = model.create_algorithm(C, kernel, max_iter)
        algo_svm.fit(X_train, y_train)
        y_predict = algo_svm.predict(X_test)
        y_prob = algo_svm.predict_proba(X_test)

    elif classifier == LearnerName.randomforest:
        model = Random_Forest("Random Forest")
        n_estimators = st.sidebar.slider('n_estimators', 
                        value=150, min_value=50, max_value=400, step=25)
        max_depth = st.sidebar.slider('Max_depth', 
                        value=3, min_value=2, max_value=8, step=1)
        max_samples = st.sidebar.slider('max_samples', 
                        value=0.3, min_value=0.2, max_value=0.8, step=0.1)
        algo_rf = model.create_algorithm(max_depth, max_samples, n_estimators)
        algo_rf.fit(X_train, y_train)
        y_predict = algo_rf.predict(X_test)
        y_prob = algo_rf.predict_proba(X_test)


    return y_predict, y_prob

def make_prediction(y_test, y_predict):
    precision, recall, fscore, _ = score(y_test, y_predict, average='binary')
    accuracy = accuracy_score(y_test, y_predict)
    conf_matrix_array = confusion_matrix(y_test, y_predict)
    tnr = precision_score(y_test, y_predict, pos_label=0, average='binary')
    npv = recall_score(y_test, y_predict, pos_label=0, average='binary')
    
    return precision, recall, fscore, accuracy, tnr, npv, conf_matrix_array

