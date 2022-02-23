import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, auc
from sklearn.metrics import make_scorer, recall_score, precision_score
from sklearn.metrics import roc_curve, precision_recall_curve, matthews_corrcoef
from sklearn.metrics import auc
from sklearn import metrics


def load_pickle():
    """
    Loading pickle files of trained models and data suitable for evaluation
    Argument: None
    Returns:
      trained_models: a pickle file with learned parameters
      X_train, y_train, X_test, y_test: examples and test in tabular format
    """
    trained_models = pickle.load(open('base_models.p', 'rb'))
    X_train, y_train, X_test, y_test = pickle.load(open('training_splits.p', 'rb')).values()
    return trained_models, X_train, y_train, X_test, y_test

def ROC_eval(*input):
    trained_models = input[0]
    X_test = input[1]
    y_test = input[2]
    fig, ax = plt.subplots(nrows=1,ncols=1)
    # axList = axList.flatten()
    fig.set_size_inches(11, 8)
    n = 0

    # Plot the ROC-AUC curve
    for label, model in trained_models.items():
        # ax = axList[n]
        # n += 1
    # Get the probabilities for each of the two categories
        try:
            y_prob = model.predict_proba(X_test)
        except:
            continue
        fpr, tpr, thresholds = roc_curve(y_test, y_prob[:,1])
        auc = roc_auc_score(y_test, y_prob[:,1])
        ax.plot(fpr, tpr, linewidth=3, label = "{}, AUC={:.3f}".format(label, auc))
    # It is customary to draw a diagonal dotted line in ROC plots.
    # This is to indicate completely random prediction. Deviation from this
    # dotted line towards the upper left corner signifies the power of the model.
        ax.plot([0, 1], [0, 1], ls='--', color='black', lw=.3)
        ax.set(xlabel='False Positive Rate',
                ylabel='True Positive Rate',
                xlim=[-.01, 1.01], ylim=[-.01, 1.01],
                title='ROC curve\n' + label)
        ax.legend()
        ax.grid(True)
        
    plt.tight_layout()
    plt.show()

def prec_recall_eval(*input):
    trained_models = input[0]
    X_test = input[1]
    y_test = input[2]

    fig, ax = plt.subplots(nrows=1,ncols=1)
    fig.set_size_inches(11,8)
    n = 0

    for label, model in trained_models.items():
    # Plot the Precision-Recall curve
        # Get the probabilities for each of the two categories
        try:
            y_prob = model.predict_proba(X_test)
        except:
            continue
        precision, recall, thresholds = precision_recall_curve(y_test, y_prob[:,1])
        auc = metrics.auc(recall, precision)
        ax.plot(recall, precision, linewidth=3, label = "{}, AUC={:.3f}".format(label, auc))
    # It is customary to draw a diagonal dotted line in ROC plots.
    # This is to indicate completely random prediction. Deviation from this
    # dotted line towards the upper left corner signifies the power of the model.
        ax.plot([0, 1], [0, 1], ls='--', color='black', lw=.3)
        ax.set(xlabel='Recall',
                ylabel='Precision',
                xlim=[-.01, 1.01], ylim=[-.01, 1.01],
                title='Precision-Recall curve\n' + label)
        ax.legend()
        ax.grid(True)
            
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    trained_models, X_train, y_train, X_test, y_test = load_pickle()
    ROC_eval(trained_models, X_test, y_test)
    prec_recall_eval(trained_models, X_test, y_test)
