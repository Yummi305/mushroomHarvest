import numpy as np

##########################################################################################
# Functions for evaluation
##########################################################################################


def confusion_matrix(y_true, y_pred, pos_label=1):
    """
    Calculates the confusion matrix of true vs predicted labels
    :param y_true: True labels
    :param y_pred: Predicted labels
    :return: Confusion matrix in the form of a 2x2 np.array
    """
    tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
    fp = np.sum((y_true != pos_label) & (y_pred == pos_label))
    fn = np.sum((y_true == pos_label) & (y_pred != pos_label))
    tn = np.sum((y_true != pos_label) & (y_pred != pos_label))
    return np.array([[tp, fn], [fp, tn]])


def accuracy_score(y_true, y_pred, pos_label=1):
    """
    Calculates the accuracy of the predictions
    :param y_true: True labels
    :param y_pred: Predicted labels
    :return: Accuracy as a float between 0 and 1
    """
    cm = confusion_matrix(y_true, y_pred, pos_label=pos_label)
    [[tp, fn], [fp, tn]] = cm  # use if desired
    acc = np.sum(np.diag(cm)) / np.sum(cm)
    return acc


def roc_curve(y_true, y_score):
    """
    Calculates the receiver operating characteristic (ROC) curve
    :param y_true: True labels
    :param y_score: Model score/probability for the positive class
    thresholds (list): thresholds used for decision making
    :return: False Positive Rate (FPR), True Positive Rate (TPR) as np.arrays
    """
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    # reverse: from high threshold to low
    thresholds = np.unique(y_score)[::-1]

    fpr = [0]  # with a maximum threshold, there are no positive predictions
    tpr = [0]
    for threshold in thresholds:
        y_pred = (y_score >= threshold)
        [[tp, fn], [fp, tn]] = confusion_matrix(y_true, y_pred)
        fpr.append(fp / (fp + tn))
        tpr.append(tp / (tp + fn))

    fpr = np.array(fpr)  # convert to numpy
    tpr = np.array(tpr)
    return fpr, tpr, thresholds
