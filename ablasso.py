
import warnings
import sklearn
import numpy as np
import pandas as pd

from sklearn.linear_model                  import Lasso
from sklearn.metrics                       import precision_recall_curve, auc

def ablasso(X, y, weights0, alpha, beta, n_lasso_iterations=10, **kwargs):
      """
      [A]daptive Lasso with initial [B]iasing:

      0. Features are initially biased via a heuristic re-weighting
      1. Then, features are adaptively re-weighted via iterative Lasso

      Adapted from Alexandre Gramfort, BSD (3-clause) license
      """

      X, y = X.copy(), y.copy()

      # drop unweighted features
      X = X[X.columns[weights0>0.0]]

      # scale features and targets
      X /= np.sum(X ** 2, axis=0)
      y /= np.sum(y ** 2, axis=0)

      # 0: heuristic feature re-weighting
      X /= (weights0[weights0>0.0] ** beta)

      gprime = lambda w: 1./(2.*np.sqrt(np.abs(w)) + np.finfo(float).eps)

      # 1: adaptive feature re-weighting
      weights = np.ones(X.shape[1])
      for k in range(n_lasso_iterations):
          X_w = X / weights[np.newaxis, :]
          clf = Lasso(alpha=alpha, fit_intercept=False, **kwargs)
          clf.fit(X_w, y)
          coef_ = clf.coef_ / weights
          weights = gprime(coef_)

      return X.columns[coef_ > 0.0], X.columns[coef_ < 0.0]



def infer_grn(X, weights0, alpha, beta, **kwargs):

    adj = pd.DataFrame(0, index=X.columns, columns=X.columns)

    for col in X.columns:
        pos, neg = ablasso(X, X[col], weights0[col], alpha, beta, **kwargs)
        adj.loc[pos, col] = 1; adj.loc[neg, col] = -1    # signed

    return adj



def auprc(X, weights0, alpha, beta, y_true, y_pred=None, return_adj=False, **kwargs):

    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            if y_pred==None:
                grn_pred = infer_grn(X, weights0, alpha, beta, **kwargs)
                y_pred = (grn_pred.values != 0).ravel()

            # note: currently, precision & recall not signed
            pr, rec, _ = precision_recall_curve(y_true, y_pred)

            if return_adj:
                return auc(rec, pr), grn_pred
            else:
                return auc(rec, pr)

        except Warning:
            if return_adj:
                return np.nan, np.array(0)
            else:
                return np.nan
