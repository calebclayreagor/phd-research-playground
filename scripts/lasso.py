import sklearn
import numpy            as np
import pandas           as pd

from sklearn.metrics    import precision_recall_curve, auc


def lasso(s, X, Λ, σ, tol=1e-10, max_iter=1000):
    """
    Lasso with informed (unequal) regularization
    """

    # scaled features X and target y
    X = X.copy() / np.sum(X ** 2, axis=0)
    X, y = X.loc[:,s>0.].copy(), X[s.name].copy()

    # informed regularization strengths λ
    λ = Λ * 1./(s[s>0.].values ** σ)

    # regression coeficients β
    β = np.ones((X.shape[1],))

    # optimize until max_iters or convergence
    iter, conv = 0, np.array([False] * X.shape[1])
    while (iter < max_iter) and (conv.sum() < β.size):

        # cyclic coordinate descent
        for j in range(X.shape[1]):
            X_j, y_pred = X[X.columns[j]], (X @ β)
            ρ_j = X_j.T @ (y - y_pred + β[j]*X_j)

            β_prev = β[j]

            # soft threshold S
            if ρ_j < -λ[j]:
                β[j] = ρ_j + λ[j]
            elif ρ_j > λ[j]:
                β[j] = ρ_j - λ[j]
            else: β[j] = 0

            # check for convergence
            step = abs(β_prev - β[j])
            if step < tol: conv[j] = True

        iter += 1

    if conv.sum() < β.size:
        return None

    else:
        coefs_ = s.copy()
        coefs_.values[:] = 0.
        coefs_[s>0.] = β           # signed
        return coefs_



def inference(sce, tfbs, Λ, σ, exclude_selfs=True, y_true=None, **kwargs):
    """
    """

    y_pred = tfbs.copy()
    y_pred.values[:] = 0.

    if y_true is not None:
        y_true = y_true.copy()

    for col in y_pred.columns:
        coefs_ = lasso(tfbs[col], sce, Λ, σ, **kwargs)

        if coefs_ is not None:
            y_pred[col] = coefs_
        else: break

    if coefs_ is not None:
        y_pred = (y_pred!=0.)         # unsigned

        if y_true is None:
            return np.nan, y_pred

        else:
            if exclude_selfs==True:
                y_pred_ = y_pred.values[~np.eye(y_pred.shape[0],dtype=bool)]
                y_true_ = y_true.values[~np.eye(y_true.shape[0],dtype=bool)]
            else:
                y_pred_, y_true_ = y_pred.values.ravel(), y_true.values.ravel()

            precision, recall, _ = precision_recall_curve(y_true_, y_pred_)

            return auc(recall, precision),  y_pred

    else: return np.nan, np.nan



#####################################################################################

#import warnings
#import theano
#import pymc3            as pm
#import theano.tensor    as tt

#def lasso(tfbs, sce, lam, theta):
#    """
#    Bayesian LASSO with scaled priors (batches)
#    """
#
#    np.random.seed(123456)
#
#    sce, tfbs = sce.copy() / np.sum(sce ** 2, axis=0), tfbs.copy()
#
#    probabilities = tfbs.copy()
#    probabilities.values[:] = 0.0
#
#    with pm.Model() as model:
#        betas, X = dict(), dict()
#        for col in tfbs.columns:
#            if np.where(tfbs[col]>0.0)[0].size > 1:
#                # different features for each target variable
#                X[col] = sce[sce.columns[tfbs[col]>0.0]].values
#                scale = tfbs.loc[tfbs[col]>0.0, col].values
#
#                # place scaled laplacian priors on feature coefficients
#                betas[col] = pm.Laplace(name=f'betas_{col}', shape=scale.size,
#                                        mu=0.0, b=(scale ** theta) * 1/lam)
#
#                # individual likelihoods for target variables
#                sigma = pm.HalfNormal(f'sigma_{col}', sd=1.0)
#                _ = pm.Normal(f'lh_{col}', observed=sce[col].values,
#                              mu=tt.dot(X[col], betas[col]),
#                              sd=sigma, shape=sce.shape[0])
#
#        with warnings.catch_warnings():
#            warnings.simplefilter('ignore')
#
#            map_estimate = pm.find_MAP(model=model, progressbar=False,
#                                start={f'betas_{col}' : 0.0 for col in betas})
#
#    for col in tfbs.columns:
#        if np.where(tfbs[col]>0.0)[0].size > 1:
#            probabilities.loc[tfbs[col]>0.0, col] = abs(map_estimate[f'betas_{col}'])  # unsigned
#        else: probabilities.loc[tfbs[col]>0.0, col] = 1.0
#
#    return probabilities



#def lasso(tfbs_vec, sce, lam, theta, **kwargs):
#      """
#      One-at-a-time Bayesian Lasso
#      """
#
#      if np.where(tfbs_vec>0.0)[0].size > 1:
#
#          X, y, scale = sce.copy(), sce[tfbs_vec.name].copy(), tfbs_vec.copy()
#
#          # drop features without priors
#          X = X[X.columns[tfbs_vec > 0.0]]
#          scale = scale[tfbs_vec > 0.0]
#
#          # normalize features/targets
#          X /= np.sum(X ** 2, axis=0)
#          y /= np.sum(y ** 2)
#
#          with pm.Model() as model:
#              # place scaled laplacian priors on coefficients
#              betas = pm.Laplace(name='betas', shape=X.shape[1], mu=0.0,
#                                 b=(scale.values ** theta) * 1/lam)
#
#              # fit w/ noise to observed target values
#              sigma = pm.HalfNormal('sigma', sd=1.0)
#              target = pm.Normal('y', mu=tt.dot(X.values, betas),
#                                 sd=sigma, observed=y.values)
#
#              with warnings.catch_warnings():
#                  warnings.simplefilter('ignore')
#
#                  map_estimate = pm.find_MAP(model=model, **kwargs) # efficiency?
#
#          probabilities = tfbs_vec.copy()
#          probabilities.values[:] = 0.0
#          probabilities.loc[scale.index] = abs(map_estimate['betas'])  # unsigned
#
#      else:
#          probabilities = tfbs_vec.copy()
#          probabilities.values[:] = 0.0
#          probabilities.iloc[np.where(tfbs_vec>0.0)[0]] = 1.0
#
#      return probabilities
