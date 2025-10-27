import numpy as np 
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from scipy.stats import norm

# Standardize function
def standardize(X):
    """
    Standardize the given dataset X.

    Parameters:
        X (numpy.ndarray): The dataset.

    Returns:
        numpy.ndarray: The standardized dataset.
    """
    
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0, ddof=1)
    stan = (X - mu) / sigma
    return stan


# BCCH penalty function
def BCCH(X_Z_stan, y_d):
    """
    Compute the BCCH penalty.

    Parameters:
        X_Z_stan (numpy.ndarray): The dataset.
        y_d (numpy.ndarray): The response variable.

    Returns:
        float: The BCCH penalty.
    """

    n, p = X_Z_stan.shape
    c = 1.1
    alpha = 0.05

    # BCCH pilot penalty
    yXscale = (np.max((X_Z_stan.T ** 2) @ ((y_d - np.mean(y_d)) ** 2) / n)) ** 0.5
    penalty_pilot = c / np.sqrt(n) * norm.ppf(1 - alpha / (2 * p)) * yXscale # Note: Have divided by 2 due to Python definition of Lasso
    pred = Lasso(alpha=penalty_pilot).fit(X_Z_stan, y_d).predict(X_Z_stan)
    
    # BCCH updated penalty
    eps = y_d - pred 
    epsXscale = (np.max((X_Z_stan.T ** 2) @ (eps ** 2) / n)) ** 0.5
    penalty_BCCH = c * norm.ppf(1 - alpha / (2 * p)) * epsXscale / np.sqrt(n)

    return penalty_BCCH


# CV penalty function
def CV(X_Z_stan, y_d):
    """
    Compute the penalty using cross-validation (CV).

    Parameters:
        X_Z_stan (numpy.ndarray): The dataset.
        y_d (numpy.ndarray): The response variable.

    Returns:
        float: The penalty calculated by cross-validation.
    """

    fit_CV = LassoCV(cv=5, max_iter=10000).fit(X_Z_stan, y_d)
    penalty_CV = fit_CV.alpha_
    return penalty_CV

# BRT penalty function
def BRT(X_tilde,y):
    n,p = X_tilde.shape
    sigma = np.std(y)
    c = 1.1
    alpha = 0.05
    q = norm.ppf(1 - alpha / (2 * p))
    penalty_BRT= (c * sigma / np.sqrt(n)) * q 

    return penalty_BRT


def calculate_penalty(X_tilde, y, penalty_method):
    """
    Calculate penalty based on the chosen method.

    Parameters:
        X_tilde (numpy.ndarray): The dataset.
        y (numpy.ndarray): The response variable.
        penalty_method (str): The chosen penalty method ('BCCH' or 'CV').

    Returns:
        float: The calculated penalty.
    """

    if penalty_method == 'BCCH':
        return BCCH(X_tilde, y)
    elif penalty_method == 'BRT':
        return BRT(X_tilde, y)
    elif penalty_method == 'CV':
        return CV(X_tilde, y)
    else:
        raise ValueError("Invalid penalty method. Use 'BCCH' or 'CV' or 'BRT.")


# Post double lasso analysis function
def post_double_lasso_analysis(Z_stan, d, X_stan, y, penalty_method):
    """
    Perform post double lasso analysis.

    Parameters:
        Z_stan (numpy.ndarray): The first stage covariates.
        d (numpy.ndarray): The first stage response variable.
        X_stan (numpy.ndarray): The second stage covariates.
        y (numpy.ndarray): The second stage response variable.
        penalty_method (str): The chosen penalty method ('BCCH' or 'CV').

    Returns:
        tuple: Contains alpha, standard error, lower bound of confidence interval, and upper bound of confidence interval.
    """

    # Calculate penalties
    penalty_dz = calculate_penalty(Z_stan, d, penalty_method)
    penalty_yx = calculate_penalty(X_stan, y, penalty_method)

    # Display penalties with 3 decimal
    print("The first-stage penalty is = ", penalty_dz.round(3))
    print("The second-stage penalty is = ", penalty_yx.round(3))

    # Run Lasso first stage
    fit_dz = Lasso(alpha=penalty_dz, max_iter=10000).fit(Z_stan, d)
    coefs = fit_dz.coef_

    # Calculate residuals: (D-psi*Z)
    resdz = d - fit_dz.predict(Z_stan)

    # Count the number of non-zero coefficients
    print("The number of non-zero coefficients in the first stage is {}".format(np.count_nonzero(coefs)))

    # Run Lasso second stage
    fit_yx = Lasso(alpha=penalty_yx, max_iter=10000).fit(X_stan, y)
    coefs = fit_yx.coef_

    # Calculate residuals
    resyx = y - fit_yx.predict(X_stan)

    # Calculate Y - Z@gamma (epsilon + alpha*d)
    resyxz = y - Z_stan @ coefs[1:]

    # Count non-zero coefficients
    print("The number of non-zero coefficients in the second stage is =", np.count_nonzero(coefs))

    # Calculate beta_PDL_hat
    num = resdz.T @ resyxz
    denom = resdz.T @ d
    beta_PDL = num / denom

    # Display beta_PDL_hat
    print("beta_PDL_hat = ", beta_PDL.round(3))

    # Calculate the variance
    N = len(resdz)
    num = np.sum(resyx**2 * resdz**2) / N
    denom = ((np.sum(resdz**2)) / N) ** 2
    sigma2_PDL = num / denom

    # Calculate standard error
    se_PDL = np.sqrt(sigma2_PDL / N)

    # Display standard error
    print("SE for beta_PDL_hat ", se_PDL.round(3))

    # Calculate the quantile of the standard normal distribution that corresponds to the 95% confidence interval of a two-sided test
    q = norm.ppf(0.975)

    # Calculate confidence interval
    CI_low_PDL = beta_PDL - q * se_PDL
    CI_high_PDL = beta_PDL + q * se_PDL

    # Display confidence interval
    print("Confidence interval for alpha = ", (CI_low_PDL.round(3), CI_high_PDL.round(3)))

    return beta_PDL, se_PDL, CI_low_PDL, CI_high_PDL
    


#Lasso path
import matplotlib.pyplot as plt

def lasso_path_plot(X, y, feature_names=None, penalty_methods=None, n_lambdas=100):
    """
    Plot Lasso coefficient paths from low to high penalty (left to right).

    Parameters:
        X (ndarray): Design matrix (will be standardized if not already).
        y (ndarray): Response variable (not standardized).
        feature_names (list, optional): Variable names for the legend.
        penalty_methods (dict, optional): Dictionary of {name: function}, e.g. {"BCCH": a2.BCCH, "CV": a2.CV}.
        n_lambdas (int): Number of penalty values.

    Returns:
        lambda_grid, coefs, vlines (tuple)
    """

    n, p = X.shape

    # --- Standardize X if not standardized ---
    #X_std = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
    #X_std = np.nan_to_num(X_std)
    X_std = standardize(X)

    # --- Compute penalties ---
    vlines = {}
    penalties = []
    if penalty_methods is not None:
        for name, func in penalty_methods.items():
            val = func(X_std, y)
            vlines[name] = val
            penalties.append(val)

    # --- Dynamic lambda grid (low → high) ---
    if len(penalties) > 0:
        lam_min = min(penalties) / 10
        lam_max = max(penalties) * 10
    else:
        lam_max = np.max(np.abs(X_std.T @ y)) / n
        lam_min = 0.01 * lam_max

    # reversed order: low to high penalty (left to right)
    lambda_grid = np.logspace(np.log10(lam_min), np.log10(lam_max), n_lambdas)

    # --- Fit Lasso path ---
    coefs = np.zeros((n_lambdas, p))
    for i, lam in enumerate(lambda_grid):
        fit = Lasso(alpha=lam, max_iter=10000).fit(X_std, y)
        coefs[i, :] = fit.coef_

    # --- Plot paths ---
    fig, ax = plt.subplots(figsize=(8, 5))

    for j in range(p):
        ax.plot(lambda_grid, coefs[:, j],
                label=feature_names[j] if feature_names is not None else f"X{j+1}")

    ax.set_xscale('log')
    ax.set_xlabel('Penalty λ (log scale)')
    ax.set_ylabel('Coefficient value')
    #ax.set_title('Lasso Path')

    # --- Add vertical lines and markers ---
    colors = {"BCCH": "red", "CV": "blue"}
    for name, lam in vlines.items():
        color = colors.get(name, "grey")
        ax.axvline(x=lam, linestyle='--', color=color, alpha=0.8, label=f"{name} penalty")

        # Find nearest λ index
        idx = np.argmin(np.abs(lambda_grid - lam))
        yvals = coefs[idx, :]
        ax.scatter([lam] * p, yvals, color=color, s=40, marker='o', edgecolor='black', zorder=5)

    ax.legend().set_visible(False)
    plt.tight_layout()
    plt.show()

    return lambda_grid, coefs, vlines

