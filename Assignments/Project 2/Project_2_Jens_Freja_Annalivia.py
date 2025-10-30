import numpy as np 
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from scipy.stats import norm

# Standardize function
def standardize(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0, ddof=1)
    stan = (X - mu) / sigma
    return stan


# BCCH penalty function
def BCCH(X_Z_stan, y_d):
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
    if penalty_method == 'BCCH':
        return BCCH(X_tilde, y)
    elif penalty_method == 'BRT':
        return BRT(X_tilde, y)
    elif penalty_method == 'CV':
        return CV(X_tilde, y)
    else:
        raise ValueError("Invalid penalty method. Use 'BCCH' or 'CV' or 'BRT.")


#################################### Post double lasso analysis function ####################################
def post_double_lasso_analysis(Z_stan, d, X_stan, y, penalty_method, feature_names=None):

    # Calculate penalties
    penalty_dz = calculate_penalty(Z_stan, d, penalty_method)
    penalty_yx = calculate_penalty(X_stan, y, penalty_method)

    # Display penalties with 3 decimal
    print("The first-stage penalty is = ", penalty_dz.round(3))
    print("The second-stage penalty is = ", penalty_yx.round(3))

    # Run Lasso first stage
    fit_dz = Lasso(alpha=penalty_dz, max_iter=10000).fit(Z_stan, d)
    coefs_dz = fit_dz.coef_

    # Calculate residuals: (D - psi*Z)
    resdz = d - fit_dz.predict(Z_stan)

    # Count and display non-zero coefficients
    nonzero_dz = np.nonzero(coefs_dz)[0]
    print("The number of non-zero coefficients in the first stage is {}".format(len(nonzero_dz)))
    if feature_names is not None:
        selected_features_dz = [feature_names[i] for i in nonzero_dz]
        print("Selected variables in the first stage:")
        print(selected_features_dz)

    # Run Lasso second stage
    fit_yx = Lasso(alpha=penalty_yx, max_iter=10000).fit(X_stan, y)
    coefs_yx = fit_yx.coef_

    # Calculate residuals
    resyx = y - fit_yx.predict(X_stan)

    # Calculate Y - Z@gamma (epsilon + alpha*d)
    resyxz = y - Z_stan @ coefs_yx[1:]

    # Count and display non-zero coefficients
    nonzero_yx = np.nonzero(coefs_yx)[0]
    print("The number of non-zero coefficients in the second stage is =", len(nonzero_yx))
    if feature_names is not None:
        selected_features_yx = [feature_names[i] for i in nonzero_yx]
        print("Selected variables in the second stage:")
        print(selected_features_yx)

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
    print("SE for beta_PDL_hat ", se_PDL.round(3))

    # Confidence interval
    q = norm.ppf(0.975)
    CI_low_PDL = beta_PDL - q * se_PDL
    CI_high_PDL = beta_PDL + q * se_PDL
    print("Confidence interval for alpha = ", (CI_low_PDL.round(3), CI_high_PDL.round(3)))

    return beta_PDL, se_PDL, CI_low_PDL, CI_high_PDL
    

###################################### Lasso path ######################################
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

def lasso_path_plot(X, y, feature_names=None, main_feature_names=None, penalty_methods=None, n_lambdas=100):

    n, p = X.shape
    X_std = X

    # Compute penalties 
    vlines = {}
    penalties = []
    if penalty_methods is not None:
        for name, func in penalty_methods.items():
            val = func(X_std, y)
            vlines[name] = val
            penalties.append(val)

    # Dynamic lambda grid  
    if len(penalties) > 0:
        lam_min = min(penalties) / 100
        lam_max = max(penalties) * 10
    else:
        lam_max = np.max(np.abs(X_std.T @ y)) / n
        lam_min = 0.01 * lam_max

    lambda_grid = np.logspace(np.log10(lam_min), np.log10(lam_max), n_lambdas)

    # Fit Lasso path 
    coefs = np.zeros((n_lambdas, p))
    for i, lam in enumerate(lambda_grid):
        fit = Lasso(alpha=lam, max_iter=10000).fit(X_std, y)
        coefs[i, :] = fit.coef_

    # Plot paths
    fig, ax = plt.subplots(figsize=(8, 5))
    lines = []
    for j in range(p):
        line, = ax.plot(lambda_grid, coefs[:, j],
                        label=feature_names[j] if feature_names is not None else f"X{j+1}",
                        linewidth=1)
        lines.append(line)

    ax.set_xscale('log')
    ax.set_xlabel('Penalty λ (log scale)')
    ax.set_ylabel('Coefficient value')

    # Add vertical lines and markers
    colors = {"BCCH": "red", "CV": "blue"}
    for name, lam in vlines.items():
        color = colors.get(name, "grey")
        ax.axvline(x=lam, linestyle='--', color=color, alpha=0.8)
        idx = np.argmin(np.abs(lambda_grid - lam))
        yvals = coefs[idx, :]
        ax.scatter([lam] * p, yvals, color=color, s=40, marker='o',
                   edgecolor='black', zorder=5)

    # Highlight main features 
    if feature_names is not None and main_feature_names is not None:
        legend_indices = [i for i, name in enumerate(feature_names) if name in main_feature_names]

        # Colors for highlighted features
        cmap = get_cmap("tab20")  # giver op til 20 tydelige farver
        highlight_colors = [cmap(i / max(1, len(legend_indices) - 1)) for i in range(len(legend_indices))]

        for i, line in enumerate(lines):
            if i in legend_indices:
                color_idx = legend_indices.index(i)
                line.set_label(feature_names[i])
                line.set_color(highlight_colors[color_idx])
                line.set_linewidth(1.8)
            else:
                line.set_color('grey')
                line.set_alpha(0.6)
                line.set_label(None)

        # Legend for highlighted features
        handles = [lines[i] for i in legend_indices]
        labels = [feature_names[i] for i in legend_indices]
        ax.legend(handles, labels, loc=(1.04, 0), ncol=1, fontsize='small')

    plt.tight_layout()
    plt.show()

    return lambda_grid, coefs, vlines