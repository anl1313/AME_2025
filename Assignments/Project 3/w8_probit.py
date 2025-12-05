from operator import index
import numpy as np
from numpy import indices, random
from numpy import linalg as la
from scipy import optimize
from scipy.stats import norm
from scipy.stats import t
from sklearn.feature_selection import chi2
from tabulate import tabulate
import w8_estimation as est
import pandas as pd
from scipy.stats import norm, chi2

name = 'Probit'

# global flag to make silly checks 
# disable to increase speed 
DOCHECKS = True 

def G(z): 
    return norm.cdf(z)

def q(theta, y, x): 
    return -loglikelihood(theta, y, x)

def loglikelihood(theta, y, x):

    if DOCHECKS: 
        assert np.isin(y, [0,1]).all(), f'y must be binary: found non-binary elements.'
        assert y.ndim == 1
        assert x.ndim == 2 
        N,K = x.shape 
        assert y.shape[0] == N
        assert theta.ndim == 1 
        assert theta.size == K 

    z = x@theta
    Gxb = G(z)
    
    # we cannot take the log of 0.0
    Gxb = np.fmax(Gxb, 1e-8)    # truncate below at 1e-8 
    Gxb = np.fmin(Gxb, 1.-1e-8) # truncate above at 0.99999999

    ll = (y==1)*np.log(Gxb) + (y==0)*np.log(1.0 - Gxb) 
    return ll

def Ginv(p): 
    '''Inverse cdf
    Args. 
        p: N-array of values in [0;1] (probabilities)
    Returns
        x: N-array of values in (-inf; inf) 
    '''
    return norm.ppf(p)

def starting_values(y,x): 
    b_ols = np.linalg.solve(x.T@x, x.T@y)
    return b_ols*2.5

def predict(theta, x): 
    # the "prediction" is just Pr(y=1|x)
    yhat = G(x@theta) 
    return yhat 

def sim_data(theta: np.ndarray, N:int) -> tuple: 
    '''sim_data: simulate a dataset of size N with true K-parameter theta

    Args. 
        theta: (K,) vector of true parameters (k=0 will always be a constant)
        N (int): number of observations to simulate 
    
    Returns
        tuple: y,x
            y (float): binary outcome taking values 0.0 and 1.0
            x: (N,K) matrix of explanatory variables
    '''

    # 0. unpack parameters from theta
    # (simple, we are only estimating beta coefficients)
    beta = theta

    K = theta.size 
    assert K>1, f'Not implemented for constant-only'
    
    # 1. simulate x variables, adding a constant 
    oo = np.ones((N,1))
    xx = np.random.normal(size=(N,K-1))
    x = np.hstack([oo, xx]);
    
    # 2. simulate y values
    
    # 2.a draw error terms 
    uniforms = np.random.uniform(size=(N,))
    u = Ginv(uniforms)

    # 2.b compute latent index 
    ystar = x@beta + u
    
    # 2.b compute observed y (as a float)
    y = (ystar>=0).astype(float)

    # 3. return 
    return y, x

def compute_ape(thetahat, x, index):
    """
    Compute the Average Partial Effect (APE) on the probability of experiencing force.

    Parameters:
    - thetahat: A numpy array of estimated coefficients.
    - x: A numpy array of explanatory variables.
    - index: index of the regressor we want to calculate the average partial effect of

    Returns:
    - ape: The Average Partial Effect of the regressor we are considering.
    """

    # Number of observations
    N = x.shape[0]

    # Compute the baseline probabilities
    x_baseline = x.copy()
    x_baseline[:, 1:3] = 0 #white baseline
    baseline_probs = predict(thetahat, x_baseline)

    # Compute the counterfactual probabilities 
    x_counterfactual = x_baseline.copy()
    x_counterfactual[:, index] = 1  
    counterprobs = predict(thetahat, x_counterfactual)

    # Compute the individual-level difference in probabilities
    prob_differences = counterprobs - baseline_probs

    # Compute Average Partial Effect
    ape = np.mean(prob_differences)

    return ape

def properties(x, thetahat, print_out: bool, se: bool, indices, labels):
    """
    Compute various properties and statistics for a given dataset and estimated parameters for multiple regressors.
    
    Parameters:
    - x (numpy.ndarray): 2D array representing the dataset with dimensions (N, K),
                        where N is the number of observations, and K is the number of parameters.
    - thetahat (numpy.ndarray): Estimated parameters for the model.
    - print_out (bool): If True, print the results as a DataFrame.
    - indices (list): List of indices corresponding to the regressors we want to calculate the APE for.
    - labels (list): List of labels corresponding to each regressor.
    
    Returns:
    - If print_out is True, returns a DataFrame containing estimates, standard errors,
      t-values, and p-values for various model properties.
    - If print_out is False, returns a numpy.ndarray containing the same information.
    """

    # Initialize lists to store the results
    ape_list = []

    # Loop through the indices to compute the APE for each regressor
    for index in indices:
        ape = compute_ape(thetahat, x, index)
        ape_list.append(ape)

    # Organize the results into a DataFrame if `print_out` is True
    if print_out:
        # Create a DataFrame with the results and use the labels as the index
        data = {
            'Estimate': ape_list,
        }
        df = pd.DataFrame(data, index=labels)  # Use labels for the index
        df = df.round(3)  # Round the results to 4 decimal places
        return df
    else:
        # If `print_out` is False, return the raw data
        return {
            'Estimate': ape_list,
        }

# test af misspecifikation
def whites_imt_probit(theta, y, x):
    """
    White's Information Matrix Test (IMT) for probit model.

    Params:
    theta : (K,) array
        Estimated probit parameters
    y : (N,) array
        Binary dependent variable (0/1)
    x : (N,K) array
        Regressor matrix

    Returns
    -------
    stat : float
        IM test statistic
    pval : float
        Chi-square p-value with K(K+1)/2 degrees of freedom
    """
    # dimensions
    N, K = x.shape

    # linear index and probit objects
    z = x @ theta
    g = norm.pdf(z)       # phi(z)
    G = norm.cdf(z)       # Phi(z)

    #safety for probabilities
    eps = 1e-8
    G = np.clip(G, eps, 1 - eps)

    # score contributions s_i
    adj = g / (G * (1 - G))        
    scores = (y - G)[:, None] * adj[:, None] * x    

    #outer products  
    B_hat = scores.T @ scores / N

    #informtion matrix A_hat
    w = g**2 / (G * (1 - G))         # (N,)
    A_hat = (x.T @ (w[:, None] * x)) / N

    # 4. S-bar = B_hat - A_hat 
    S_bar = B_hat - A_hat

    #construct vech(S_i) for each observation
    m = K * (K + 1) // 2   
    S_vech = np.empty((N, m))

    # helper: index til vech  
    tri_idx = np.triu_indices(K)

    for i in range(N):
        s_i = scores[i, :][:, None]          # (K,1)
        x_i = x[i, :][:, None]               # (K,1)

        B_i = s_i @ s_i.T                    # s_i s_i'
        A_i = w[i] * (x_i @ x_i.T)           # w_i x_i x_i'

        S_i = B_i - A_i                      # (K,K)
        S_vech[i, :] = S_i[tri_idx]          # vech(S_i)

    #Omega-hat som kovarians af vech(S_i)
    S_bar_vech = S_vech.mean(axis=0)                         
    S_tilde = S_vech - S_bar_vech[None, :]                   
    Omega_hat = (S_tilde.T @ S_tilde) / N                    

    #Robust 
    Omega_inv = np.linalg.pinv(Omega_hat)

    #IM teststatistik
    vech_S_bar = S_bar[tri_idx]                              # (m,)
    stat = float(N * vech_S_bar @ Omega_inv @ vech_S_bar)

    df = m
    pval = 1 - chi2.cdf(stat, df)

    return stat, pval

def print_test_stats(stat, pval):
    """
    Printer resultater af test
    param: 
    stat: teststørrelse
    pval: p-værdi

    """
    print("White's Information Matrix Test for Probit Model Misspecification")
    print("------------------------------------------------------------------")
    print(f"Test Statistic: {stat:.4f}")
    print(f"P-value: {pval:.4f}")
    if pval < 0.05:
        print("Result: Reject the null hypothesis of correct model specification at the 5% significance level.")
    else:
        print("Result: Fail to reject the null hypothesis of correct model specification at the 5% significance level.")

def bootstrap(y, x, thetahat, indices, nB=1000):
    """
    beregner bootstrappede standardfejl for APE

    input: 
    y: binær outcome (vold eller ej), N x 1
    x: kovariater, (N x K)
    thetahat: estimator
    indices: kolonner fra den fulde kovariatvektor
    nB: antal resamplings

    returns:
    standardfejl på thetahat

    """
    # dimensioner
    N = x.shape[0]
    B = len(indices)

    # tom matrix til res
    boot_mat = np.zeros((nB, B))
    # for hver resample i rækken...
    for b in range(nB):
        try:

            # a. træk tilfældigt sample med replacement
            idx = np.random.choice(N, size=N, replace=True)
            x_b = x[idx, :]
            y_b = y[idx]

            # b. find APEs for de resamplede træk
            for j, index in enumerate(indices):
                boot_mat[b, j] = compute_ape(thetahat, x_b, index)
        except Exception as e:
            print(f"Bootstrap it. number {b} failed: {e}")
            boot_mat[b,:] = np.nan

    # bootstrap standardfejl
    se = np.nanstd(boot_mat, axis=0) # vi har ingen NaNs i udgangspuntket, men dette er robust over for dem

    return se