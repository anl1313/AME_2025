import numpy as np
from numpy import random
from numpy import linalg as la
from scipy import optimize
from scipy.stats import norm
from scipy.stats import t
from tabulate import tabulate
import w8_estimation as est
import pandas as pd
from scipy.stats import norm, chi2
name = 'Logit'

DOCHECKS = True 

def G(z): 
    Gz = 1. / (1. + np.exp(-z))
    return Gz

def q(theta, y, x): 
    return -loglikelihood(theta, y, x)

def loglikelihood(theta, y, x):

    if DOCHECKS: 
        assert np.isin(y, [0,1]).all(), f'y must be binary: found non-binary elements.'
        assert y.ndim == 1
        assert x.ndim == 2 
        N,K = x.shape 
        assert y.size == N
        assert theta.ndim == 1 
        assert theta.size == K 

    # 0. unpack parameters 
    # (trivial, we are just estimating the coefficients on x)
    beta = theta 
    
    # 1. latent index
    z = x@beta
    Gxb = G(z)
    
    # 2. avoid log(0.0) errors
    h = 1e-8 # a tiny number 
    Gxb = np.fmax(Gxb, h)     # truncate below at 1e-8 
    Gxb = np.fmin(Gxb, 1.0-h) # truncate above at 0.99999999

    ll = (y==1)*np.log(Gxb) + (y==0)*np.log(1.0 - Gxb) 
    return ll

def Ginv(u): 
    '''Inverse logistic cdf: u should be in (0;1)'''
    x = - np.log( (1.0-u) / u )
    return x

def starting_values(y,x): 
    b_ols = la.solve(x.T@x, x.T@y)
    return b_ols*4.0

def predict(theta, x): 
    # the "prediction" is the response probability, Pr(y=1|x)
    yhat = G(x@theta) 
    return yhat 

def sim_data(theta: np.ndarray, N:int): 
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
    # (trivial, only beta parameters)
    beta = theta

    K = theta.size 
    assert K>1, f'Only implemented for K >= 2'
    
    # 1. simulate x variables, adding a constant 
    oo = np.ones((N,1))
    xx = np.random.normal(size=(N,K-1))
    x  = np.hstack([oo, xx]);
    
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


def whites_imt_logit(theta, y, x):
    """
    White's Information Matrix Test (IMT) for logit model.

    Parameters
    ----------
    theta : (K,) array
        Estimated logit parameters
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
    N, K = x.shape

    #linear predictor
    z = x @ theta
    p = 1.0 / (1.0 + np.exp(-z))

    #safety for log(0)
    eps = 1e-8
    p = np.clip(p, eps, 1 - eps)

    #score pr. observation
    scores = (y - p)[:, None] * x          

    #outer products 
    B_hat = scores.T @ scores / N

    #A_hat: expected information matrix
    w = p * (1 - p)                         
    A_hat = (x.T @ (w[:, None] * x)) / N

    #S_bar = B_hat - A_hat
    S_bar = B_hat - A_hat

    #S_i = B_i - A_i pr. observation 
    m = K * (K + 1) // 2
    S_vech = np.empty((N, m))
    tri_idx = np.triu_indices(K)

    for i in range(N):
        s_i = scores[i, :][:, None]         
        x_i = x[i, :][:, None]             

        B_i = s_i @ s_i.T                  
        A_i = w[i] * (x_i @ x_i.T)         

        S_i = B_i - A_i                     
        S_vech[i, :] = S_i[tri_idx]        

    #Omega-hat som kovarians af vech(S_i)
    S_bar_vech = S_vech.mean(axis=0)
    S_tilde = S_vech - S_bar_vech[None, :]
    Omega_hat = (S_tilde.T @ S_tilde) / N

    Omega_inv = np.linalg.pinv(Omega_hat)

    #IM test statistic
    vech_S_bar = S_bar[tri_idx]
    stat = float(N * vech_S_bar @ Omega_inv @ vech_S_bar)

    df = m
    pval = 1 - chi2.cdf(stat, df)

    return stat, pval


def print_test_stats(stat, pval):
    """
    Print the results of White's Information Matrix Test (IMT) for probit model misspecification.

    Parameters:
    - stat: float
        The IM test statistic.
    - pval: float
        The p-value associated with the test statistic.

    Returns:
    None
    """
    print("White's Information Matrix Test for Probit Model Misspecification")
    print("------------------------------------------------------------------")
    print(f"Test Statistic: {stat:.4f}")
    print(f"P-value: {pval:.4f}")
    if pval < 0.05:
        print("Result: Reject the null hypothesis of correct model specification at the 5% significance level.")
    else:
        print("Result: Fail to reject the null hypothesis of correct model specification at the 5% significance level.")
        
        

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

    # compute standard errors using delta method
    

    return ape


def properties(x, thetahat, print_out: bool, se: bool, indices, labels):
    """
    Compute various properties and statistics for a given dataset and estimated parameters for multiple regressors.
    
    Parameters:
    - x (numpy.ndarray): 2D array representing the dataset with dimensions (N, K),
                        where N is the number of observations, and K is the number of characteristics.
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

def bootstrap(y, x, thetahat, indices, nB=1000):
    """
    calculates bootstrapped std errors for APEs

    input: 
    y: binary outcome (0/1), N x 1
    x: covariates, (N x K)
    thetahat: estimator from original sample (used as starting values)
    indices: columns from the covariate vector
    nB: number of resamplings

    returns:
    Standard errors on estimator

    """
    # dimensions
    N = x.shape[0]
    B = len(indices)

    # create empty matrix to store results
    boot_mat = np.zeros((nB, B))
    # for every resample...
    for b in range(nB):
        try:
            # a. draw randomly with replacement
            idx = np.random.choice(N, size=N, replace=True) 
            x_b = x[idx, :]
            y_b = y[idx]

            # b. RE-ESTIMATE the model on the bootstrap sample
            # Use original thetahat as starting values
            theta_b = est.estimate(q, thetahat, y_b, x_b, cov_type='Sandwich')['theta']

            # c. find APEs for resampled draws using the bootstrap estimates
            for j, index in enumerate(indices):
                boot_mat[b, j] = compute_ape(theta_b, x_b, index)
        except Exception as e:
            print(f"Bootstrap iteration {b} failed: {e}")
            boot_mat[b,:] = np.nan

    # bootstrap standard errors
    se = np.nanstd(boot_mat, axis=0)

    return se
def LM_test(y, x_restricted, theta_restricted, x_additional):
    """
    Lagrange Multiplier (LM) test for inclusion of additional variables in Probit model.
    
    Tests H0: coefficients on x_additional variables = 0
    against H1: at least one coefficient on x_additional is non-zero
    
    Params:
    y : (N,) array
        Binary dependent variable (0/1)
    x_restricted : (N, K) array
        Regressor matrix for the RESTRICTED model (without additional variables)
    theta_restricted : (K,) array
        Estimated parameters from the RESTRICTED model
    x_additional : (N, q) array
        Additional regressors to test for inclusion (q = number of additional variables)
    
    Returns:
    stat : LM test statistic
    pval : P-value from chi-square distribution with q degrees of freedom
    df : Degrees of freedom (number of additional variables tested)
    """
    
    # Get dimensions
    N, K = x_restricted.shape
    # define number of additional variables being tested
    q = x_additional.shape[1]  
    
    # a. find fitted vals from restricted model
    z_restricted = x_restricted @ theta_restricted
    yhat = G(z_restricted)  
    g = yhat * (1 - yhat)
    
    # b. compute residuals from restricted model
    resid = y - yhat
    
    # c. Compute the score contributions for the additional variables
    eps = 1e-8
    yhat_safe = np.clip(yhat, eps, 1 - eps)
    
    # d. Score contributions for additional variables
    score_additional = resid[:, None] * x_additional  # (N, q)
    
    # e. Compute average score (should be close to 0 under H0)
    score_mean = score_additional.mean(axis=0)  # (q,)
    
    # f. Compute the information matrix for the additional variables
    w = yhat_safe * (1 - yhat_safe)  # p(1-p)
    I_qq = (x_additional.T @ (w[:, None] * x_additional)) / N
    
    # g. Compute LM statistic
    try:
        I_qq_inv = np.linalg.inv(I_qq)
        stat = float(N * score_mean @ I_qq_inv @ score_mean)
    except np.linalg.LinAlgError:
        # If singular, use pseudo-inverse
        I_qq_inv = np.linalg.pinv(I_qq)
        stat = float(N * score_mean @ I_qq_inv @ score_mean)
    
    # h. Compute p-value from chi-square distribution
    pval = 1 - chi2.cdf(stat, q)
    
    return stat, pval, q

def print_LM_test(stat, pval, df, var_names=None):
    """
    Print results of LM test for variable inclusion.
    
    Parameters:
    stat : LM test statistic
    pval : P-value
    df : Degrees of freedom
    var_names : Names of the additional variables being tested
    """
    print("\nLagrange Multiplier (LM) Test for Variable Inclusion")
    print("=" * 60)
    if var_names:
        print(f"Testing inclusion of: {', '.join(var_names)}")
    print(f"Degrees of freedom: {df}")
    print(f"LM Test Statistic: {stat:.4f}")
    print(f"P-value: {pval:.4f}")
    print("-" * 60)
    if pval < 0.01:
        print("Result: Reject H0 at 1% level - additional variables are significant")
    elif pval < 0.05:
        print("Result: Reject H0 at 5% level - additional variables are significant")
    elif pval < 0.10:
        print("Result: Reject H0 at 10% level - additional variables are significant")
    else:
        print("Result: Fail to reject H0 - additional variables are not significant")
    print("=" * 60) 