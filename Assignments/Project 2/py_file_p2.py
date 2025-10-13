# defining code for coordinate descent
import numpy as np

def soft_threshold(rho, lambda_):
    '''Soft threshold function used for normalized data and lasso regression'''
    # i. if coefficient is than -lambda, return rho + lambda
    if rho < - lambda_:
        return (rho + lambda_)
    # ii. if coefficient is greater than lambda, return rho - lambda
    elif rho >  lambda_:
        return (rho - lambda_)
    # iii. if rho = lambda, return 0
    else: 
        return 0
    

def coordinate_descent_lasso(theta,X,y, lambda_ = .01, num_iters=100, intercept = False):
    '''Coordinate gradient descent for lasso regression - for normalized data. 
    The intercept parameter allows to specify whether or not we regularize theta_0'''
    
    #Initialisation of useful values 
    m,n = X.shape
    # a. Normalizing X
    X = X / (np.linalg.norm(X,axis = 0)) #normalizing X in case it was not done before
    
    #b. Looping until max number of iterations
    for i in range(num_iters): 
        
        #c. Looping through each coordinate
        for j in range(n):

            # d. Vectorized implementation
            X_j = X[:,j].reshape(-1,1) # taking the j'th column of X 
            y_pred = X @ theta # predicted y with current theta
            rho = X_j.T @ (y - y_pred  + theta[j]*X_j) # finding rho
        
            #Checking intercept parameter
            if intercept == True:  
                if j == 0: 
                    theta[j] =  rho 
                else:
                    theta[j] =  soft_threshold(rho, lambda_)  

            if intercept == False:
                theta[j] =  soft_threshold(rho, lambda_)   
            
    return theta.flatten()
