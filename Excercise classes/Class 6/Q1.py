import w6_estimation as est
import numpy as np
# Set step sizes
h = np.sqrt(np.finfo(float).eps) # optimal step size for numerical gradients of "pretty" functions
l = 6e-6  # a larger step size for the outer step

def Q(theta):
    """function to be minimized """
    return 1/theta + np.exp(theta)


def minimize(Q, theta_start, maxit=100, tolerance=1e-8, h=h, l = l): 
    '''minimize: Newton-Raphson algorithm for a general function input, f. 
    '''
    theta0 = theta_start # to avoid writing to inputs later on  
      # set up numerical derivative functions 
    dQ  =  est.forward_diff(Q, theta0, h) # fill in
    ddQ = est.forward_diff(dQ, theta0, l) # fill in

    # primary algorithm loop 
    for it in range(maxit): 
        
        # Fill in update step for theta 
        grad0 = dQ(Q, theta0)
        hess0 = ddQ(dQ,theta0)

        # avoid division by (near) zero Hessian
        if abs(hess0) < 1e-12:
            print(f'Near-zero Hessian at start {theta_start} on iter {it}, stopping early.')
            break

        theta0 = theta0 - grad0 / hess0

        # check convergence 
        criterion = abs(dQ(theta0))
        if criterion <= tolerance:
            print(f'Tolerance satisfied on iteration {it}.')
            break
        
        # update previous theta
        theta0 = theta

    if it == maxit-1: 
        print(f'Warning: non-convergence after {maxit} it., theta is {theta:8.4g}.')
    
    return theta