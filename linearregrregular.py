# Austin Hester
# Linear Regression with Regularization
# CS 4340 - Intro to Machine Learning
# 12.04.17

import numpy as np
import random
import matplotlib.pyplot as plt

def make_x ():
    X = [ [ 1, i-2 ] for i in range ( 13 ) ]
    return np.array ( X )

def get_y ( X ):
    y = [ x**2 + 10. for x in X ]
    return np.array ( y )

def get_estimated_y ( X, w_lin ):
    return np.array ( [ round ( yi ) for yi in np.dot ( X, w_lin) ] )

def estimate_single ( N, w_lin ):
    return np.dot ( np.array ( [1, N] ), w_lin )

def print_results ( X, y, y_hat, w_lin ):
    print ( "X =\t", X )
    print ( "Wlin =\t", w_lin )
    print ( "y  =\t", y )
    print ( "y^ =\t", y_hat.T )
    slope, intercept = np.polyfit ( X, y_hat, 1 )
    print ( "Equation of linear regression line: ", "y = ",
            slope, "*x + ", intercept )
    
def plot ( X, y, y_hat, w_lin , turn=0 ):
    fig = plt.figure ( figsize=(6,6) )
    fig.suptitle ( "Linear Regression of Y on X" )
    # Plot training points
    ax = fig.add_subplot ( 1,1,1 )
    ax.scatter ( X, y, cmap='prism' )
    # Plot line learned from linear regression
    slope, intercept = np.polyfit ( X, y_hat, 1 )
    ablines = [ slope * i + intercept for i in X ]
    cx = fig.add_subplot ( 1,1,1 )
    cx.plot ( X, ablines, 'or-' )
    plt.ylabel ( 'y & y^' )
    plt.xlabel ( 'x' )
    plt.xlim ( -2, 10 )
    plt.ylim ( 0, 120 )
    #fig.savefig ( "run%d.png" % turn )

def get_MSE ( y, y_hat, w_norm, lam=1 ):
    total = 0
    for i in range ( y.size ):
        total += ( y [i] - y_hat [i] )**2
    return total / y.size + ( lam * w_norm )

def validate ( X, y, turn=0 , lam=1 ):

    if ( turn == 0 ):
        
        Xt, w_lin = linear_regression ( X [:8], y [:8])
        y_hat = get_estimated_y (X, w_lin)
        w_norm = np.dot ( w_lin.T, w_lin )
        print( "lambda = ", lam )
        print("MSE turn ", turn+1, " = ", get_MSE (y, y_hat, w_norm, lam ))
        return Xt, w_lin
    if ( turn == 1 ):
        middle = X[:4] + X[8:12]
        ymiddle = y[:4] +  y[8:12]
        Xt, w_lin = linear_regression ( middle, ymiddle )
        y_hat = get_estimated_y (X, w_lin)
        w_norm = np.dot ( w_lin.T, w_lin )
        print( "lambda = ", lam )
        print("MSE turn ", turn+1, " = ", get_MSE (y, y_hat, w_norm , lam ))
        return Xt, w_lin
    if ( turn == 2 ):
        Xt, w_lin = linear_regression ( X [4:], y [4:] )
        y_hat = get_estimated_y (X, w_lin)
        w_norm = np.dot ( w_lin.T, w_lin )
        print( "lambda = ", lam )
        print("MSE turn ", turn+1, " = ", get_MSE (y, y_hat, w_norm, lam ))
        return Xt, w_lin
    
def regularize_ridge ( X, y, y_hat, w_lin, lam=1 ):
    w_norm = np.dot ( w_lin.T, w_lin )
    validate ( X, y )
    return get_MSE ( y, y_hat, w_norm, lam )

def linear_regression ( X, y ):
    XxXT = np.dot ( X.T, X )            # Compute X.T * X <- A
    XxXT_inv = np.linalg.inv ( XxXT )   # Compute inverse of A <- B
    Xt = np.dot ( XxXT_inv, X.T )       # Psuedo-inverse of X = B * X.T <- C
    w_lin = np.dot ( Xt, y.T )          # Compute weight vector w_lin = C * y
    return Xt, w_lin.T

def run ():
    # Get training points
    X = make_x ()
    X1d = X.T [1] [:]
    y = get_y ( X1d )
    # Run linear regression on those points
    Xt, w_lin = linear_regression ( X, y )
    y_hat = get_estimated_y ( X, w_lin )
    plot ( X1d, y, y_hat , w_lin )
    # Regularize and Validate
    w_norm = np.dot ( w_lin.T, w_lin )
    print ( "MSE = ", get_MSE(y, y_hat, w_norm, 0.1 ))
    print_results ( X1d, y, y_hat, w_lin )
    for j in [ 0.1, 1, 10, 100 ]:
        for i in range(3):
            print ( "---------------------------------------------------" )
            print ( "Validation", i+1 )
            Xt, w_lin = validate ( X, y, i, j )
            print( w_lin )
            y_hat = get_estimated_y ( X, w_lin )
            plot ( X1d, y, y_hat , w_lin, i+1 )
            print_results ( X1d, y, y_hat, w_lin )


run ()
#plt.show ()
