# Austin Hester
# Linear Regression with Regularization
# CS 4340 - Intro to Machine Learning
# 12.04.17

import numpy as np
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

def get_MSE ( y, y_hat, w_norm, lam=0.1 ):
    total = 0
    for i in range ( y.size ):
        total += ( y [i] - y_hat [i] )**2
    return total / y.size + ( lam * w_norm )

def get_Eval ( y, y_hat, w_norm, lam=0.1 ):
    Eval = get_MSE ( y, y_hat, w_norm, lam )
    return Eval

def validate ( X, y, turn=0 , lam=1 ):
    if ( turn == 0 ):
        Xtrain = X[:8]
        ytrain = y[:8]
        Xval =   X[8:]
        yval =   y[8:]
    elif ( turn == 1):
        Xtrain = X[:4]  +  X[8:12]
        ytrain = y[:4]  +  y[8:12]
        Xval =   X[4:8]
        yval =   y[4:8]
    else:
        Xtrain = X[4:]
        ytrain = y[4:]
        Xval =   X[:4]
        yval =   y[:4]
    Xt, w_lin = regularize_ridge ( Xtrain, ytrain, lam)
    y_hat = get_estimated_y (X, w_lin)
    w_norm = np.dot ( w_lin.T, w_lin )
    print ( "lambda = ", lam )
    Eval = get_Eval ( y , y_hat , w_norm, lam )
    print ( "Eval\t=", Eval )
    X1d = X.T [1] [:]
    print_results ( X1d, y, y_hat, w_lin )

    return Xt, w_lin, Eval


    
def regularize_ridge ( X, y, lam=1 ):
    XxXT = np.dot ( X.T, X )            # Compute X.T * X <- A
    Z = XxXT + ( lam * np.identity ( 2 ) )
    Zinv = np.linalg.inv ( Z )
    Xt = np.dot ( Zinv, X.T )
    w_lin = np.dot ( Xt, y.T )          # Compute weight vector w_lin = C * y
    return Xt, w_lin.T

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
    print ( "Non-regularized Linear Regression" )
    print ( "MSE = ", get_MSE(y, y_hat, w_norm, 0.1 ))
    print_results ( X1d, y, y_hat, w_lin )
    ls = [ 0.1, 1, 10, 100 ]
    Ecvs = []
    for j in ls:
        Es = []
        for i in range(3):
            print ( "---------------------------------------------------" )
            print ( "Validation", i+1 )
            Xt, w_lin, Eval = validate ( X, y, i, j )
            Es.append ( Eval )
            print( w_lin )
            y_hat = get_estimated_y ( X, w_lin )
            #plot ( X1d, y, y_hat , w_lin, i+1 )
        w_norm = np.dot ( w_lin.T, w_lin )
        avgMSE = (Es[0] + Es[1] + Es[2]) / 3
        Ecvs.append ( avgMSE )
        print ( "---------------------------------------------------" )
        print ( "lambda  = ", j )
        print ( "E _ c.v.= ", avgMSE )
        print ( "---------------------------------------------------" )
    low = 9999999999
    for i, ecv in enumerate(Ecvs):
        print ( "lambda  = ", ls[i] )
        print ( "E _ c.v.= ", ecv )
        if ( ecv < low ):
            low = ls[i]
    print ( "---------------------------------------------------" )
    print ( "We chose lambda = ", low )
    Xt, w_reg = regularize_ridge ( X, y, low )
    y_hat = get_estimated_y ( X, w_reg )
    print_results ( X1d, y, y_hat, w_reg )
    Eval = get_Eval ( y , y_hat , np.dot ( w_reg.T, w_reg ), low )
    print ( "Eval\t=", Eval )
    plot ( X1d, y, y_hat , w_lin, i+1 )

    
run ()
plt.show ()
