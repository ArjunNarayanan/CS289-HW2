import numpy as np

def finite_difference(func,x,index,epsilon=1e-5):
    d = len(x)
    xplus = np.copy(x)
    xminus = np.copy(x)
    xplus[index] += epsilon
    xminus[index] -= epsilon
    return (func(xplus) - func(xminus))/(2*epsilon)

def myfunc(x):
    return np.sin(x[0])*(x[1]**2)