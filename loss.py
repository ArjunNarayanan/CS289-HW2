import numpy as np

def logistic_loss(g,y):
    assert len(g) == len(y)
    loss = -(y*np.log(g) + (1-y)*np.log(1-g))
    grad = (g - y)/(g*(1-g))
    return loss, grad

