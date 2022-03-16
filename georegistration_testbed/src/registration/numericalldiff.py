#!/usr/bin/env python

import numpy as np

class NumericalDiff(object):
    """An object that numerically differentiates a function."""

    def __init__(self):
        # Define your image topic
        self.initialized = False

    def derivative(self, f, a, method='central', h=0.01):
        '''Compute the difference formula for f'(a) with step size h.

        Parameters
        ----------
        f : function
            Vectorized function of one variable
        a : number
            Compute derivative at x = a
        method : string
            Difference formula: 'forward', 'backward' or 'central'
        h : number
            Step size in difference formula

        Returns
        -------
        float
            Difference formula:
                central: f(a+h) - f(a-h))/2h
                forward: f(a+h) - f(a))/h
                backward: f(a) - f(a-h))/h
        '''
        if method == 'central':
            return (f(a + h) - f(a - h))/(2*h)
        elif method == 'forward':
            return (f(a + h) - f(a))/h
        elif method == 'backward':
            return (f(a) - f(a - h))/h
        else:
            raise ValueError("Method must be 'central', 'forward' or 'backward'.")

if __name__ == '__main__':
    df = NumericalDiff()
    print(df.derivative(np.exp, 0, 'central', 0.0001))