#!/usr/bin/env python

from enum import Enum
import math
import numpy as np


class NumericalDiff(object):
    """An object that numerically differentiates a function."""

    class Method(Enum):
        CENTRAL_DIFFERENCE = 1
        FIVE_POINT_DIFFERENCE = 2
        SEVEN_POINT_DIFFERENCE = 2
        NINE_POINT_DIFFERENCE = 3
        FORWARD_DIFFERENCE = 4

    def __init__(self, fhandle, method, dim_input=1, dim_value=1):
        # Define your image topic
        self.initialized = False
        self.f = fhandle
        self.inputs = dim_input
        self.values = dim_value
        if method == 'central':
            self.method = NumericalDiff.Method.CENTRAL_DIFFERENCE
        elif method == 'five-point':
            self.method = NumericalDiff.Method.FIVE_POINT_DIFFERENCE
        elif method == 'seven-point':
            self.method = NumericalDiff.Method.SEVEN_POINT_DIFFERENCE
        elif method == 'nine-point':
            self.method = NumericalDiff.Method.NINE_POINT_DIFFERENCE
        else:
            self.method = NumericalDiff.Method.FORWARD_DIFFERENCE

    def df(self, xin):
        # nfev=0;
        # epsilon = Constants.epsilon; % sqrt(eps('double'))
        epsilon = 1.0e-6
        val1 = np.zeros((self.inputs, 1), dtype=np.float64)
        val2 = np.zeros((self.inputs, 1), dtype=np.float64)
        val3 = np.zeros((self.inputs, 1), dtype=np.float64)
        val4 = np.zeros((self.inputs, 1), dtype=np.float64)
        val5 = np.zeros((self.inputs, 1), dtype=np.float64)
        val6 = np.zeros((self.inputs, 1), dtype=np.float64)
        val7 = np.zeros((self.inputs, 1), dtype=np.float64)
        val8 = np.zeros((self.inputs, 1), dtype=np.float64)
        J = np.zeros((self.values, self.inputs), dtype=np.float64)
        x = xin
        # initialization
        if self.method == NumericalDiff.Method.FORWARD_DIFFERENCE:
            # compute f(x)
            val1 = self.f(x)
            # nfev = nfev + 1
        # elif self.method == NumericalDiff.Method.CENTRAL_DIFFERENCE:
        # do nothing
        # elif self.method == NumericalDiff.Method.FIVE_POINT_DIFFERENCE:
        # do nothing
        # elif self.method == NumericalDiff.Method.SEVEN_POINT_DIFFERENCE:
        # do nothing
        # elif self.method == NumericalDiff.Method.NINE_POINT_DIFFERENCE:
        # do nothing
        else:
            print('NumericalDiff: Error no such method!\n')
        # Function Body
        for dim in range(0, self.inputs):
            h = epsilon * abs(math.sqrt(x[dim]))
            if h < epsilon:
                h = epsilon
            if self.method == NumericalDiff.Method.FORWARD_DIFFERENCE:
                x[dim] = x[dim] + h
                val2 = self.f(x)
                # nfev = nfev + 1
                x[dim] = xin[dim]
                J[:, dim] = (val2 - val1) / h
            elif self.method == NumericalDiff.Method.CENTRAL_DIFFERENCE:
                x[dim] = x[dim] + h
                val2 = self.f(x)
                # nfev = nfev + 1
                x[dim] = x[dim] - 2 * h
                val1 = self.f(x)
                # nfev = nfev + 1
                x[dim] = xin[dim]
                J[:, dim] = (val2 - val1) / (2 * h)
            elif self.method == NumericalDiff.Method.FIVE_POINT_DIFFERENCE:
                x[dim] = x[dim] - 2 * h
                val1 = self.f(x)
                # nfev = nfev + 1
                x[dim] = x[dim] + h
                val2 = self.f(x)
                # nfev = nfev + 1
                x[dim] = x[dim] + 2 * h
                val3 = self.f(x)
                # nfev = nfev + 1
                x[dim] = x[dim] + h
                val4 = self.f(x)
                # nfev = nfev + 1
                x[dim] = xin[dim]
                J[:, dim] = (val1 - 8 * val2 + 8 * val3 - val4) / (12 * h)
            elif self.method == NumericalDiff.Method.SEVEN_POINT_DIFFERENCE:
                x[dim] = x[dim] - 3 * h
                val1 = self.f(x)
                # nfev = nfev + 1
                x[dim] = x[dim] + h
                val2 = self.f(x)
                # nfev = nfev + 1
                x[dim] = x[dim] + h
                val3 = self.f(x)
                # nfev = nfev + 1
                x[dim] = x[dim] + 2 * h
                val4 = self.f(x)
                # nfev = nfev + 1
                x[dim] = x[dim] + h
                val5 = self.f(x)
                # nfev = nfev + 1
                x[dim] = x[dim] + h
                val6 = self.f(x)
                # nfev = nfev + 1
                x[dim] = xin[dim]
                J[:, dim] = (-val1 + 9 * val2 - 45 * val3 + 45 * val4 - 9 * val5 + val6) / (60 * h)
            elif self.method == NumericalDiff.Method.NINE_POINT_DIFFERENCE:
                x[dim] = x[dim] - 4 * h
                val1 = self.f(x)
                # nfev = nfev + 1
                x[dim] = x[dim] + h
                val2 = self.f(x)
                # nfev = nfev + 1
                x[dim] = x[dim] + h
                val3 = self.f(x)
                # nfev = nfev + 1
                x[dim] = x[dim] + h
                val4 = self.f(x)
                # nfev = nfev + 1
                x[dim] = x[dim] + 2 * h
                val5 = self.f(x)
                # nfev = nfev + 1
                x[dim] = x[dim] + h
                val6 = self.f(x)
                # nfev = nfev + 1
                x[dim] = x[dim] + h
                val7 = self.f(x)
                # nfev = nfev + 1
                x[dim] = x[dim] + h
                val8 = self.f(x)
                # nfev = nfev + 1
                x[dim] = xin[dim]
                J[:, dim] = (val1 - val8) / (280 * h) - 4 * (val2 - val7) / (105 * h) + (
                            (val3 - val6) - 4 * (val4 - val5)) / (5 * h)
            else:
                print('NumericalDiff: Error no such method!\n')
        return J

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
            return (f(a + h) - f(a - h)) / (2 * h)
        elif method == 'forward':
            return (f(a + h) - f(a)) / h
        elif method == 'backward':
            return (f(a) - f(a - h)) / h
        else:
            raise ValueError("Method must be 'central', 'forward' or 'backward'.")


if __name__ == '__main__':
    df = NumericalDiff()
    print(df.derivative(np.exp, 0, 'central', 0.0001))
