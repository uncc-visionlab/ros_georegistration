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

    def __init__(self, fhandle, dim_input=1, dim_output=1, method='central'):
        # Define your image topic
        self.initialized = False
        self.f = fhandle
        self.inputs = dim_input
        self.outputs = dim_output
        self.set_method(method)

    def set_method(self, method='central'):
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
        # could use MATLAB's eps('double') or eps('single') when made
        # data type agnostic Eigen often uses sqrt(eps('double')) for
        # epsilon in Numerical differentiation (line 72)
        # https://github.com/libigl/eigen/blob/master/unsupported/Eigen/src/NumericalDiff/NumericalDiff.h
        # that works out to about 1.5e-8
        # Another approach to stepsize calculation is described here:
        # https://math.stackexchange.com/questions/815113/is-there-a-general-formula-for-estimating-the-step-size-h-in-numerical-different
        # h = pow(K,1/3)
        # where K = 6*eps/M_3 where :
        # eps is the roundoff error ~1.0e-16*f(p) for double precision ~1e-8*f(p) for single precision
        # M_3 is the Lipschitz bound of the 3rd derivative of the function f in the vicinity of the point where
        # the derivative is being evaluated
        self.epsilon = 1.5e-8
        #self.epsilon = 1.5e-4

    def hessian(self, xin):
        """Compute the Hessian (partial derivatives) of the vector-valued function, f, having a D-dimensional input
         and a K-dimensional output.

        Parameters
        ----------
        x : the D-dimensional input vector where the Hessian should be calculated
        Returns
        -------
        H : The (DxDxK) Hessian matrix holding the second order partial differentials of each
        component of the K-dimensional output vector with respect to the variables of the D-dimensional input vector.
        """
        xin = np.atleast_1d(xin)
        x = xin.copy()
        x_p = x
        x_m = x
        x_pp = x
        x_pm = x
        x_mp = x
        x_mm = x

        H = np.zeros((self.inputs, self.inputs, self.outputs), dtype=np.float64)

        # compute diagonal elements
        for k in range(0, self.outputs):
            for x_i in range(0, self.inputs):
                h_x_i = self.epsilon * math.sqrt(abs(x[x_i]))
                if h_x_i < self.epsilon:
                    h_x_i = self.epsilon
                val2 = np.atleast_1d(self.f(x))
                x[x_i] = x[x_i] + h_x_i
                val1 = np.atleast_1d(self.f(x))
                x[x_i] = x[x_i] - 2 * h_x_i
                val3 = np.atleast_1d(self.f(x))
                x[x_i] = xin[x_i]
                # print("H_diagonal numerator = %s %s %s " % (str(val1[k]), str(val2[k]), str(val3[k])))
                Hiik = (val1[k] - 2 * val2[k] + val3[k]) / (h_x_i * h_x_i)
                # print("H_diagonal = %s " % str(Hiik))
                H[x_i, x_i, k] = Hiik
        x = xin.copy()
        # compute off diagonal elements
        for k in range(0, self.outputs):
            for x_i in range(0, self.inputs):
                h_x_i = self.epsilon * math.sqrt(abs(x[x_i]))
                if h_x_i < self.epsilon:
                    h_x_i = self.epsilon
                # print("before x_p = %s " % str(x_p))
                x_p = x.copy()
                x_m = x.copy()
                # x_p[x_i] = x[x_i] + h_x_i
                x_p[x_i] = x_p[x_i] + h_x_i
                # print("after x_p = %s " % str(x_p))
                # x_m[x_i] = x[x_i] - h_x_i
                x_m[x_i] = x_m[x_i] - h_x_i
                for x_j in range(x_i + 1, self.inputs):
                    h_x_j = self.epsilon * math.sqrt(abs(x[x_j]))
                    if h_x_j < self.epsilon:
                        h_x_j = self.epsilon
                    x_pp = x_p.copy()
                    # print("before x_pp = %s " % str(x_pp))
                    x_pp[x_j] = x_pp[x_j] + h_x_j
                    # print("after x_pp = %s " % str(x_pp))
                    x_pm = x_p.copy()
                    x_pm[x_j] = x_pm[x_j] - h_x_j
                    x_mp = x_m.copy()
                    x_mp[x_j] = x_mp[x_j] + h_x_j
                    x_mm = x_m.copy()
                    x_mm[x_j] = x_mm[x_j] - h_x_j
                    # print("H_off_diagonal numerator x = %s %s %s %s " % (str(x_pp), str(x_pm), str(x_mp), str(x_mm)))
                    val1 = np.atleast_1d(self.f(x_pp))
                    val2 = np.atleast_1d(self.f(x_pm))
                    val3 = np.atleast_1d(self.f(x_mp))
                    val4 = np.atleast_1d(self.f(x_mm))
                    # print("H_off_diagonal numerator = %s %s %s %s " % (str(val1[k]), str(val2[k]), str(val3[k]), str(val4[k])))
                    Hijk = (val1[k] - val2[k] - val3[k] + val4[k]) / (4 * h_x_i * h_x_j)
                    # print("H_off_diagonal = %s " % str(Hijk))
                    H[x_i, x_j, k] = Hijk
                    H[x_j, x_i, k] = H[x_i, x_j, k]

                # x_p[x_i] = xin[x_i]
                # x_m[x_i] = xin[x_i]
        if self.outputs == 1:
            H = H[:, :, 0]
        return H

    def jacobian(self, xin):
        """Compute the Jacobian (partial derivatives) of the vector-valued function, f, having a D-dimensional input
         and a K-dimensional output.

        Parameters
        ----------
        x : the D-dimensional input vector where the derivatives should be calculated
        Returns
        -------
        J : The (KxD) Jacobian matrix holding the partial differentials of the function's K-dimensional output vector
        with respect to the variables of the D-dimensional input vector.
        """
        # nfev=0;
        # epsilon = Constants.epsilon; % sqrt(eps('double'))
        # epsilon = self.epsilon
        # val1 = np.zeros((self.inputs, 1), dtype=np.float64)
        # val2 = np.zeros((self.inputs, 1), dtype=np.float64)
        # val3 = np.zeros((self.inputs, 1), dtype=np.float64)
        # val4 = np.zeros((self.inputs, 1), dtype=np.float64)
        # val5 = np.zeros((self.inputs, 1), dtype=np.float64)
        # val6 = np.zeros((self.inputs, 1), dtype=np.float64)
        # val7 = np.zeros((self.inputs, 1), dtype=np.float64)
        # val8 = np.zeros((self.inputs, 1), dtype=np.float64)
        xin = np.atleast_1d(xin)
        x = xin
        J = np.zeros((self.outputs, self.inputs), dtype=np.float64)
        # initialization
        # if self.method == NumericalDiff.Method.FORWARD_DIFFERENCE:
        # elif self.method == NumericalDiff.Method.CENTRAL_DIFFERENCE:
        # do nothing
        # elif self.method == NumericalDiff.Method.FIVE_POINT_DIFFERENCE:
        # do nothing
        # elif self.method == NumericalDiff.Method.SEVEN_POINT_DIFFERENCE:
        # do nothing
        # elif self.method == NumericalDiff.Method.NINE_POINT_DIFFERENCE:
        # do nothing
        # else:
        # print('NumericalDiff: Error no such method!\n')

        # Function Body
        for dim in range(0, self.inputs):
            h = self.epsilon * math.sqrt(abs(x[dim]))
            if h < self.epsilon:
                h = self.epsilon
            if self.method == NumericalDiff.Method.FORWARD_DIFFERENCE:
                # compute f(x)
                val1 = self.f(x)
                # nfev = nfev + 1
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
        if self.outputs == 1:
            J = J[0, :]
            print("norm(J)^2 = " + str(np.dot(J, J)))
        return J

    def derivative(self, f, a, method='central', h=0.01):
        """Compute the difference formula for f'(a) with step size h.
        https://personal.math.ubc.ca/~pwalls/math-python/differentiation/differentiation/
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
        """
        if method == 'central':
            return (f(a + h) - f(a - h)) / (2 * h)
        elif method == 'forward':
            return (f(a + h) - f(a)) / h
        elif method == 'backward':
            return (f(a) - f(a - h)) / h
        else:
            raise ValueError("Method must be 'central', 'forward' or 'backward'.")


if __name__ == '__main__':
    df = NumericalDiff(np.exp, 1, 1, 'central')
    print(df.jacobian(0))
    print(df.derivative(np.exp, 0, 'central', 0.0001))
