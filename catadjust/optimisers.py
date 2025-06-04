#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from tqdm.auto import tqdm


def gd(fun, x0, args=(), alpha=0.001, niter=1000, ftol=1e-6, amin=-np.inf,
       amax=np.inf, k0=0, k1=0):
    """Adaptive Moment Estimation gradient descent with weight clipping.

    Parameters
    ----------
    fun : function
        Cost function which returns cost and gradient.
    x0 : ndarray
        Initial values for optimisation.
    args : tuple, optional
        Arguments to be passed to cost function.
    alpha : float, optional
        Learning rate.
    niter : int, optional
        Maximum number of iterations.
    ftol : float, optional
        Convergence criterion for cost function. Stop once the absolute
        value of the cost function is less than this.
    amin : float, optional
        Minimum value allowed for input values.
    amax : float, optional
        Maximum value allowed for input values.
    k0 : float, optional
        Start value for annealing parameter.
    k1 : float, optional
        End value for annealing parameter.

    Returns
    -------
    res : dict
        Dictionary with final optimised values, cost function evaluation,
        gradient and number of iterations.
    fs : ndarray
        Array of cost function evaluations at each iteration.
    """

    x = x0*1
    fs = np.zeros(niter)
    ks = np.logspace(k0, k1, niter)

    pbar = tqdm(range(niter))
    for i in pbar:
        fs[i], grad, deltas = fun(x, *(args+(ks[i],)))

        # Convergence checks
        if i >= 1:
            ftol_msg = f'f={fs[i]:.2e}{">" if fs[i] > ftol else "<="}{ftol:.2e}'
            pbar.set_description(f'{ftol_msg}')
            if fs[i] < ftol:
                return dict(x=x, fun=fs[i], jac=grad, nit=i, deltas=deltas), fs[fs>0]

        # Update step
        x -= grad * alpha

        # Weight clipping
        x = np.clip(x, amin, amax)

    f, grad, deltas = fun(x, *(args+(ks[i],)))
    print('Warning: Iteration limit reached before cost function converged within tolerance')
    return dict(x=x, fun=f, jac=grad, nit=i, deltas=deltas), fs[fs>0]

def adam(fun, x0, args=(), alpha=0.001, beta1=0.9, beta2=0.999, niter=1000,
         ftol=1e-6, amin=-np.inf, amax=np.inf, k0=0, k1=0):
    """Adaptive Moment Estimation gradient descent with weight clipping.

    Parameters
    ----------
    fun : function
        Cost function which returns cost and gradient.
    x0 : ndarray
        Initial values for optimisation.
    args : tuple, optional
        Arguments to be passed to cost function.
    alpha : float, optional
        Learning rate.
    beta1 : float, optional
        Exponential decay rate for gradient momentum.
    beta2 : float, optional
        Exponential decay rate for gradient variance.
    niter : int, optional
        Maximum number of iterations.
    ftol : float, optional
        Convergence criterion for cost function. Stop once the absolute
        value of the cost function is less than this.
    amin : float, optional
        Minimum value allowed for input values.
    amax : float, optional
        Maximum value allowed for input values.
    k0 : float, optional
        Start value for annealing parameter.
    k1 : float, optional
        End value for annealing parameter.

    Returns
    -------
    res : dict
        Dictionary with final optimised values, cost function evaluation,
        gradient and number of iterations.
    fs : ndarray
        Array of cost function evaluations at each iteration.
    """

    x, m, v = x0*1, 0, 0
    fs = np.zeros(niter)
    ks = np.logspace(k0, k1, niter)

    pbar = tqdm(range(niter))
    for i in pbar:
        fs[i], grad, deltas, _ = fun(x, *(args+(ks[i],)))

        # Convergence checks
        if i >= 1:
            ftol_msg = f'f={fs[i]:.2e}{">" if fs[i] > ftol else "<="}{ftol:.2e}'
            pbar.set_description(f'{ftol_msg}')
            if fs[i] < ftol:
                return dict(x=x, fun=fs[i], jac=grad, nit=i, deltas=deltas), fs[fs>0]

        # Estimates of first and second moment of gradient
        m = (1 - beta1)*grad + beta1*m
        v = (1 - beta2)*grad**2 + beta2*v

        # Bias correction
        mhat = m/(1 - beta1**(i+1))
        vhat = v/(1 - beta2**(i+1))

        # Update step
        x -= alpha * mhat/(np.sqrt(vhat) + 1e-8)

        # Weight clipping
        x = np.clip(x, amin, amax)

    f, grad, deltas, _ = fun(x, *(args+(ks[i],)))
    print('Warning: Iteration limit reached before cost function converged within tolerance')
    return dict(x=x, fun=f, jac=grad, nit=i, deltas=deltas), fs[fs>0]
