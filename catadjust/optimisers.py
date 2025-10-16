#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from tqdm.auto import tqdm


def adam(fun, theta0, args=(), alpha=1e-3, beta1=0.9, beta2=0.999, nepochs=100,
         ftol=1e-3, amin=-np.inf, amax=np.inf, k0=0, k1=0, annealing='log'):
    """Adaptive Moment Estimation gradient descent.

    Parameters
    ----------
    fun : function
        Cost function which returns cost and gradient.
    theta0 : ndarray
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
    amin : float or ndarray, optional
        Minimum value(s) allowed for input values.
    amax : float or ndarray, optional
        Maximum value(s) allowed for input values.
    k0 : float, optional
        Start value for annealing parameter.
    k1 : float, optional
        End value for annealing parameter.
    annealing : str, optional
        Annealing schedule. One of log, lin, cos.

    Returns
    -------
    res : dict
        Dictionary with final optimised values, cost function evaluation,
        gradient, number of iterations and learning curve.
    """

    x, m, v = theta0*1, 0, 0
    fs = np.zeros(nepochs)
    if 'log' in annealing.lower():
        ks = np.logspace(k0, k1, nepochs)
    elif 'lin' in annealing.lower():
        ks = np.linspace(k0, k1, nepochs)
    elif 'cos' in annealing.lower():
        log = np.logspace(k0, k1, nepochs)
        lin = np.linspace(log[0], log[-1], nepochs)
        midline = (lin+log)/2
        amplitude = (lin-log)/2
        ks = midline + amplitude*np.cos(2*np.pi*lin)
    else:
        ks = np.logspace(k0, k1, nepochs)

    pbar = tqdm(range(nepochs))
    for i in pbar:
        fs[i], grad, deltas, _ = fun(x, *(args+(None, ks[i],)))

        # Convergence checks
        if i >= 1:
            ftol_msg = f'f={fs[i]:.2e}{">" if fs[i] > ftol else "<="}{ftol:.2e}'
            pbar.set_description(f'{ftol_msg}')
            if fs[i] < ftol:
                return dict(theta=x, fun=fs[i], jac=grad, nit=i, deltas=deltas,
                            annealing=ks, fs=fs[fs>0])

        # Estimates of first and second moment of gradient
        m = (1 - beta1)*grad + beta1*m
        v = (1 - beta2)*grad**2 + beta2*v

        # Bias correction
        mhat = m/(1 - beta1**(i+1))
        vhat = v/(1 - beta2**(i+1))

        # Update step
        x -= alpha * mhat/(np.sqrt(vhat) + 1e-8)

        # Bound values
        x = np.clip(x, amin, amax)

    f, grad, deltas, _ = fun(x, *(args+(None, ks[i],)))
    print('Iteration limit reached before cost function within tolerance')
    return dict(theta=x, fun=f, jac=grad, nit=i, deltas=deltas, annealing=ks,
                fs=fs[fs>0])

def adam_mb(fun, theta0, args=(), alpha=1e-3, beta1=0.9, beta2=0.999,
            nepochs=100, batch_size=0, rng=None, nrecs=1, ftol=1e-3,
            amin=-np.inf, amax=np.inf, k0=0, k1=0, annealing='log'):
    """Adaptive Moment Estimation gradient descent for mini-batch/SGD.

    Parameters
    ----------
    fun : function
        Cost function which returns cost and gradient.
    theta0 : ndarray
        Initial values for optimisation.
    args : tuple, optional
        Arguments to be passed to cost function.
    alpha : float, optional
        Learning rate.
    beta1 : float, optional
        Exponential decay rate for gradient momentum.
    beta2 : float, optional
        Exponential decay rate for gradient variance.
    nepochs : int, optional
        Maximum number of epochs.
    batch_size : int, optional
        Batch size.
    rng : Generator, optional
        Instance of numpy.random.default_rng() for SGD and mini-batch SGD.
    nrecs : int, optional
        Number of records.
    ftol : float, optional
        Convergence criterion for cost function. Stop once the absolute
        value of the cost function is less than this.
    amin : float or ndarray, optional
        Minimum value(s) allowed for input values.
    amax : float or ndarray, optional
        Maximum value(s) allowed for input values.
    k0 : float, optional
        Start value for annealing parameter.
    k1 : float, optional
        End value for annealing parameter.
    annealing : str, optional
        Annealing schedule. One of log, lin, cos.

    Returns
    -------
    res : dict
        Dictionary with final optimised values, cost function evaluation,
        gradient, number of iterations and learning curve.
    """

    x, m, v = theta0*1, 0, 0
    fs = np.zeros(nepochs)
    nrecs_rng = np.arange(nrecs, dtype=np.int64)

    if 'log' in annealing.lower():
        ks = np.logspace(k0, k1, nepochs)
    elif 'lin' in annealing.lower():
        ks = np.linspace(k0, k1, nepochs)
    elif 'cos' in annealing.lower():
        log = np.logspace(-1, 1, nepochs)
        lin = np.linspace(log[0], log[-1], nepochs)
        midline = (lin+log)/2
        amplitude = (lin-log)/2
        ks = midline + amplitude*np.cos(2*np.pi*lin)
    else:
        ks = np.logspace(k0, k1, nepochs)
    if rng is None:
        rng = np.random.default_rng(42)

    pbar = tqdm(range(nepochs))
    for i in pbar:
        # Create mini-batch indices
        locs_mb_epoch = rng.permutation(nrecs).astype(np.int64)
        for j in np.arange(0, nrecs, batch_size):
            locs_mb = locs_mb_epoch[j:j+batch_size]
            # Don't log cost functions or deltas during minibatch
            _, grad, _, _ = fun(x, *(args+(locs_mb, ks[i])))

            # Estimates of first and second moment of gradient
            m = (1 - beta1)*grad + beta1*m
            v = (1 - beta2)*grad**2 + beta2*v

            # Bias correction
            mhat = m/(1 - beta1**(i+1))
            vhat = v/(1 - beta2**(i+1))

            # Update step
            x -= alpha * mhat/(np.sqrt(vhat) + 1e-8)

            # Bound values
            x = np.clip(x, amin, amax)

        # Log overall cost function and deltas only after each epoch
        fs[i], grad, deltas, _ = fun(x, *(args+(nrecs_rng, ks[i])))

        # Convergence checks
        if i >= 1:
            ftol_msg = f'f={fs[i]:.2e}{">" if fs[i] > ftol else "<="}{ftol:.2e}'
            pbar.set_description(f'{ftol_msg}')
            if fs[i] < ftol:
                return dict(theta=x, fun=fs[i], jac=grad, nit=i, deltas=deltas,
                            annealing=ks, fs=fs[fs>0])

    # Reevaluate cost function and gradient for all records
    f, grad, deltas, _ = fun(x, *(args+(nrecs_rng, ks[i])))
    print('Iteration limit reached before cost function within tolerance')
    return dict(theta=x, fun=f, jac=grad, nit=i, deltas=deltas, annealing=ks,
                fs=fs[fs>0])
