#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

try:
    import numba as nb
    _use_numba = True
except:
    _use_numba = False


class HazardELTAdjustment:
    """Adjust a catastrophe model location-level ELT with hazard values
    to match arbitrary target location-level hazard EEF curves as closely
    as possible by scaling event rates.
    """
    def __init__(self, elt_raw, loccol, eventcol, ratecol, hazcol):
        """Load raw location-level ELT and pre-process.

        Parameters
        ----------
        elt_raw : DataFrame
            Raw location-level ELT.
        loccol: str
            Name of column containing locationIDs.
        eventcol: str
            Name of column containing eventIDs.
        ratecol: str
            Name of column containing event rates.
        hazcol: str
            Name of column containing event-location hazard intensity.
        """

        print('This module name is deprecated and will be removed in a future update.')
        print('From now on, the ELTRateAdjustment class should be used for all rate adjustments.')

        # Load ELT to be adjusted and pre-process
        elt = elt_raw.astype({loccol: np.int64, eventcol: np.int64,
                              ratecol: np.float64, hazcol: np.float64}
                            ).drop_duplicates([loccol, eventcol]
                                             ).sort_values([loccol, hazcol],
                                                           ascending=[True, False]).dropna()
        self.loccol = loccol
        self.eventcol = eventcol
        self.ratecol = ratecol
        self.hazcol = hazcol
        self.elt = self.calc_eef(elt)
        m = self.elt.shape[0]

        # Sorted array of unique eventIDs
        self.eventIDs = np.sort(self.elt[eventcol].unique())
        self.nevents = self.eventIDs.size

        # Convert eventIDs in ELT to indices in event array
        self.loceventixs = np.searchsorted(self.eventIDs, self.elt[eventcol])

        # Indices in ELT where location changes
        locbreaks = np.nonzero(np.diff(self.elt[loccol]))[0] + 1
        self.loc_slicers = np.stack([np.r_[0, locbreaks], np.r_[locbreaks, m]]).T

    def calc_eef(self, elt):
        """Calculate EEFs from a location-level ELT sorted by descending hazard.

        Parameters
        ----------
        elt : DataFrame
            Processed and sorted (in descending hazard intensity) location-level ELT.

        Returns
        -------
        elt : DataFrame
            Input ELT with additional EEF column.
        """

        elt['eef'] = elt.groupby(self.loccol, sort=False)[self.ratecol].transform(np.cumsum)
        return elt

    def adjust(self, eefs_targ, x0=None, min_rate=1e-6, alpha=1e-4, niter=100,
               ftol=1e-3, xtol=1e-6, relative=True, wts=None, use_numba=_use_numba):
        """Adjust ELT to match location-level hazard curves.

        Parameters
        ----------
        eefs_targ : Series or ndarray
            Target EEFs in the same order as the processed ELT.
        x0 : Series or ndarray, optional
            Initial guess to use for rate adjustment.
        min_rate : float, optional
            Minimum allowable rate constraint.
        alpha : float, optional
            Learning rate in Adam gradient descent algorithm.
        niter : int, optional
            Maximum number of iterations.
        ftol : float, optional
            Convergence criterion for cost function. Stop once the absolute
            value of the cost function is less than this.
        xtol : float, optional
            Convergence criterion for values. Stop once the mean absolute
            change in x between successive iterations is less than this.
        relative : bool, optional
            Use relative (percentage) error in cost function.
        wts : ndarray, optional
            User-defined weights to apply to each location-event. By default,
            locations are equally weighted.
        use_numba : boolean, optional
            Whether to use numba for a ~50-100% speedup.

        Returns
        -------
        elt_adj : DataFrame
            Adjusted ELT.
        res : dict
            Results dict.
        fs : ndarray
            Learning curve.
        """

        eefs_targ = np.array(eefs_targ, dtype=np.float64)

        # Best initial guess for adjusted rates
        if x0 is None:
            eefs_loc = np.split(eefs_targ, self.loc_slicers[1:,0])
            rates0 = np.concatenate([np.r_[eef_loc[0], np.diff(eef_loc)] for eef_loc in eefs_loc])
            x0 = np.array(pd.DataFrame({self.eventcol: self.elt[self.eventcol].values, self.ratecol: rates0}
                                       ).groupby(self.eventcol)[self.ratecol].mean())
        else:
            x0 = np.array(x0, dtype=np.float64)

        if wts is None:
            self.wts = np.ones_like(eefs_targ, dtype=np.float64)/eefs_targ.shape[0]
        else:
            self.wts = np.array(wts, dtype=np.float64)/np.sum(wts)

        if not use_numba:
            args = (eefs_targ,)
            cost = self._cost_rel if relative else self._cost_abs
        else:
            args = (eefs_targ, self.loceventixs, self.loc_slicers, self.wts)
            cost = self._cost_rel_numba if relative else self._cost_abs_numba

        res, fs = self._adam(cost, x0, args, alpha=alpha, niter=niter, ftol=ftol, xtol=xtol, amin=min_rate)
        self.theta = pd.Series(res['x'], index=pd.Index(self.eventIDs, name=self.eventcol))
        elt_adj = self.elt.copy()
        elt_adj[self.ratecol] = res['x'][self.loceventixs]
        elt_adj = self.calc_eef(elt_adj)
        elt_adj['eef_targ'] = eefs_targ
        elt_adj['delta'] = res['deltas']
        elt_adj['wt'] = self.wts
        return elt_adj, res, fs

    def _cost_rel(self, theta, eefs_targ):
        """Cost function for fitting an ELT to a target EEF by adjusting
        event rates. Cost function is based on relative (percentage) errors.

        Parameters
        ----------
        theta : ndarray
            Rates to calculate cost function for, in unique eventID order.
        eefs_targ : ndarray
            Target EEFs for location-events in the same order as the
            pre-processed ELT.

        Returns
        -------
        cost : float
            Cost function evaluated at theta.
        cost_grad : ndarray
            Gradient of cost function.
        deltas : ndarray
            Location-event differences.
        """

        # Calculate EEFs for each location by chunked cumulative sums
        eefs_pred = np.empty_like(eefs_targ)

        # Expand event rates to event-location rates
        rates = theta[self.loceventixs]
        for a, b in self.loc_slicers:
            eefs_pred[a:b] = rates[a:b].cumsum()

        # Calculate deltas and cost function for current parameters
        deltas = ((eefs_pred/eefs_targ) - 1)
        cost = (self.wts * deltas**2).sum()

        # Calculate gradient of cost function wrt to event rates
        grad_cost = np.zeros_like(theta)
        for a, b in self.loc_slicers:
            grad_cost[self.loceventixs[a:b]] += deltas[a:b][::-1].cumsum()[::-1]*self.wts[a:b]/eefs_targ[a:b]

        return cost, 2*grad_cost, deltas

    def _cost_abs(self, theta, eefs_targ):
        """Cost function for fitting an ELT to a target EEF by adjusting
        event rates. Cost function is based on absolute errors.

        Parameters
        ----------
        theta : ndarray
            Rates to calculate cost function for, in unique eventID order.
        eefs_targ : ndarray
            Target EEFs for location-events in the same order as the
            pre-processed ELT.

        Returns
        -------
        cost : float
            Cost function evaluated at theta.
        cost_grad : ndarray
            Gradient of cost function.
        deltas : ndarray
            Location-event differences.
        """

        # Calculate EEFs for each location by chunked cumulative sums
        eefs_pred = np.empty_like(eefs_targ)

        # Expand event rates to event-location rates
        rates = theta[self.loceventixs]
        for a, b in self.loc_slicers:
            eefs_pred[a:b] = rates[a:b].cumsum()

        # Calculate deltas and cost function for current parameters
        deltas = (eefs_pred - eefs_targ)
        cost = (self.wts * deltas**2).sum()

        # Calculate gradient of cost function wrt to event rates
        grad_cost = np.zeros_like(theta)
        for a, b in self.loc_slicers:
            grad_cost[self.loceventixs[a:b]] += deltas[a:b][::-1].cumsum()[::-1]*self.wts[a:b]

        return cost, 2*grad_cost, deltas

    @staticmethod
    @nb.njit('Tuple((float64,float64[:],float64[:]))(float64[:],float64[:],int64[:],int64[:,:],float64[:])')
    def _cost_rel_numba(theta, eefs_targ, loceventixs, loc_slicers, wts):
        """Cost function for fitting an ELT to a target EEF by adjusting
        event rates. Cost function is based on relative (percentage) errors.

        Parameters
        ----------
        theta : ndarray
            Rates to calculate cost function for, in unique eventID order.
        eefs_targ : ndarray
            Target EEFs for location-events in the same order as the
            pre-processed ELT.

        Returns
        -------
        cost : float
            Cost function evaluated at theta.
        cost_grad : ndarray
            Gradient of cost function.
        deltas : ndarray
            Location-event differences.
        """

        # Calculate EEFs for each location by chunked cumulative sums
        eefs_pred = np.empty_like(eefs_targ)

        # Expand event rates to event-location rates
        rates = theta[loceventixs]
        for a, b in loc_slicers:
            eefs_pred[a:b] = rates[a:b].cumsum()

        # Calculate deltas and cost function for current parameters
        deltas = ((eefs_pred/eefs_targ) - 1)
        cost = (wts * deltas**2).sum()

        # Calculate gradient of cost function wrt to event rates
        grad_cost = np.zeros_like(theta)
        for a, b in loc_slicers:
            grad_cost[loceventixs[a:b]] += deltas[a:b][::-1].cumsum()[::-1]*wts[a:b]/eefs_targ[a:b]

        return cost, 2*grad_cost, deltas

    @staticmethod
    @nb.njit('Tuple((float64,float64[:],float64[:]))(float64[:],float64[:],int64[:],int64[:,:],float64[:])')
    def _cost_abs_numba(theta, eefs_targ, loceventixs, loc_slicers, wts):
        """Cost function for fitting an ELT to a target EEF by adjusting
        event rates. Cost function is based on absolute errors.

        Parameters
        ----------
        theta : ndarray
            Rates to calculate cost function for, in unique eventID order.
        eefs_targ : ndarray
            Target EEFs for location-events in the same order as the
            pre-processed ELT.

        Returns
        -------
        cost : float
            Cost function evaluated at theta.
        cost_grad : ndarray
            Gradient of cost function.
        deltas : ndarray
            Location-event differences.
        """

        # Calculate EEFs for each location by chunked cumulative sums
        eefs_pred = np.empty_like(eefs_targ)

        # Expand event rates to event-location rates
        rates = theta[loceventixs]
        for a, b in loc_slicers:
            eefs_pred[a:b] = rates[a:b].cumsum()

        # Calculate deltas and cost function for current parameters
        deltas = (eefs_pred - eefs_targ)
        cost = (wts * deltas**2).sum()

        # Calculate gradient of cost function wrt to event rates
        grad_cost = np.zeros_like(theta)
        for a, b in loc_slicers:
            grad_cost[loceventixs[a:b]] += deltas[a:b][::-1].cumsum()[::-1]*wts[a:b]

        return cost, 2*grad_cost, deltas

    def _adam(self, fun, x0, args=(), alpha=0.001, beta1=0.9, beta2=0.999,
              niter=1000, ftol=1e-6, xtol=1e-9, amin=-np.inf, amax=np.inf):
        """Adaptive Moment Estimation gradient descent with weight clipping.

        Parameters
        ----------
        cost : function
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
        xtol : float, optional
            Convergence criterion for values. Stop once the mean absolute
            change in x between successive iterations is less than this.
        amin : float, optional
            Minimum value allowed for input values.
        amax : float, optional
            Maximum value allowed for input values.

        Returns
        -------
        res : dict
            Dictionary with final optimised values, cost function evaluation,
            gradient and number of iterations.
        fs : ndarray
            Array of cost function evaluations at each iteration.
        """

        x, m, v = x0, 0, 0
        fs = np.zeros(niter)

        pbar = tqdm(range(niter))
        for i in pbar:
            fs[i], grad, deltas = fun(x, *args)

            # Convergence checks
            if i >= 1:
                dxa = np.abs(dx).mean()
                ftol_msg = f'f={fs[i]:.2e}{">" if fs[i] > ftol else "<="}{ftol:.2e}'
                xtol_msg = f'dx={dxa:.2e}{">" if dxa > xtol else "<="}{xtol:.2e}'
                pbar.set_description(f'{ftol_msg} | {xtol_msg}')
                if fs[i] < ftol or dxa < xtol:
                    return dict(x=x, fun=fs[i], jac=grad, nit=i, dx=dxa, deltas=deltas), fs[fs>0]

            # Estimates of first and second moment of gradient
            m = (1 - beta1)*grad + beta1*m
            v = (1 - beta2)*grad**2 + beta2*v

            # Bias correction
            mhat = m/(1 - beta1**(i+1))
            vhat = v/(1 - beta2**(i+1))

            # Update step
            dx = alpha * mhat/(np.sqrt(vhat) + 1e-8)
            x = x - dx

            # Weight clipping
            x = np.clip(x, amin, amax)

        f, grad, deltas = fun(x, *args)
        print('Warning: Iteration limit reached before cost function converged within tolerance')
        return dict(x=x, fun=f, jac=grad, nit=i, dx=dxa, deltas=deltas), fs[fs>0]
