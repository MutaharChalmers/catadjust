#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from .optimisers import gd, adam

try:
    import numba as nb
    _use_numba = True
except:
    _use_numba = False


class ELTRateAdjustment:
    """Adjust a catastrophe model location-level ELT to match arbitrary target
    location-level loss or hazard EEF curves by scaling event rates.
    """
    def __init__(self, elt_raw, loccol, eventcol, ratecol, refcol):
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
        refcol: str
            Name of column containing event-location loss or hazard intensity.
        """

        # Load ELT to be adjusted and pre-process
        elt = elt_raw.astype({loccol: np.int64, eventcol: np.int64,
                              ratecol: np.float64, refcol: np.float64}
                            ).drop_duplicates([loccol, eventcol]
                                             ).sort_values([loccol, refcol],
                                                           ascending=[True, False]).dropna()
        self.loccol = loccol
        self.eventcol = eventcol
        self.ratecol = ratecol
        self.refcol = refcol
        self.elt = self.calc_eef(elt)
        m = self.elt.shape[0]

        # Sorted array of unique eventIDs
        self.eventIDs = np.sort(self.elt[eventcol].unique())
        self.nevents = self.eventIDs.size

        # Convert eventIDs in ELT to indices in event array
        self.loceventixs = np.searchsorted(self.eventIDs, self.elt[eventcol])

        # Indices in ELT where location changes
        locbreaks = np.nonzero(np.diff(self.elt[loccol]))[0] + 1
        self.loc_slicers = np.hstack([np.r_[0, locbreaks][:,None],
                                      np.r_[locbreaks, m][:,None]])

    def calc_eef(self, elt):
        """Calculate EEFs for ELT already sorted by descending loss or hazard.

        Parameters
        ----------
        elt : DataFrame
            Processed and sorted (in descending loss or hazard intensity)
            location-level ELT.

        Returns
        -------
        elt : DataFrame
            Input ELT with additional EEF column.
        """

        elt['eef'] = elt.groupby(self.loccol, sort=False)[self.ratecol].transform(np.cumsum)
        return elt

    def adjust(self, eefs_targ, x0=None, optim='adam', k0=-1, k1=1, alpha=1e-4,
               niter=100, ftol=1e-4, relative=True, min_rate=1e-6, wts=None,
               use_numba=_use_numba):
        """Adjust ELT to match location-level hazard curves.

        Parameters
        ----------
        eefs_targ : Series or ndarray
            Target EEFs in the same order as the processed ELT.
        x0 : Series or ndarray, optional
            Initial guess to use for rate adjustment.
        optim : str, optional
            Optimiser to use. One of 'adam' (default), 'sgd', 'gd'.
        k0 : float, optional
            Log10 of initial annealing parameter.
        k1 : float, optional
            Log10 of final annealing parameter.
        alpha : float, optional
            Learning rate in Adam gradient descent algorithm.
        niter : int, optional
            Maximum number of iterations.
        ftol : float, optional
            Convergence criterion for cost function. Stop once the absolute
            value of the cost function is less than this.
        relative : bool, optional
            Use relative (percentage) error in cost function.
        min_rate : float, optional
            Minimum allowable rate constraint.
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

        if optim.lower() == 'adam':
            optimise = adam
        elif optim.lower() == 'sgd':
            optimise = gd
        elif optim.lower() == 'gd':
            optimise = gd
        else:
            optimise = adam
        res, fs = optimise(cost, x0, args, alpha=alpha, niter=niter, ftol=ftol, amin=min_rate, k0=k0, k1=k1)

        self.theta = pd.Series(res['x'], index=pd.Index(self.eventIDs, name=self.eventcol))
        elt_adj = self.elt.copy()
        elt_adj[self.ratecol] = res['x'][self.loceventixs]
        elt_adj = self.calc_eef(elt_adj)
        elt_adj['eef_targ'] = eefs_targ
        elt_adj['delta'] = res['deltas']
        elt_adj['wt'] = self.wts
        return elt_adj, res, fs

    def _cost_rel(self, theta, eefs_targ, k=1.):
        """Cost function for fitting an ELT to a target EEF by adjusting
        event rates. Cost function is based on relative (percentage) errors.

        Parameters
        ----------
        theta : ndarray
            Rates to calculate cost function for, in unique eventID order.
        eefs_targ : ndarray
            Target EEFs for location-events in the same order as the
            pre-processed ELT.
        k : float, optional
            Annealing parameter.

        Returns
        -------
        cost : float
            Cost function evaluated at theta.
        cost_grad : ndarray
            Gradient of cost function.
        deltas : ndarray
            Location-event differences.
        eefs_pred : ndarray
            Predicted EEFs.
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

        return cost, 2*grad_cost, deltas, eefs_pred

    def _cost_abs(self, theta, eefs_targ, k=1.):
        """Cost function for fitting an ELT to a target EEF by adjusting
        event rates. Cost function is based on absolute errors.

        Parameters
        ----------
        theta : ndarray
            Rates to calculate cost function for, in unique eventID order.
        eefs_targ : ndarray
            Target EEFs for location-events in the same order as the
            pre-processed ELT.
        k : float, optional
            Annealing parameter.

        Returns
        -------
        cost : float
            Cost function evaluated at theta.
        cost_grad : ndarray
            Gradient of cost function.
        deltas : ndarray
            Location-event differences.
        eefs_pred : ndarray
            Predicted EEFs.
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

        return cost, 2*grad_cost, deltas, eefs_pred

    @staticmethod
    @nb.njit('Tuple((float64,float64[:],float64[:],float64[:]))(float64[:],float64[:],int64[:],int64[:,:],float64[:],float64)')
    def _cost_rel_numba(theta, eefs_targ, loceventixs, loc_slicers, wts, k=1.):
        """Cost function for fitting an ELT to a target EEF by adjusting
        event rates. Cost function is based on relative (percentage) errors.

        Parameters
        ----------
        theta : ndarray
            Rates to calculate cost function for, in unique eventID order.
        eefs_targ : ndarray
            Target EEFs for location-events in the same order as the
            pre-processed ELT.
        k : float, optional
            Annealing parameter.

        Returns
        -------
        cost : float
            Cost function evaluated at theta.
        cost_grad : ndarray
            Gradient of cost function.
        deltas : ndarray
            Location-event differences.
        eefs_pred : ndarray
            Predicted EEFs.
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

        return cost, 2*grad_cost, deltas, eefs_pred

    @staticmethod
    @nb.njit('Tuple((float64,float64[:],float64[:],float64[:]))(float64[:],float64[:],int64[:],int64[:,:],float64[:],float64)')
    def _cost_abs_numba(theta, eefs_targ, loceventixs, loc_slicers, wts, k=1.):
        """Cost function for fitting an ELT to a target EEF by adjusting
        event rates. Cost function is based on absolute errors.

        Parameters
        ----------
        theta : ndarray
            Rates to calculate cost function for, in unique eventID order.
        eefs_targ : ndarray
            Target EEFs for location-events in the same order as the
            pre-processed ELT.
        k : float, optional
            Annealing parameter.

        Returns
        -------
        cost : float
            Cost function evaluated at theta.
        cost_grad : ndarray
            Gradient of cost function.
        deltas : ndarray
            Location-event differences.
        eefs_pred : ndarray
            Predicted EEFs.
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

        return cost, 2*grad_cost, deltas, eefs_pred
