#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from .optimisers import adam, adam_mb

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

    def adjust(self, eefs_targ, x0=None, nepochs=100, batch_size=0, ftol=1e-3,
               alpha=0.001, beta1=0.9, beta2=0.999, relative=True, seed=42,
               min_rate=1e-6, max_rate=np.inf, wts=None, k0=0, k1=0, use_numba=_use_numba):
        """Adjust ELT to match location-level hazard curves.

        Parameters
        ----------
        eefs_targ : Series or ndarray
            Target EEFs in the same order as the processed ELT.
        x0 : Series or ndarray, optional
            Initial guess to use for rate adjustment.
        nepochs : int, optional
            Number of training epochs.
        batch_size : int, optional
            Size of batch. <1 = batch; 1 = SGD; >1 = mini-batch.
        ftol : float, optional
            Convergence criterion for cost function. Stop once the
            absolute value of the cost function is less than this.
        alpha : float, optional
            Learning rate in Adam gradient descent algorithm.
        beta1 : float, optional
            Beta1 parameter in Adam gradient descent algorithm.
        beta2 : float, optional
            Beta2 parameter in Adam gradient descent algorithm.
        relative : bool, optional
            Use relative (percentage) error in cost function.
        seed : int, optional
            Seed for random number generator used for SGD and mini-batch GD.
        min_rate : float, optional
            Minimum allowable rate constraint.
        max_rate : float, optional
            Maximum allowable rate constraint.
        wts : ndarray, optional
            User-defined weights to apply to each location-event. By default,
            locations are equally weighted.
        k0 : float, optional
            Log10 of initial annealing parameter.
        k1 : float, optional
            Log10 of final annealing parameter.
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

        # Create RNG object for SGD and mini-batch SGD
        if batch_size > 0:
            nlocs = self.loc_slicers.shape[0]
            rng = np.random.default_rng(seed)
            stoc_args = {'nrecs': nlocs, 'rng': rng, 'batch_size': batch_size}

        # Create dict to pass arguments for the optimiser
        opt_args = {'alpha': alpha, 'beta1': beta1, 'beta2': beta2,
                    'nepochs': nepochs, 'ftol': ftol, 'amin': min_rate,
                    'amax': max_rate, 'k0': k0, 'k1': k1}

        if not use_numba:
            cost_args = (eefs_targ,)
            if batch_size > 0:
                cost = self._cost_rel_minibatch if relative else self._cost_abs_minibatch
                optimise = adam_mb
                opt_args = {**opt_args, **stoc_args}
            else:
                cost = self._cost_rel if relative else self._cost_abs
                optimise = adam
        else:
            # TODO NOT IMPLEMENTED YET ===========================================================================
            cost_args = (eefs_targ, self.loceventixs, self.loc_slicers, self.wts)
            if batch_size > 0:
                cost = self._cost_rel_minibatch_numba if relative else self._cost_abs_minibatch_numba
                optimise = adam_mb
                opt_args = {**opt_args, **stoc_args}
            # /TODO NOT IMPLEMENTED YET ===========================================================================
            else:
                cost = self._cost_rel_numba if relative else self._cost_abs_numba
                optimise = adam

        # Do the optimisation)
        res, fs = optimise(cost, x0, cost_args, **opt_args)

        # Post-processing of results
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
            grad_cost[loceventixs[a:b]] += 2*deltas[a:b][::-1].cumsum()[::-1]*wts[a:b]/eefs_targ[a:b]

        return cost, grad_cost, deltas, eefs_pred

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
            grad_cost[loceventixs[a:b]] += 2*deltas[a:b][::-1].cumsum()[::-1]*wts[a:b]

        return cost, grad_cost, deltas, eefs_pred

    def _cost_rel_minibatch(self, theta, eefs_targ, locs_mb, k=1.):
        """Cost function for fitting an ELT to a target EEF by adjusting
        event rates. Cost function is based on relative (percentage) errors.
        This function implements mini-batching.

        Parameters
        ----------
        theta : ndarray
            Rates to calculate cost function for, in unique eventID order.
        eefs_targ : ndarray
            Target EEFs for location-events in the same order as the
            pre-processed ELT.
        locs_mb : ndarray
            Indices of the locations in this mini-batch.
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

        # Initialise variables
        cost = 0
        eefs_pred = np.zeros(eefs_targ.size, np.float64)
        deltas = np.zeros(eefs_targ.size, np.float64)
        grad_cost = np.zeros(theta.size, np.float64)
        r = np.full(self.wts.size, False, dtype=np.bool_)
        wts = np.zeros(self.wts.size, np.float64)

        # Expand event rates to event-location rates
        rates = theta[self.loceventixs]

        # Calculate mini-batch weights
        for a,b in self.loc_slicers[locs_mb]:
            r[a:b] = True
        wts[r] = self.wts[r]
        wts /= wts.sum()

        # Calculate deltas, cost function and gradient wrt to event rates
        for a, b in self.loc_slicers[locs_mb]:
            eefs_pred[a:b] = rates[a:b].cumsum()
            deltas[a:b] = eefs_pred[a:b]/eefs_targ[a:b] - 1
            cost += (wts[a:b] * deltas[a:b]**2).sum()
            grad_cost[self.loceventixs[a:b]] += 2*deltas[a:b][::-1].cumsum()[::-1]*wts[a:b]/eefs_targ[a:b]

        return cost, grad_cost, deltas, eefs_pred

    @staticmethod
    @nb.njit('Tuple((float64,float64[:],float64[:],float64[:]))(float64[:],float64[:],int64[:],int64[:,:],float64[:],int64[:],float64)')
    def _cost_rel_minibatch_numba(theta, eefs_targ, loceventixs, loc_slicers, wts, locs_mb, k=1.):
        """Cost function for fitting an ELT to a target EEF by adjusting
        event rates. Cost function is based on relative (percentage) errors.
        This function implements mini-batching.

        Parameters
        ----------
        theta : ndarray
            Rates to calculate cost function for, in unique eventID order.
        eefs_targ : ndarray
            Target EEFs for location-events in the same order as the
            pre-processed ELT.
        locs_mb : ndarray
            Indices of the locations in this mini-batch.
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

        # Initialise variables
        cost = 0
        eefs_pred = np.zeros(eefs_targ.size, np.float64)
        deltas = np.zeros(eefs_targ.size, np.float64)
        grad_cost = np.zeros(theta.size, np.float64)
        r = np.full(wts.size, False, dtype=np.bool_)
        wts_r = np.zeros(wts.size, np.float64)

        # Expand event rates to event-location rates
        rates = theta[loceventixs]

        # Calculate mini-batch weights
        for a,b in loc_slicers[locs_mb]:
            r[a:b] = True
        wts_r[r] = wts[r]
        wts_r /= wts_r.sum()

        # Calculate deltas, cost function and gradient wrt to event rates
        for a, b in loc_slicers[locs_mb]:
            eefs_pred[a:b] = rates[a:b].cumsum()
            deltas[a:b] = eefs_pred[a:b]/eefs_targ[a:b] - 1
            cost += (wts[a:b] * deltas[a:b]**2).sum()
            grad_cost[loceventixs[a:b]] += 2*deltas[a:b][::-1].cumsum()[::-1]*wts_r[a:b]/eefs_targ[a:b]

        return cost, grad_cost, deltas, eefs_pred
