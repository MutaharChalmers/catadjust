#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from .optimisers import adam, adam_mb

try:
    import numba as nb
    # Numba implementation currently slower than numpy, so default to False
    _use_numba = False
except:
    _use_numba = False


class ELTLossAdjustment:
    """Adjust a catastrophe model location-level ELT to match arbitrary target
    location-level loss EEF curves by scaling event losses.
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
            Name of column containing event-location loss.
        """

        # Load ELT to be adjusted, convert datatypes, drop duplicates and sort
        elt = elt_raw.astype({loccol: str, eventcol: np.int64,
                              ratecol: np.float64, refcol: np.float64}
                              ).drop_duplicates([loccol, eventcol]).dropna()
        elt = elt.sort_values([loccol, refcol], ascending=[True, False])

        # Mapping from locationIDs to internal locids
        locations = elt[loccol].unique()
        locids = np.arange(locations.size, dtype=np.int64)
        self.locmap = dict(zip(locations, locids))
        elt['_locid'] = elt[loccol].map(self.locmap)

        self.loccol = loccol
        self.eventcol = eventcol
        self.ratecol = ratecol
        self.refcol = refcol
        self.elt = self.calc_eef(elt)
        m = self.elt.shape[0]

        # Arrays of unique eventIDs and rates in eventID order
        self.eventIDs, ix = np.unique(self.elt[eventcol], return_index=True)
        self.rates = self.elt[ratecol].values[ix]
        self.nevents = self.eventIDs.size

        # Convert eventIDs in ELT to indices in event array
        self.loceventixs = np.searchsorted(self.eventIDs, self.elt[eventcol])

        # Indices in ELT where location changes
        locbreaks = np.nonzero(np.diff(self.elt['_locid']))[0] + 1
        self.loc_slicers = np.hstack([np.r_[0, locbreaks][:,None],
                                      np.r_[locbreaks, m][:,None]])

        # Maximum EEFs in ELT by location - use to make mask for cost function
        self.max_eefs = self.elt.groupby('_locid', sort=False
                                         )['eef'].max().values[:,None]

    def expit(self, x):
        return np.exp(-np.logaddexp(0, -x))

    def calc_eef(self, elt):
        """Calculate EEFs from a location-level ELT sorted by descending hazard.

        Parameters
        ----------
        elt : DataFrame
            Processed and sorted (in descending hazard intensity) ELT.

        Returns
        -------
        elt : DataFrame
            Input ELT with additional EEF column.
        """

        elt['eef'] = elt.groupby('_locid', sort=False
                                 )[self.ratecol].transform('cumsum')
        return elt

    def adjust(self, loss_targ, eefs_targ, x0=None, nepochs=1_000, batch_size=0,
               ftol=1e-6, alpha=0.001, beta1=0.9, beta2=0.999, relative=False,
               seed=42, min_loss_fac=0, max_loss_fac=np.inf, wts=None, k0=-1,
               k1=1, annealing_schedule='log', tol=15, use_numba=_use_numba):
        """Adjust ELT losses to match location-level loss EEF curves.

        Parameters
        ----------
        loss_targ : ndarray
            Target losses in an (m, n) array for m locations and n target EEFs.
        eefs_targ : ndarray
            Target EEFs in (n,) array.
        x0 : Series or ndarray, optional
            Initial guess to use for loss adjustment.
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
        min_loss_fac : float, optional
            Minimum allowable loss factor constraint.
        max_loss_fac : float, optional
            Maximum allowable loss factor constraint.
        wts : ndarray, optional
            Weights to apply to each location-target loss. Should be the same
            shape as loss_targ. Locations are equally weighted by default.
        k0 : float, optional
            Log10 of initial annealing parameter.
        k1 : float, optional
            Log10 of final annealing parameter.
        annealing_schedule : str, optional
            Annealing schedule. One of log, lin, cos.
        tol : float, optional
            Tolerance used to mask out inputs to logistic function to speed up
            calculations on the distance matrix in cost functions.
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

        loss_targ = np.array(loss_targ, dtype=np.float64)
        eefs_targ = np.array(eefs_targ, dtype=np.float64)

        # Only count costs where EEF is valid and target losses are >=0
        cost_mask = (self.max_eefs >= eefs_targ) & (loss_targ >= 0)
        self.cost_mask = cost_mask

        # Best initial guess for loss scaling factors
        if x0 is None:
            x0 = np.ones(self.nevents)
        else:
            x0 = np.array(x0, dtype=np.float64)
        self.x0 = x0

        if wts is None:
            self.wts = np.ones_like(loss_targ, dtype=np.float64
                                    )/np.prod(loss_targ.shape)
        else:
            self.wts = np.array(wts, dtype=np.float64)/np.sum(wts)

        # Create RNG object for SGD and mini-batch SGD
        if batch_size > 0:
            nlocs = self.loc_slicers.shape[0]
            rng = np.random.default_rng(seed)
            stoc_args = {'nrecs': nlocs, 'rng': rng, 'batch_size': batch_size}

        # Create dict to pass arguments for the optimiser
        opt_args = {'alpha': alpha, 'beta1': beta1, 'beta2': beta2,
                    'nepochs': nepochs, 'ftol': ftol, 'amin': min_loss_fac,
                    'amax': max_loss_fac, 'k0': k0, 'k1': k1,
                    'annealing_schedule': annealing_schedule}

        if not use_numba:
            cost_args = (loss_targ, eefs_targ, cost_mask, tol)
            if batch_size > 0:
                # TODO NOT IMPLEMENTED YET ======================================
                #cost = self._cost_rel_mb if relative else self._cost_abs_mb
                #optimise = adam_mb
                #opt_args = {**opt_args, **stoc_args}
                print('Minibatch not yet implemented - reverting to batch')
                cost = self._cost_rel if relative else self._cost_abs
                optimise = adam
                # /TODO NOT IMPLEMENTED YET =====================================
            else:
                cost = self._cost_rel if relative else self._cost_abs
                optimise = adam
        else:
            loss = self.elt[self.refcol].values
            rates = self.elt[self.ratecol].values
            cost_args = (loss_targ, eefs_targ, cost_mask, loss, rates,
                         self.loceventixs, self.loc_slicers, self.wts, tol)
            if batch_size > 0:
                # TODO NOT IMPLEMENTED YET ======================================
                #cost = self._cost_rel_mb_nb if relative else self._cost_abs_mb_nb
                #optimise = adam_mb
                #opt_args = {**opt_args, **stoc_args}
                # /TODO NOT IMPLEMENTED YET =====================================
                print('Minibatch+numba not yet implemented - using batch+numba')
                cost = self._cost_rel_nb if relative else self._cost_abs_nb
                optimise = adam
            else:
                cost = self._cost_rel_nb if relative else self._cost_abs_nb
                optimise = adam

        # Do the optimisation
        res, fs = optimise(cost, x0, cost_args, **opt_args)

        # Post-processing of results
        event_ix = pd.Index(self.eventIDs, name=self.eventcol)
        self.theta = pd.Series(res['x'], index=event_ix)
        elt_adj = self.elt.copy()
        elt_adj[self.refcol] = res['x'][self.loceventixs]*self.elt[self.refcol]
        elt_adj = elt_adj.sort_values([self.loccol, self.refcol],
                                      ascending=[True, False])
        elt_adj = self.calc_eef(elt_adj)
        elt_adj['rp'] = 1/(1-np.exp(-elt_adj['eef']))
        return elt_adj, res, fs

    def _cost_rel(self, theta, loss_targ, eefs_targ, cost_mask, tol=20, k=1.):
        """Cost function for fitting an ELT to a target EEF by adjusting
        event losses. Cost function is based on relative (percentage) errors.

        Parameters
        ----------
        theta : ndarray
            Losses to calculate cost function for, in unique eventID order.
        loss_targ : ndarray
            2D array of target losses with rows corresponding to locations,
            and columns to EEF values which are the same for all locations.
        eefs_targ : ndarray
            1D array of target EEFs for all locations.
        cost_mask : ndarray
            Boolean mask to use only delta values at location-EEF combinations
            where the target EEF is less than or equal to the largest EEF in
            the ELT at that location.
        tol : float, optional
            Tolerance used to mask out inputs to logistic function to speed up
            calculations on the distance matrix.
        k : float, optional
            Logistic function scale parameter (or growth rate), governing the
            smoothness of the continuous approximation to the EEF function.

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

        # Initialise various arrays
        eefs_pred = np.empty_like(loss_targ, dtype=np.float64)
        deltas = np.zeros_like(loss_targ, dtype=np.float64)
        grad_cost = np.zeros_like(theta, dtype=np.float64)

        # Expand event loss factors to event-locations and scale losses
        loss = self.elt[self.refcol].values
        loss_pred = loss * theta[self.loceventixs]
        rates = self.elt[self.ratecol].values

        # Loop over locations and calculate EEFs
        for i, (a, b) in enumerate(self.loc_slicers):
            loss_ab, loss_pred_ab = loss[a:b], loss_pred[a:b]
            rates_ab = rates[a:b]

            # Logistic function of 'distance matrix' of target and predicted
            dmat = k*(loss_targ[i][:,None] - loss_pred_ab)
            logistic = np.zeros_like(dmat, dtype=np.float64)
            logistic[dmat>=tol] = 1
            mask = (dmat>-tol) & (dmat<tol)
            logistic[mask] = self.expit(dmat[mask])

            # Calculate predicted EEFs, deltas and cost function gradient
            eefs_pred[i,:] = rates_ab.sum() - logistic @ rates_ab
            deltas[i,:] = np.where(cost_mask[i], eefs_pred[i,:]/eefs_targ-1, 0)
            partial_i = rates_ab*loss_ab*logistic*(1-logistic)/eefs_targ[:,None]
            dg = 2*k*((self.wts[i]*deltas[i])[:,None]*partial_i).sum(axis=0)
            grad_cost[self.loceventixs[a:b]] += dg

        # Calculate cost function and gradient for current parameters
        cost = (self.wts * deltas**2).sum()
        return cost, grad_cost, deltas, eefs_pred

    def _cost_abs(self, theta, loss_targ, eefs_targ, cost_mask, tol=20, k=1.):
        """Cost function for fitting an ELT to a target EEF by adjusting
        event losses. Cost function is based on absolute errors.

        Parameters
        ----------
        theta : ndarray
            Losses to calculate cost function for, in unique eventID order.
        loss_targ : ndarray
            2D array of target losses with rows corresponding to locations,
            and columns to EEF values which are the same for all locations.
        eefs_targ : ndarray
            1D array of target EEFs for all locations.
        cost_mask : ndarray
            Boolean mask to use only delta values at location-EEF combinations
            where the target EEF is less than or equal to the largest EEF in
            the ELT at that location.
        tol : float, optional
            Tolerance used to mask out inputs to logistic function to speed up
            calculations on the distance matrix.
        k : float, optional
            Logistic function scale parameter (or growth rate), governing the
            smoothness of the continuous approximation to the EEF function.

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

        # Initialise various arrays
        eefs_pred = np.empty_like(loss_targ, dtype=np.float64)
        deltas = np.zeros_like(loss_targ, dtype=np.float64)
        grad_cost = np.zeros_like(theta, dtype=np.float64)

        # Expand event loss factors to event-locations and scale losses
        loss = self.elt[self.refcol].values
        loss_pred = loss * theta[self.loceventixs]
        rates = self.elt[self.ratecol].values

        # Loop over locations and calculate EEFs
        for i, (a, b) in enumerate(self.loc_slicers):
            loss_ab, loss_pred_ab = loss[a:b], loss_pred[a:b]
            rates_ab = rates[a:b]

            # Logistic function of 'distance matrix' of target and predicted
            dmat = k*(loss_targ[i][:,None] - loss_pred_ab)
            logistic = np.zeros_like(dmat, dtype=np.float64)
            logistic[dmat>=tol] = 1
            mask = (dmat>-tol) & (dmat<tol)
            logistic[mask] = self.expit(dmat[mask])

            # Calculate predicted EEFs, deltas and cost function gradient
            eefs_pred[i,:] = rates_ab.sum() - (rates_ab*logistic).sum(axis=1)
            deltas[i,:] = np.where(cost_mask[i], eefs_pred[i,:] - eefs_targ, 0)
            partial_i =  rates_ab*loss_ab*logistic*(1-logistic)
            dg =  2*k*((self.wts[i]*deltas[i])[:,None]*partial_i).sum(axis=0)
            grad_cost[self.loceventixs[a:b]] += dg

        # Calculate cost function and gradient for current parameters
        cost = (self.wts * deltas**2).sum()
        return cost, grad_cost, deltas, eefs_pred

    @staticmethod
    @nb.njit('Tuple((float64,float64[:],float64[:,:],float64[:,:]))' \
             '(float64[:],float64[:,:],float64[:],boolean[:,:],float64[:],' \
             'float64[:],int64[:],int64[:,:],float64[:,:],float64,float64)')
    def _cost_rel_nb(theta, loss_targ, eefs_targ, cost_mask, loss, rates,
                     loceventixs, loc_slicers, wts, tol=20., k=1.):
        """Cost function for fitting an ELT to a target EEF by adjusting
        event losses. Cost function is based on relative (percentage) errors.

        Parameters
        ----------
        theta : ndarray
            Losses to calculate cost function for, in unique eventID order.
        loss_targ : ndarray
            2D array of target losses with rows corresponding to locations,
            and columns to EEF values which are the same for all locations.
        eefs_targ : ndarray
            1D array of target EEFs for all locations.
        cost_mask : ndarray
            Boolean mask to use only delta values at location-EEF combinations
            where the target EEF is less than or equal to the largest EEF in
            the ELT at that location.
        loss : ndarray
            Losses from ELT.
        rates: ndarray
            Rates from ELT.
        k : float, optional
            Logistic function scale parameter (or growth rate), governing the
            smoothness of the continuous approximation to the EEF function.

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

        # Initialise various arrays
        eefs_pred = np.empty_like(loss_targ, dtype=np.float64)
        deltas = np.zeros_like(loss_targ, dtype=np.float64)
        grad_cost = np.zeros_like(theta, dtype=np.float64)

        # Expand event loss factors to event-locations and scale losses
        loss_pred = loss * theta[loceventixs]

        def expit(x):
            return np.exp(-np.logaddexp(0, -x))

        # Loop over locations and calculate EEFs
        for i, (a, b) in enumerate(loc_slicers):
            loss_ab, loss_pred_ab  = loss[a:b], loss_pred[a:b]
            rates_ab = rates[a:b]

            # Logistic function of 'distance matrix' of target and predicted
            dmat = k*(loss_targ[i][:,None] - loss_pred_ab)
            logistic = expit(dmat)

            # Calculate predicted EEFs, deltas and cost function gradient
            eefs_pred[i,:] = rates_ab.sum() - (rates_ab*logistic).sum(axis=1)
            deltas[i,:] = np.where(cost_mask[i], eefs_pred[i,:]/eefs_targ-1, 0)
            partial_i = rates_ab*loss_ab*logistic*(1-logistic)/eefs_targ[:,None]
            dg = 2*k*((wts[i]*deltas[i])[:,None]*partial_i).sum(axis=0)
            grad_cost[loceventixs[a:b]] += dg

        # Calculate cost function and gradient for current parameters
        cost = (wts * deltas**2).sum()
        return cost, grad_cost, deltas, eefs_pred

    @staticmethod
    @nb.njit('Tuple((float64,float64[:],float64[:,:],float64[:,:]))' \
             '(float64[:],float64[:,:],float64[:],boolean[:,:],float64[:],' \
             'float64[:],int64[:],int64[:,:],float64[:,:],float64,float64)')
    def _cost_abs_nb(theta, loss_targ, eefs_targ, cost_mask, loss, rates,
                     loceventixs, loc_slicers, wts, tol=20, k=1.):
        """Cost function for fitting an ELT to a target EEF by adjusting
        event losses. Cost function is based on absolute errors.

        Parameters
        ----------
        theta : ndarray
            Losses to calculate cost function for, in unique eventID order.
        loss_targ : ndarray
            2D array of target losses with rows corresponding to locations,
            and columns to EEF values which are the same for all locations.
        eefs_targ : ndarray
            1D array of target EEFs for all locations.
        cost_mask : ndarray
            Boolean mask to use only delta values at location-EEF combinations
            where the target EEF is less than or equal to the largest EEF in
            the ELT at that location.
        loss : ndarray
            Losses from ELT.
        rates: ndarray
            Rates from ELT.
        k : float, optional
            Logistic function scale parameter (or growth rate), governing the
            smoothness of the continuous approximation to the EEF function.

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

        # Initialise various arrays
        eefs_pred = np.empty_like(loss_targ, dtype=np.float64)
        deltas = np.zeros_like(loss_targ, dtype=np.float64)
        grad_cost = np.zeros_like(theta, dtype=np.float64)

        # Expand event loss factors to event-locations and scale losses
        loss_pred = loss * theta[loceventixs]

        def expit(x):
            return np.exp(-np.logaddexp(0, -x))

        # Loop over locations and calculate EEFs
        for i, (a, b) in enumerate(loc_slicers):
            loss_ab, loss_pred_ab = loss[a:b], loss_pred[a:b]
            rates_ab = rates[a:b]

            # Logistic function of 'distance matrix' of target and predicted
            dmat = k*(loss_targ[i][:,None] - loss_pred_ab)
            logistic = expit(dmat)

            # Calculate predicted EEFs, deltas and cost function gradient
            eefs_pred[i,:] = rates_ab.sum() - (rates_ab*logistic).sum(axis=1)
            deltas[i,:] = np.where(cost_mask[i], eefs_pred[i,:] - eefs_targ, 0)
            partial_i = rates_ab*loss_ab*logistic*(1-logistic)
            dg = 2*k*((wts[i]*deltas[i])[:,None]*partial_i).sum(axis=0)
            grad_cost[loceventixs[a:b]] += dg

        # Calculate cost function and gradient for current parameters
        cost = (wts * deltas**2).sum()
        return cost, grad_cost, deltas, eefs_pred
