#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from .optimisers import adam, adam_mb


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
        self.elt = elt.sort_values([loccol, refcol], ascending=[True, False])

        # Mapping from locationIDs to internal locids
        locations = self.elt[loccol].unique()
        locids = np.arange(locations.size, dtype=np.int64)
        self.locmap = dict(zip(locations, locids))
        self._locmap = pd.Series(self.locmap).sort_values()
        self.elt['_locid'] = self.elt[loccol].map(self.locmap)
        self.elt['eef'] = self.elt.groupby('_locid', sort=False
                                           )[ratecol].transform('cumsum')

        self.loccol = loccol
        self.eventcol = eventcol
        self.ratecol = ratecol
        self.refcol = refcol
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
        """Logistic sigmoid function."""
        return np.exp(-np.logaddexp(0, -x))

    def adjust(self, target, theta0=None, nepochs=100, ftol=1e-3, alpha=1e-3,
               beta1=0.9, beta2=0.999, relative=True, adj_bnds=(0, np.inf),
               wts=None, annealing='log', ks=(-1, 1), batch_size=0, seed=42):
        """Adjust ELT losses to match location-level loss EEF curves.

        Parameters
        ----------
        target : DataFrame
            Target losses in an (m locations, n target EEFs) DataFrame with
            EEFs as columns and the same locations in the index as the ELT.
        theta0 : Series or ndarray, optional
            Initial guess to use for loss adjustment.
        nepochs : int, optional
            Number of training epochs.
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
        adj_bnds : (float, float), optional
            Minimum and maximum adjustment bounds on loss factors.
        wts : ndarray, optional
            Weights to apply to each location-target loss. Should be the same
            shape as loss_targ. Locations are equally weighted by default.
        annealing : str, optional
            Annealing schedule. One of log, lin, cos.
        ks : (float, float), optional
            Log10 of initial and final annealing parameters.
        batch_size : int, optional
            Size of batch. <1 = batch; 1 = SGD; >1 = mini-batch.
        seed : int, optional
            Seed for random number generator used for SGD and mini-batch GD.

        Returns
        -------
        elt_adj : DataFrame
            Adjusted ELT.
        res : dict
            Results dict.
        """

        # Input validation
        if not isinstance(target, pd.DataFrame):
            print('target must be DataFrame')
            return None, None

        # Check that every location in the ELT has a corresponding row in target
        missing_targ_locs = set(self.locmap).symmetric_difference(target.index)
        if len(missing_targ_locs) > 0:
            locs_elt_not_target = list(set(self.locmap).difference(target.index))
            locs_target_not_elt = list(target.index.difference(self.locmap))
            if len(locs_elt_not_target) > 0:
                print(f'ELT locations missing in target: {locs_elt_not_target}')
            if len(locs_target_not_elt) > 0:
                print(f'Target locations missing in ELT: {locs_target_not_elt}')
            return None, None

        # Extract numpy arrays from input target DataFrame, sorting
        eefs = target.columns.to_numpy()
        targ = target.reindex(self._locmap.index).to_numpy()

        # Check that targ is increasing along axis 1
        if (targ[:,:-1] > targ[:,1:]).any() or (eefs[:-1] < eefs[1:]).any():
            print('targ values/columns must increase/decrease along axis 1')
            return None, None

        # Only count costs where EEF is valid and target losses are >=0
        cost_mask = (self.max_eefs >= eefs) & (targ >= 0)
        self.cost_mask = cost_mask

        # Best initial guess for loss scaling factors
        if theta0 is None:
            theta0 = np.ones(self.nevents)
        else:
            theta0 = np.array(theta0, dtype=np.float64)
        self.theta0 = theta0

        if wts is None:
            self.wts = np.ones_like(targ, dtype=np.float64)/np.prod(targ.shape)
        else:
            self.wts = np.array(wts, dtype=np.float64)/np.sum(np.array(wts))

        # Create RNG object for SGD and mini-batch SGD
        if batch_size > 0:
            nlocs = self.loc_slicers.shape[0]
            rng = np.random.default_rng(seed)
            stoc_args = {'nrecs': nlocs, 'rng': rng, 'batch_size': batch_size}

        # Create dict to pass arguments for the optimiser
        opt_args = {'alpha': alpha, 'beta1': beta1, 'beta2': beta2,
                    'nepochs': nepochs, 'ftol': ftol, 'amin': adj_bnds[0],
                    'amax': adj_bnds[1], 'k0': ks[0], 'k1': ks[1],
                    'annealing': annealing}

        if batch_size > 0:
            # TODO NOT IMPLEMENTED YET ======================================
            #optimise = adam_mb
            #opt_args = {**opt_args, **stoc_args}
            print('Minibatch not yet implemented - reverting to batch')
            optimise = adam
            # /TODO NOT IMPLEMENTED YET =====================================
        else:
            optimise = adam

        # Hard-code tol to 16
        cost_args = (targ, eefs, relative, cost_mask, 15)

        # Do the optimisation
        res = optimise(self.cost, theta0, cost_args, **opt_args)
        res['eventIDs'] = self.eventIDs

        # Post-processing of results
        event_ix = pd.Index(self.eventIDs, name=self.eventcol)
        self.theta = pd.Series(res['theta'], index=event_ix)

        # Create adjusted ELT DataFrame
        elt_adj = self.elt.copy()
        elt_adj[self.refcol] = res['theta'][self.loceventixs]*self.elt[self.refcol]
        elt_adj = elt_adj.sort_values([self.loccol, self.refcol],
                                      ascending=[True, False])
        elt_adj['eef'] = elt_adj.groupby('_locid', sort=False
                                         )[self.ratecol].transform('cumsum')
        elt_adj['rp'] = 1/(1-np.exp(-elt_adj['eef']))
        return elt_adj, res

    def cost(self, theta, loss_targ, eefs_targ, relative, cost_mask, tol=20,
             locs_mb=None, k=1.):
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
        relative : bool
            Relative or absolute error cost function.
        cost_mask : ndarray
            Boolean mask to use only delta values at location-EEF combinations
            where the target EEF is less than or equal to the largest EEF in
            the ELT at that location.
        tol : float, optional
            Tolerance used to mask out inputs to logistic function to speed up
            calculations on the distance matrix.
        locs_mb : ndarray, optional
            Indices of the locations in this mini-batch. If None, normal
            batch cost calculated. Not yet implemented.
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
            if relative:
                deltas[i,:] = np.where(cost_mask[i], eefs_pred[i,:]/eefs_targ-1, 0)
                partial_i = rates_ab*loss_ab*logistic*(1-logistic)/eefs_targ[:,None]
            else:
                deltas[i,:] = np.where(cost_mask[i], eefs_pred[i,:] - eefs_targ, 0)
                partial_i =  rates_ab*loss_ab*logistic*(1-logistic)

            dg = 2*k*((self.wts[i]*deltas[i])[:,None]*partial_i).sum(axis=0)
            grad_cost[self.loceventixs[a:b]] += dg

        # Calculate cost function and gradient for current parameters
        cost = (self.wts * deltas**2).sum()
        return cost, grad_cost, deltas, eefs_pred
