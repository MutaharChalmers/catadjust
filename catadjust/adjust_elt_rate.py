#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from .optimisers import adam, adam_mb


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
        self.rates_orig = self.elt.groupby(self.eventcol)[self.ratecol].mean()

        # Sorted array of unique eventIDs
        self.eventIDs = np.sort(self.elt[eventcol].unique())
        self.nevents = self.eventIDs.size

        # Convert eventIDs in ELT to indices in event array
        self.loceventixs = np.searchsorted(self.eventIDs, self.elt[eventcol])

        # Indices in ELT where location changes
        locbreaks = np.nonzero(np.diff(self.elt['_locid']))[0] + 1
        self.loc_slicers = np.hstack([np.r_[0, locbreaks][:,None],
                                      np.r_[locbreaks, m][:,None]])

    def adjust(self, target, theta0=None, nepochs=100, ftol=1e-3, alpha=1e-3,
               beta1=0.9, beta2=0.999, relative=True, adj_bnds=(1e-18, 1e3),
               wts=None, scale=False, log=False, batch_size=0, seed=42):
        """Adjust ELT rates to match location-level loss or hazard EEF curves.

        Parameters
        ----------
        target : DataFrame
            Target hazard or losses in an (m locations, n target EEFs) DataFrame
            with EEFs as columns and the same locations in the index as the ELT.
        theta0 : Series or ndarray, optional
            Initial guess to use for rate adjustment.
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
            Minimum and maximum adjustment bounds. If scale is true, these are
            limiting rate scaling factors, otherwise these are absolute limits
            on the values the rates can take.
        wts : ndarray, optional
            Weights to apply to each location-event. Array with the same shape
            as targ. By default, locations are equally weighted.
        scale : bool, optional
            Optimise by scaling rates or not.
        log : bool, optional
            Optimise in log-space or not.
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
        
        # Extract numpy arrays from input targ DataFrame
        eefs = target.columns.to_numpy()
        targ = target.reindex(self._locmap.index).to_numpy()
        
        # Check that targ is increasing along axis 1
        if (targ[:,:-1] > targ[:,1:]).any() or (eefs[:-1] < eefs[1:]).any():
            print('targ values/columns must increase/decrease along axis 1')
            return None, None

        # Interpolate input target EEFs to all rows of ELT
        eefs_targ = np.concatenate([np.interp(x[self.refcol], targ[i], eefs)
                                    for i, x in self.elt.groupby('_locid')])

        # Estimate target rates by location
        eefs_targ_by_loc = np.split(eefs_targ, self.loc_slicers[1:,0])
        rates_targ_by_loc = []
        for eefs_targ_loc in eefs_targ_by_loc:
            # Take differences between successive EEFs to estimate rates
            rates_targ_loc = np.diff(eefs_targ_loc)
            # np.diff on length n array returns n-1 values so add first rate
            if rates_targ_loc[0] > 0:
                r0 = eefs_targ_loc[0]
            else:
                r0 = 0.
            rates_targ_by_loc.append(np.r_[r0, rates_targ_loc])
        rates_targ_by_loc = np.concatenate(rates_targ_by_loc)
        rtl_df = pd.DataFrame({self.eventcol: self.elt[self.eventcol].values,
                               self.ratecol: rates_targ_by_loc}
                              ).replace({self.ratecol: {0: np.nan}})

        # Initial guess for adjusted rates based on mean location rate by event
        if theta0 is None:
            rates0 = rtl_df.groupby(self.eventcol)[self.ratecol].mean()
            if scale:
                theta0 = (rates0/self.rates_orig).fillna(np.spacing(1))
            else:
                theta0 = rates0.fillna(np.spacing(1))
            if log:
                theta0 = np.log(theta0)
        self.theta0 = np.array(theta0)

        # Default weights are uniform
        if wts is None:
            wts = np.ones_like(targ)
        else:
            wts = np.array(wts)

        # Interpolate wts into ELT wrt hazard/loss
        wts = [np.interp(x[self.refcol], targ[i], wts[i], left=0, right=0)
               for i, x in self.elt.groupby('_locid')]
        wts = np.concatenate(wts)
        self.wts = np.array(wts, dtype=np.float64)/np.sum(wts)

        # Create RNG object for SGD and mini-batch SGD
        if batch_size > 0:
            nlocs = self.loc_slicers.shape[0]
            rng = np.random.default_rng(seed)
            stoc_args = {'nrecs': nlocs, 'rng': rng, 'batch_size': batch_size}

        # Create dict to pass arguments for the optimiser
        opt_args = {'alpha': alpha, 'beta1': beta1, 'beta2': beta2,
                    'nepochs': nepochs, 'ftol': ftol, 'k0': 0., 'k1': 0.,
                    'amin': np.log(adj_bnds[0]) if log else adj_bnds[0], 
                    'amax': np.log(adj_bnds[1]) if log else adj_bnds[1]}

        if batch_size > 0:
            optimise = adam_mb
            opt_args = {**opt_args, **stoc_args}
        else:
            optimise = adam

        cost_args = (eefs_targ, relative, scale, log)

        # Do the optimisation, removing the unused annealing key-value pair
        res = optimise(self.cost, self.theta0, cost_args, **opt_args)
        event_ix = pd.Index(self.eventIDs, name=self.eventcol)
        self.theta = pd.Series(res['theta'], index=event_ix)

        if scale and log:
            res['rates'] = self.rates_orig * np.exp(res['theta'])
            self.rates = self.rates_orig * np.exp(self.theta)
        elif scale and not log:
            res['rates'] = self.rates_orig * res['theta']
            self.rates = self.rates_orig * self.theta
        elif not scale and log:
            res['rates'] = np.exp(res['theta'])
            self.rates = np.exp(self.theta)
        else:
            res['rates'] = res['theta']
            self.rates =  self.theta * 1
        res['eventIDs'] = self.eventIDs
        res['scale'] = scale
        res['log'] = log

        # Calculate cost by location over hazard curve
        tse = np.add.reduceat(res['deltas']**2, self.loc_slicers.ravel()[::2])
        n = np.diff(self.loc_slicers, axis=1).ravel()
        res['loc_mse'] = tse/n   
        res.pop('annealing')   

        # Create adjusted ELT DataFrame
        elt_adj = self.elt.copy()
        elt_adj[self.ratecol] = self.rates.values[self.loceventixs]
        elt_adj['eef'] = elt_adj.groupby('_locid', sort=False
                                         )[self.ratecol].transform('cumsum')
        elt_adj['rp'] = 1/(1-np.exp(-elt_adj['eef']))
        elt_adj['eef_targ'] = eefs_targ
        elt_adj['deltas'] = res['deltas']
        elt_adj['wts'] = self.wts      

        return elt_adj, res

    def cost(self, theta, eefs_targ, relative, scale, log, locs_mb=None, k=1.):
        """Cost function for fitting an ELT to a target EEF by adjusting
        event rates. Cost function handles relative or absolute errors, direct
        rate adjustment, or optimisation by scaling factors, log space or not,
        and batch or minibatch.

        Parameters
        ----------
        theta : ndarray
            Rates to calculate cost function for, in unique eventID order.
        eefs_targ : ndarray
            Target EEFs for location-events in the same order as the
            pre-processed ELT.
        relative : bool
            Relative or absolute error cost function.
        scale : bool
            Optimise by scaling rates or not.
        log : bool
            Optimise in log-space or not.
        locs_mb : ndarray, optional
            Indices of the locations in this mini-batch. If None, normal
            batch cost calculated.
        k : float, optional
            Annealing parameter - not used, kept for API consistency.

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
        eefs_pred = np.empty_like(eefs_targ)
        grad_cost = np.zeros_like(theta)

        # Define weights, handling minibatch case
        if locs_mb is None:
            wts = self.wts
        else:
            # Calculate mini-batch weights
            wts = np.zeros(self.wts.size, np.float64)
            r = np.full(self.wts.size, False, dtype=np.bool_)
            for a, b in self.loc_slicers[locs_mb]:
                r[a:b] = True
            wts[r] = self.wts[r]
            wts /= wts.sum()

        if log:
            theta = np.exp(theta)

        # Expand event rates to event-location rates
        if scale:
            rates = (theta*self.rates_orig.values)[self.loceventixs]
        else:
            rates = theta[self.loceventixs]

        # Calculate EEFs for each location by chunked cumulative sums
        for a, b in self.loc_slicers:
            eefs_pred[a:b] = rates[a:b].cumsum()

        # Calculate deltas and cost function for current parameters
        if relative:
            deltas = (eefs_pred/eefs_targ) - 1
        else:
            deltas = eefs_pred - eefs_targ
        cost = (wts * deltas**2).sum()

        # Calculate gradient of cost function wrt to event rates
        if scale and log:
            dbr = deltas * (theta*self.rates_orig.values)[self.loceventixs]
        elif scale and not log:
            dbr = deltas * self.rates_orig.values[self.loceventixs]
        elif not scale and log:
            dbr = deltas * self.rates_orig.values[self.loceventixs]
        else:
            dbr = deltas
        
        wts_eff = wts/eefs_targ if relative else wts
        
        for a, b in self.loc_slicers:
            dg = 2*(dbr[a:b]*wts_eff[a:b])[::-1].cumsum()[::-1]
            grad_cost[self.loceventixs[a:b]] += dg

        return cost, grad_cost, deltas, eefs_pred
