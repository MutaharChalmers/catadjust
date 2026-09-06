#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import scipy.special as ss
from tqdm.auto import tqdm
from .optimisers import adam, adam_mb


class RateAdjustment:
    """Adjust a catastrophe model location-level event loss table (ELT) or
    event hazard table (EHT) to match arbitrary target location-level loss or
    hazard EEF curves by scaling event rates.
    """
    def __init__(self, elt_raw, loccol, eventcol, ratecol, refcol,
                 locratewtcol=None):
        """Load raw location-level ELT/EHT and pre-process.

        Parameters
        ----------
        elt_raw : DataFrame
            Raw location-level ELT/EHT.
        loccol : str
            Name of column containing locationIDs.
        eventcol : str
            Name of column containing eventIDs.
        ratecol : str
            Name of column containing event rates.
        refcol : str
            Name of column containing event-location loss or hazard intensity.
        locratewtcol : str, optional
            Name of column containing location-rate weights, used for applying
            location-specific rate factors. These are used for example to adjust
            rates of earthquake events where the event "footprint" is all the
            locations where the epicentry *could* be, and the hazard intensity
            is a single eventwide parameter such as moment magnitude.
        """

        if locratewtcol is None:
            elt_raw = elt_raw.assign(locratewt=1)
            locratewtcol = 'locratewt'

        # Load ELT/EHT, convert datatypes, drop duplicates and sort
        elt = elt_raw.astype({loccol: str, eventcol: np.int64,
                              ratecol: np.float64, refcol: np.float64,
                              locratewtcol: np.float64}
                              ).drop_duplicates([loccol, eventcol]).dropna()
        self.elt = elt.sort_values([loccol, refcol], ascending=[True, False])

        # Mapping from locationIDs to internal locids
        self.locations = self.elt[loccol].unique()
        locids = np.arange(self.locations.size, dtype=np.int64)
        self.locmap = dict(zip(self.locations, locids))
        self._locmap = pd.Series(self.locmap).sort_values()
        self.elt['_locid'] = self.elt[loccol].map(self.locmap)
        self.elt['eef'] = self.elt.groupby('_locid', sort=False
                                           )[ratecol].transform('cumsum')

        self.loccol = loccol
        self.eventcol = eventcol
        self.ratecol = ratecol
        self.refcol = refcol
        self.locratewtcol = locratewtcol
        self.unc = False
        m = self.elt.shape[0]
        self.rates_orig = self.elt.groupby(self.eventcol)[self.ratecol].mean()
        self.locratewts = self.elt[locratewtcol].values

        # Sorted array of unique eventIDs
        self.eventIDs = np.sort(self.elt[eventcol].unique())
        self.nevents = self.eventIDs.size

        # Convert eventIDs in ELT/EHT to indices in event array
        self.loceventixs = np.searchsorted(self.eventIDs, self.elt[eventcol])

        # Indices in ELT/EHT where location changes
        locbreaks = np.nonzero(np.diff(self.elt['_locid']))[0] + 1
        self.loc_slicers = np.hstack([np.r_[0, locbreaks][:,None],
                                      np.r_[locbreaks, m][:,None]])

    def adjust(self, target, theta0=None, orig=True, nepochs=100, ftol=1e-3,
               alpha=1e-3, beta1=0.9, beta2=0.999, relative=True, mirror=True,
               adj_bnds=(1e-18, 1e3), wts=None, batch_size=0, seed=42):
        """Adjust rates to match location-level loss or hazard EEF curves.

        Parameters
        ----------
        target : DataFrame
            Target hazard or losses in an (m locations, n target EEFs) DataFrame
            with EEFs as columns and locations the index. The locations in the
            index must be exactly the same ones as in the ELT/EHT.
        theta0 : Series or ndarray, optional
            Initial guess to use for rate adjustment.
        orig : bool, optional
            Use original rates as initial values for adjustment optimisation.
            Defaults to True.
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
        mirror : bool, optional
            Use mirror (log) descent. Defaults to True.
        adj_bnds : (float, float), optional
            Minimum and maximum adjustment bounds. If scale is true, these are
            limiting rate scaling factors, otherwise these are absolute limits
            on the values the rates can take.
        wts : DataFrame, optional
            Weights to apply to each location-EEF. Array with the same shape
            as targ. By default, locations are equally weighted.
        batch_size : int, optional
            Size of batch. <1 = batch; 1 = SGD; >1 = mini-batch.
        seed : int, optional
            Seed for random number generator used for SGD and mini-batch GD.

        Returns
        -------
        elt_adj : DataFrame
            Adjusted ELT/EHT.
        res : dict
            Results dict.
        """

        # Input validation
        if not isinstance(target, pd.DataFrame):
            raise TypeError('target must be DataFrame')
    
        # Check that each location in the ELT/EHT corresponds to a row in target
        missing_targ_locs = set(self.locmap).symmetric_difference(target.index)
        if len(missing_targ_locs) > 0:
            locs_elt_not_target = list(set(self.locmap).difference(target.index))
            locs_target_not_elt = list(target.index.difference(self.locmap))
            if len(locs_elt_not_target) > 0:
                raise ValueError(f'ELT/EHT locations missing in target: {locs_elt_not_target}')
            if len(locs_target_not_elt) > 0:
                raise ValueError(f'Target locations missing in ELT/EHT: {locs_target_not_elt}')
        
        # Extract numpy arrays from input targ DataFrame
        eefs = target.columns.to_numpy()
        targ = target.reindex(self._locmap.index).to_numpy()
        
        # Check that targ is increasing along axis 1
        if (targ[:,:-1] > targ[:,1:]).any() or (eefs[:-1] < eefs[1:]).any():
            raise ValueError('targ values/columns must increase/decrease along axis 1')

        # Interpolate input target EEFs to all rows of ELT/EHT
        eefs_targ = np.concatenate([np.interp(x[self.refcol], targ[i], eefs)
                                    for i, x in self.elt.groupby('_locid')])

        # Initial guess for adjusted rates based on mean location rate by event
        if theta0 is None and orig:
            theta0 = self.rates_orig
        elif theta0 is None:
            eefs_targ_by_loc = np.split(eefs_targ, self.loc_slicers[1:,0])
            rates_targ_by_loc = []
            for eefs_targ_loc in eefs_targ_by_loc:
                # Take differences between successive EEFs to estimate rates
                # diff on length-n array gives n-1 values so prepend first rate
                rates_targ_loc = np.r_[eefs_targ_loc[0], np.diff(eefs_targ_loc)]
                rates_targ_by_loc.append(rates_targ_loc)
            rates_targ_by_loc = np.concatenate(rates_targ_by_loc)
            rtl = pd.DataFrame({self.eventcol: self.elt[self.eventcol].values,
                                self.ratecol: rates_targ_by_loc}
                               ).replace({self.ratecol: {0: np.nan}})
            rates0 = rtl.groupby(self.eventcol)[self.ratecol].mean()
            theta0 = rates0.fillna(np.spacing(1))
        else:
            pass
        self.theta0 = np.array(theta0)

        # Default weights are uniform
        if wts is None:
            wts = np.ones_like(targ)
        else:
            if isinstance(wts, pd.DataFrame):
                wts = wts.reindex(self._locmap.index).to_numpy()
            else:
                raise TypeError('wts must be a DataFrame')

        # Interpolate wts into ELT/EHT wrt hazard/loss
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
                    'nepochs': nepochs, 'mirror': mirror, 'ftol': ftol,
                    'k0': 0, 'k1': 0, 'amin': adj_bnds[0], 'amax': adj_bnds[1]}

        if batch_size > 0:
            optimise = adam_mb
            opt_args = {**opt_args, **stoc_args}
        else:
            optimise = adam

        cost_args = (eefs_targ, relative)

        # Do the optimisation, removing the unused annealing key-value pair
        res = optimise(self.cost, self.theta0, cost_args, **opt_args)
        event_ix = pd.Index(self.eventIDs, name=self.eventcol)
        self.rates = pd.Series(res['theta'], index=event_ix)
        res['rates'] = res['theta']
        res['eventIDs'] = self.eventIDs

        # Calculate cost by location over hazard curve
        tse = np.add.reduceat(res['deltas']**2, self.loc_slicers.ravel()[::2])
        n = np.diff(self.loc_slicers, axis=1).ravel()
        res['loc_mse'] = tse/n   
        res.pop('annealing')   

        # Create adjusted ELT/EHT DataFrame
        elt_adj = self.elt.copy()
        elt_adj[self.ratecol] = self.rates.values[self.loceventixs] * self.locratewts
        elt_adj['eef'] = elt_adj.groupby('_locid', sort=False
                                         )[self.ratecol].transform('cumsum')
        elt_adj['rp'] = 1/(1-np.exp(-elt_adj['eef']))
        elt_adj['eef_targ'] = eefs_targ
        elt_adj['deltas'] = res['deltas']
        elt_adj['wts'] = self.wts

        if self.unc:
            print('Generating EEFs accounting for uncertainty...')
            elt_adj['eef_unc'] = self.curves_with_uncertainty(elt_adj)
            elt_adj['rp_unc'] = 1/(1-np.exp(-elt_adj['eef_unc']))
            print('Complete.')

        return elt_adj, res

    def cost(self, theta, eefs_targ, relative, locs_mb=None, k=1.):
        """Cost function for fitting an ELT/EHT to a target EEF by adjusting
        event rates. Cost function handles relative or absolute errors, optional
        uncertainty in the hazard or loss variable, and batch or minibatch.

        Parameters
        ----------
        theta : ndarray
            Rates to calculate cost function for, in unique eventID order.
        eefs_targ : ndarray
            Target EEFs for location-events in the same order as the
            pre-processed ELT/EHT.
        relative : bool
            Relative or absolute error cost function.
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
        if not self.unc:
            eefs_pred = np.empty_like(eefs_targ)
            grad_cost = np.zeros_like(theta)
        else:
            eefs_node = np.empty((self.nn, eefs_targ.size))
            eefs_pred = np.empty((self.nn, eefs_targ.size))

            # Optimising cost(E[.]) - track grad(cost(E[.])])
            if self.meancurve:
                grad_cost = np.zeros_like(theta)
            # Optimising E[cost(.)] - track multiple 'realisations' of gradient
            else:
                grad_cost = np.zeros((self.nn, theta.size))

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

        # Expand event rates to event-location rates
        rates = theta[self.loceventixs] * self.locratewts

        # Calculate predicted EEFs for each location without uncertainty
        if not self.unc:
            for a, b in self.loc_slicers:
                eefs_pred[a:b] = rates[a:b].cumsum()
        # Calculate EEFs for each location with uncertainty
        else:
            for a, b in self.loc_slicers:
                eefs_node[:,a:b] = rates[a:b][self.sorter[:,a:b]].cumsum(axis=1)
                eefs_pred[:,a:b] = np.take_along_axis(eefs_node[:,a:b], 
                                                      self.indexer[:,a:b],
                                                      axis=1)

            # Calculate expected value of predicted EEFs over integration nodes
            if self.meancurve:
                eefs_pred = self.node_wts @ eefs_pred

        # Calculate deltas and cost function for current parameters
        if relative:
            deltas = (eefs_pred/eefs_targ) - 1
            wts_eff = wts/eefs_targ
        else:
            deltas = eefs_pred - eefs_targ
            wts_eff = wts
        cost = deltas**2 @ wts

        # Calculate gradient of cost function wrt to event rates
        if not self.unc or self.meancurve:
            for a, b in self.loc_slicers:
                dg = 2*(deltas[a:b]*wts_eff[a:b])[::-1].cumsum()[::-1]
                grad_cost[self.loceventixs[a:b]] += dg * self.locratewts[a:b]
        else:
            # Only for non-mean curve uncertainty calculations
            for a, b in self.loc_slicers:
                dg = 2*(deltas[:,a:b]*wts_eff[a:b])[:,::-1].cumsum(axis=1)[:,::-1]
                grad_cost[:,self.loceventixs[a:b]] += dg * self.locratewts[a:b]

            # Calculate expected values
            cost = self.node_wts @ cost
            grad_cost = self.node_wts @ grad_cost
            deltas = self.node_wts @ deltas
            eefs_pred = self.node_wts @ eefs_pred

        return cost, grad_cost, deltas, eefs_pred

    def add_uncertainty(self, dist, sigcol, method='ghq', nn=5, meancurve=True):
        """Adds uncertainty to the rate adjustment. Changes deterministic cost
        function into a stochastic one, and minimises cost(E[.]) or E[cost(.)]
        with respect to loss or hazard uncertainty distributions. Supported
        distributions are normal and lognormal. In order to add uncertainty, a
        column representing the (log-)standard deviations must be defined in the
        event table. Intended to be used for EQ model adjustments targeting mean
        seismic hazard curves produced using classical PSHA.

        Parameters
        ----------
        dist : str
            Distribution to used. Must be 'norm' or 'logn'.
        sigcol : str
            Column containing (log-)standard deviations by location-event. If
            dist is 'norm', this column is the standard deviation of the normal
            distribution with means defined by the reference column. If dist is
            'logn', this column is the log-standard deviation of the lognormal
            distribution with medians defined by the reference column.
        method : str, optional
            Method for handling uncertainty. Options are 'qmc' (quasi-Monte
            Carlo) and 'ghq' (Gauss-Hermite quadrature). Defaults to 'ghq'.
        nn : int, optional
            Number of samples or quadrature nodes to use. Defaults to 5.
        meancurve : boolean, optional
            If True (default), the optimiser minimises cost(E[x]), i.e. the
            target EEF curves are mean curves. If False, the optimiser minimises
            E[cost(x)], i.e. stochastic optimisation allowing variance in the
            target curve. This is relevant if targeting uncertain curves, e.g.
            curves derived from real data.
        """

        if dist.lower()[:4] == 'norm':
            self.dist = 'norm'
        elif dist.lower()[:4] == 'logn':
            self.dist = 'logn'
        else:
            raise ValueError('Only normal and lognormal uncertainty supported')

        if method.lower()[:3] == 'qmc':
            self.unc_method = 'qmc'
        elif method.lower()[:4] == 'ghq':
            self.unc_method = 'ghq'
        else:
            raise ValueError('method must be qmc or ghq')

        self.unc = True
        self.nn = nn
        self.meancurve = meancurve

        # If using quasi-Monte Carlo, define fixed node z-scores and weights
        if self.unc_method == 'qmc':
            cumprobs_raw = np.linspace(0, 1, nn+1)
            cumprobs = 0.5*(cumprobs_raw[1:] + cumprobs_raw[:-1])
            z = ss.ndtri(cumprobs)[:,None]
            self.node_wts = np.ones(nn)/nn
        # If using Gauss-Hermite, define fixed node z-scores and weights
        else:
            nodes_gh, weights_gh = np.polynomial.hermite.hermgauss(nn)
            # Convert numpy "physicist's" Gauss-Hermite to "probabilist's"
            z = nodes_gh[:,None] * np.sqrt(2)
            self.node_wts = weights_gh / np.sqrt(np.pi)
        
        # Convert node z-scores to node values
        if self.dist == 'norm':
            self.sig = self.elt[sigcol].values
            self.mu = self.elt[self.refcol].values
            nodes = (self.mu[None,:] + self.sig[None,:]*z)
        else: # self.dist == 'logn':
            self.sig = self.elt[sigcol].values
            self.mu = np.log(self.elt[self.refcol].values) - self.sig**2/2
            nodes = np.exp(self.mu[None,:] + self.sig[None,:]*z)

        # Create sorter and indexer for fast EEF lookup
        self.indexer = np.empty_like(nodes, dtype=np.int64)
        self.sorter = np.empty_like(nodes, dtype=np.int64)
        ref = self.elt[self.refcol].values

        for a, b in self.loc_slicers:
            # Fill sorter indices for nodes in descending order
            self.sorter[:,a:b] = np.argsort(nodes[:,a:b], axis=1)[:,::-1]

            for i in range(nn):
                # searchsorted for a sequence in descending order
                j = b-a - np.searchsorted(np.sort(nodes[i,a:b]), ref[a:b]) - 1
                self.indexer[i,a:b] = np.clip(j, 0, (b-a-1))

        # Add eef_unc column to input EHT/ELT
        self.elt['eef_unc'] = self.curves_with_uncertainty(self.elt)

    def curves_with_uncertainty(self, elt, ref_range=None, n=100, ztol=3):
        """Generate hazard or loss curves incorporating pre-defined uncertainty.
        This treats event rates like annual probabilities when converting event
        conditional EPs to event EPs, which is OK for rare events. To aggregate
        over all events to get the overall probability of x1 | x2 | ... | xN,
        summing probabilities works if events are mutually exclusive, a
        reasonable approximation for rare events.

        Parameters
        ----------
        elt : DataFrame
            ELT or EHT in standard format.
        ref_range : tuple, optional
            Range over which to calculate exceedance curves. If None, calculate
            this range automatically based on the reference value in ELT/EHT.
        n : int, optional
            Number of points to calculate exceedance curves over ref_range.
        ztol : float, optional
            Z-score threshold above which CEPs not calculated for efficiency.

        Returns
        -------
        curves : ndarray
            Array of EEF curves evaluated at reference values in ELT/EHT.
        """

        if not self.unc:
            raise ValueError('Uncertainty must be applied')

        m = self.locations.size
        ref = np.empty((m, n))
        if ref_range is None:
            for i in range(m):
                a, b = self.loc_slicers[i]
                ref_lo = elt[self.refcol].values[b-1]*0.9
                ref_hi = elt[self.refcol].values[a]*1.1
                ref[i] = np.linspace(ref_lo, ref_hi, n)
        else:
            ref[:] = np.linspace(*ref_range, n)

        if self.dist == 'logn':
            _ref = np.log(ref)
        else:
            _ref = ref

        # Get event rates by location
        rates = elt[self.ratecol].values

        # Generate the curves efficiently by only calculating material CEPs
        curves = np.empty(elt.shape[0])
        for i in tqdm(range(m)):
            a, b = self.loc_slicers[i]

            # Calculate z-scores
            z = (_ref[i][:,None] - self.mu[a:b])/self.sig[a:b]

            # Only calculate the CEP for z-scores less than ztol
            cep = np.zeros((n, b-a))
            j, k = np.where(z < ztol)
            cep[j,k] = 1 - ss.ndtr(z[j,k])

            # Calculate EPs and convert to EEFs
            eefs = -np.log(1 - cep@rates[a:b])
            curves[a:b] = np.interp(elt[self.refcol].values[a:b], ref[i], eefs)
        return curves

    def hazard_maps(self, elt, ref_col, eef_col, rps, extrap=True):
        """Make DataFrame of hazard maps interpolated to selected RPs.

        Parameters
        ----------
        elt : DataFrame
            ELT or EHT in standard format.
        ref_col : str
            Name of column containing event-location loss or hazard intensity.
        eef_col : str
            Name of column containing EEF values.
        RPs : ndarray
            Array of return periods at which to calculate the hazard.
        extrap : bool, optional
            If True, extrapolates out-of-bounds values crudely per np.interp,
            otherwise returns nan for them.

        Returns
        -------
        haz_maps : DataFrame
            Hazard maps.
        """

        eefs = -np.log(1 - 1/rps)

        haz_maps = {}
        for i, (a, b) in enumerate(self.loc_slicers):
            eef_loc = elt[eef_col].values[a:b]
            ref_loc = elt[ref_col].values[a:b]
            if extrap:
                haz_maps[i] = np.interp(eefs, eef_loc, ref_loc)
            else:
                haz_maps[i] = np.interp(eefs, eef_loc, ref_loc,
                                        left=np.nan, right=np.nan)
        return pd.DataFrame(haz_maps, index=pd.Index(rps, name='RP')).T
