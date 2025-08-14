#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import scipy.stats as st
from tqdm.auto import tqdm

from .calcep import calcEP_YELT, calcEP_ELT


def adjust_elt(elt_ref, yelt_obj, year_range, RPmax, pct_obj, niter=1, 
               method='oep', sesf_bounds=(1e-9,1e9), pp='median',
               manual_RP=False, Nq=50, dist=None):
    """Adjust a reference ELT to match a target EP curve from a objective YELT.

    The reference ELT is typically derived from a stochastic model and the
    objective YELT can either be derived from historic data, or another
    stochastic model. The target EP curve is calculated as the weighted sum of
    the EP curves from the reference ELT and the objective YELT.

    Parameters
    ----------
        elt_ref : DataFrame
            Reference ELT.
        yelt_obj : DataFrame
            Objective YELT.
        year_range : (int, int)
            Range of years to be used from the objective YELT.
        RPmax : float
            Largest RP targeted for adjustment in the reference ELT.
        pct_obj : (float, float)
            Percentages of objective EP to use from RP1 to RP{RPmax}.
            Weights are interpolated linearly with respect to RP. For example,
            (100, 0) is suitable for the "adjust stochastic ELT to match a
            historic YELT" use-case. In this case, 100% and 0% weights are given
            to the objective (historic) at RPs 1 and RPmax respectively.
        niter : int, optional
            Number of iterations to do the adjustment over. Defaults to 1, which
            replicates Automatic Adjustment Tool 'v1' behaviour. Higher values
            take longer to compute but should generally give a closer match.
        method : str, optional
            Whether to use OEP or AEP as a target. Currently only OEP supported.
        sesf_bounds : (float, float), optional
            Minimum and maximum values to cap SESFs at.
        pp : string
            Plotting position estimation method. Must be 'weibull' or 'median'.
        manual_RP : boolean, optional
            Non-null values in column `RP` override calculated RPs.
        Nq : int
            Number of quantiles used to represent secondary uncertainty.
            N_quantiles = 1 uses the mean only.
        dist : str, optional
            Distribution for secondary uncertainty. If an RMS ELT is detected,
            automatically uses a beta distribution.

    Returns
    -------
        elt_adj : DataFrame
            Adjusted ELT.
        EP_adj : DataFrame
            Adjusted EP losses.
        AAL_adj : float
            Adjusted AAL.
        EP_targ : DataFrame
            Target EP losses.
    """

    # Number of iterations; niter=1 equates to original behaviour
    n = max(1, int(niter)) + 1
    wts = np.linspace(0, 1, n)

    # Calculate stochastic EP curve and calculate SESFs from target losses
    # Check for RMS based on stochastic ELT columns
    if 'PERSPVALUE' in elt_ref.columns:
        print('RMS ELT detected')
        elt_adj = elt_ref.rename(columns={'RATE':'Rate',
                                          'EVENTID': 'EventID',
                                          'PERSPVALUE': 'MeanLoss',
                                          'EXPVALUE': 'ExpValue'})
        elt_adj['StdDevLoss'] = elt_adj['STDDEVI'] + elt_adj['STDDEVC']
        dist = 'beta'

        # Track original means and standard deviations for SESF calculations
        mean_ref = elt_adj['MeanLoss']
        stddev_ref = elt_adj['StdDevLoss']

    # EP of unadjusted ELT used to calculate ultimate target losses
    EP_adj, _ = calcEP_ELT(elt_adj, Nq, dist, method)
    loss_ref = EP_adj['Loss'].to_numpy(copy=True)
    RPs_ref = EP_adj['RP'].to_numpy(copy=True)

    # Calculate objective and target EP curves
    EP_obj, _ = calcEP_YELT(yelt_obj, year_range, method, pp, manual_RP)
    EP_targ = calc_target_losses(EP_obj, EP_adj, RPmax, pct_obj)
    loss_targ = EP_targ['Loss'].to_numpy(copy=True)

    # Multiple adjustment passes to incrementally adjust the ELT
    for i in tqdm(range(1, n)):
        # Interpolate losses to current RPs
        loss_ref_interp = np.interp(EP_adj['RP'], RPs_ref, loss_ref)
        loss_targ_interp = np.interp(EP_adj['RP'], RPs_ref, loss_targ)

        # Calculate intermediate target losses
        loss_targ_wt = pd.Series(wts[i]*loss_targ_interp +
                                 (1-wts[i])*loss_ref_interp, index=EP_adj.index)
        sesfs = fit_SESFs_ELT(loss_targ_wt, elt_adj, Nq, sesf_bounds, dist)

        # Adjust ELT
        elt_adj = elt_adj.merge(sesfs, left_on='EventID', right_index=True)
        elt_adj['MeanLoss'] = elt_adj['MeanLoss']*elt_adj['MeanSESFs']
        elt_adj['StdDevLoss'] = elt_adj['StdDevLoss']*elt_adj['StdDevSESFs']
        elt_adj = elt_adj.drop(['MeanSESFs','StdDevSESFs'], axis=1)
        
        # Calculate adjusted EP and AAL
        EP_adj, AAL_adj = calcEP_ELT(elt_adj, Nq, dist, method)

    elt_adj['SESFs_mean'] = elt_adj['MeanLoss']/mean_ref
    elt_adj['SESFs_stddev'] = elt_adj['StdDevLoss']/stddev_ref

    # Convert adjusted RMS ELT to original format
    if 'PERSPVALUE' in elt_ref.columns:
        elt_adj = elt_adj.rename(columns={'Rate':'RATE',
                                          'EventID': 'EVENTID' ,
                                          'MeanLoss': 'PERSPVALUE',
                                          'ExpValue': 'EXPVALUE'}
                                          ).drop('StdDevLoss', axis=1)
        elt_adj['STDDEVI'] = elt_adj['STDDEVI']*elt_adj['SESFs_stddev']
        elt_adj['STDDEVC'] = elt_adj['STDDEVC']*elt_adj['SESFs_stddev']

    return elt_adj, EP_adj, AAL_adj, EP_targ


def adjust_yelt(yelt_ref, yelt_obj, year_range_ref, year_range_obj, RPmax, 
                pct_obj, method='oep', sesf_bounds=(1e-9,1e9), pp='median', 
                manual_RP=False, eps=1e-12):
    """Adjust a reference YELT to match a target EP curve from a objective YELT.

    The reference YELT is typically derived from a stochastic model and the
    objective YELT can either be derived from historic data, or another
    stochastic model. The target EP curve is calculated as the weighted sum of
    the EP curves from the reference YELT and the objective YELT.

    Parameters
    ----------
        yelt_ref : DataFrame
            Reference YELT.
        yelt_obj : DataFrame
            Objective YELT.
        year_range_ref : (int, int)
            Range of years to be used from the reference YELT.
        year_range_obj : (int, int)
            Range of years to be used from the objective YELT.
        RPmax : float
            Largest RP targeted for adjustment in the reference YELT.
        pct_obj : (float, float)
            Percentages of objective EP to use from RP1 to RP{RPmax}.
            Weights are interpolated linearly with respect to RP. For example,
            (100, 0) is suitable for the "adjust stochastic ELT to match a
            historic YELT" use-case. In this case, 100% and 0% weights are given
            to the objective (historic) at RPs 1 and RPmax respectively.
        method : str, optional
            Whether to use OEP only, or AEP (+OEP) as a target.
        sesf_bounds : (float, float), optional
            Minimum and maximum values to cap SESFs at.
        pp : string
            Plotting position estimation method. Must be 'weibull' or 'median'.
        manual_RP : boolean, optional
            Non-null values in column `RP` override calculated RPs.

    Returns
    -------
        yelt_adj : DataFrame
            Adjusted YELT.
        EP_adj : DataFrame
            Adjusted EP losses.
        AAL_adj : float
            Adjusted AAL.
        EP_targ : DataFrame
            Target EP losses.
    """

    # Calculate reference, object and target EPs
    OEP_ref, _ = calcEP_YELT(yelt_ref, year_range_ref, 'oep', pp, manual_RP)
    OEP_obj, _ = calcEP_YELT(yelt_obj, year_range_obj, 'oep', pp, manual_RP)
    OEP_targ = calc_target_losses(OEP_obj, OEP_ref, RPmax, pct_obj)
    occ_targ = OEP_targ['Loss'].to_numpy()

    if method.lower() == 'aep':
        AEP_ref, _ = calcEP_YELT(yelt_ref, year_range_ref, 'aep', pp, manual_RP)
        AEP_obj, _ = calcEP_YELT(yelt_obj, year_range_obj, 'aep', pp, manual_RP)
        AEP_targ = calc_target_losses(AEP_obj, AEP_ref, RPmax, pct_obj)
        rem_targ = (AEP_targ['Loss'] - OEP_targ['Loss']).to_numpy()
        
        # Numpy array of target occurrence and remainders
        M_targ = np.stack([occ_targ, rem_targ])
    else:
        M_targ = np.atleast_2d(occ_targ)

    # Identify largest occurrence losses and remainders in reference YELT
    yelt_ref['rank'] = yelt_ref.groupby('Year')['Loss'].rank(method='first',
                                                             ascending=False
                                                             ).astype(int)

    yelt_wide = yelt_ref.set_index(['Year','rank'])['Loss'].unstack('rank')
    yelt_wide = yelt_wide.reindex(OEP_ref.index, fill_value=0)
    occ = yelt_wide[1].to_numpy()
    if method.lower() == 'aep':
        rem = yelt_wide[yelt_wide.columns[1:]].sum(axis=1).to_numpy()
        M = np.stack([occ, rem])
    else:
        M = np.atleast_2d(occ)
    
    sesfs_raw = M_targ/(M+eps)
    sesfs = np.empty(shape=yelt_wide.T.shape)
    sesfs[0] = sesfs_raw[0]
    sesfs[1:] = sesfs_raw[1] if method.lower() == 'aep' else sesfs_raw[0]
    sesfs = pd.DataFrame(sesfs.T, index=yelt_wide.index,
                         columns=yelt_wide.columns).stack().rename('sesfs')

    # Adjust reference YELT and calculate EP
    yelt_adj = yelt_ref.merge(sesfs, left_on=['Year','rank'], right_index=True)
    yelt_adj['Loss'] = yelt_adj['Loss']*yelt_adj['sesfs']
    EP_adj, AAL_adj = calcEP_YELT(yelt_adj, year_range_ref, method, pp, manual_RP)
    EP_targ = OEP_targ if method.lower() == 'oep' else AEP_targ
    return yelt_adj, EP_adj, AAL_adj, EP_targ


def calc_target_losses(EP_obj, EP_ref, RPmax, pct_obj):
    """Generate target loss EP curve as a weighted sum of objective and
    reference EP curves.

    Parameters
    ----------
        EP_obj : DataFrame
            Table of objective (e.g. historic) losses and RPs.
        EP_ref : DataFrame
            Table of reference (e.g. stochastic) losses and RPs.
        RPmax : int
            Maximum RP to use for objective contribution. Points with
            longer RP than this are 100% reference.
        pct_obj : (float, float)
            Percentages of objective EP to use from RP1 to RP{RPmax}.
            Weights are interpolated linearly with respect to RP. Defaults to
            (100, 0), which is suitable for the "adjust stochastic ELT to match
            historic YELT" use-case. In this case, 100% and 0% weights are
            given to the objective (historic) at RPs 1 and RPmax respectively.

    Returns
    -------
        targ : Series
            Series of target losses interpolated to the reference RPs.
    """

    # Interpolate objective losses and blending weights to reference RPs
    EP_obj_interp = np.interp(EP_ref['RP'], EP_obj['RP'], EP_obj['Loss'])
    w = np.interp(EP_ref['RP'], np.array([1, RPmax]), np.array(pct_obj)/100)
    
    # Weighted sum of objective and reference losses over all reference RPs
    targ = (w*EP_obj_interp + (1-w)*EP_ref['Loss']).sort_values().to_frame()
    targ['RP'] = EP_ref['RP'].to_numpy()
    return targ.reset_index(drop=True)


def q2m(quantiles, cumprobs):
    """Estimate mean and standard deviation from quantiles
    and associated cumulative probabilities for a general distribution.

    Parameters
    ----------
        quantiles : ndarray
            Array of loss quantiles.
        cumprobs : ndarray
            Array of corresponding cumulative probabilities.

    Returns
    -------
        d : dict
            Dictionary with keys mean and stddev, and corresponding values.
    """
    
    # Probability mass associated with each pair of successive quantiles
    # assuming each quantile is at the mid-point of an interval
    masses = np.diff(np.r_[0, (cumprobs[1:] + cumprobs[:-1])/2, 1])

    # Calculate first and second moments
    m1 = (masses * quantiles).sum()
    m2 = (masses * quantiles**2).sum()

    # Calculate standard deviation from first and second moments
    stddev = np.sqrt(np.clip(m2 - m1**2, 0, None))
    return {'MeanLoss': m1, 'StdDevLoss': stddev}


def q2m_logn(quantiles, cumprobs):
    """Estimate mean and standard deviation from quantiles and associated
    cumulative probabilities for a lognormal distribution.
    
    A specific method is used for the lognormal since it is unbounded to the
    right, and its moments are sensitive to the highest quantiles.
    Based on pages 2-3 here: https://www.johndcook.com/quantiles_parameters.pdf

    Parameters
    ----------
        quantiles : ndarray
            Array of loss quantiles.
        cumprobs : ndarray
            Array of corresponding cumulative probabilities.

    Returns
    -------
        d : dict
            Dictionary with keys mean and stddev and corresponding values.
    """

    if quantiles.size > 1:
        i, j = quantiles.size-1, quantiles.size-2
    
        # Apply John Cook's normal distribution formulae to logged quantiles
        log_quantiles = np.log(quantiles)
        log_sigma = ((log_quantiles[j] - log_quantiles[i])/
                     (st.norm.ppf(cumprobs[j]) - st.norm.ppf(cumprobs[i])))
        log_mu = ((log_quantiles[i]*st.norm.ppf(cumprobs[j]) - 
                   log_quantiles[j]*st.norm.ppf(cumprobs[i]))/
                  (st.norm.ppf(cumprobs[j]) - st.norm.ppf(cumprobs[i])))

        # Convert logged mean and standard deviation 
        mu = np.exp(log_mu + log_sigma**2/2)
        sigma = np.sqrt((np.exp(log_sigma**2)-1)*np.exp(2*log_mu+log_sigma**2))
    else:
        mu = quantiles[0]
        sigma = 0
    return {'MeanLoss': mu, 'StdDevLoss': sigma}


def fit_SESFs_ELT(tqs, elt_stoc, N_quantiles, sesf_bounds, dist=None):
    """Fit mean and standard deviation of stochastic eventIDs to match target
    losses, and calculate Stochastic Event Scaling Factors (SESFs).
    Works with N_quantiles=1, i.e. for expected mode calculation.

    Parameters
    ----------
        tqs : DataFrame
            Target loss quantiles with Loss and RP columns, indexed by EventID.
        elt_stoc : DataFrame
            Stochastic ELT.
        N_quantiles : int
            Number of quantiles used to represent secondary uncertainty.
            N_quantiles = 1 denotes using the mean value only.
        sesf_bounds : (float, float)
            Minimum and maximum values to cap SESFs at.
        dist : str, optional
            Distribution for secondary uncertainty. Generic approach used for
            bounded distributions like beta (default). If secondary uncertainty
            is unbounded (e.g. lognormal), use the specific distribtion option
            for better fit. Currently only lognormal supported.

    Returns
    -------
        sesfs : DataFrame
            Table of SESFs.
    """
    
    if dist is None:
        dist = ''
    
    if 'EventID' in elt_stoc.columns:
        elt_stoc = elt_stoc.set_index('EventID')

    # Generate cumulative probabilities for equiprobable quantiles
    cumprobs_raw = np.linspace(0, 1, N_quantiles+1)
    cumprobs = 0.5*(cumprobs_raw[1:] + cumprobs_raw[:-1])

    # Fit mean and standard deviation for each event given target quantiles
    if 'logn' in dist.lower():
        event_musigs = {eventID: q2m_logn(quantiles.to_numpy(), cumprobs) 
                        for eventID, quantiles in tqs.groupby(tqs.index)}
    else:
        event_musigs = {eventID: q2m(quantiles.to_numpy(), cumprobs) 
                        for eventID, quantiles in tqs.groupby(tqs.index)}
    event_musigs = pd.DataFrame.from_dict(event_musigs, orient='index'
                                          ).rename_axis(elt_stoc.index.name)
    
    # Calculate SESFs
    if N_quantiles == 1:
        cols = ['MeanLoss']
    else:
        cols = ['MeanLoss','StdDevLoss']
    sesfs = (event_musigs[cols]/elt_stoc[cols]
             ).rename(columns={'MeanLoss': 'MeanSESFs', 
                               'StdDevLoss': 'StdDevSESFs'}, errors='ignore')
    
    # Replace nans/infs by 1 and clip factors
    return sesfs.replace([np.inf, -np.inf, np.nan], 1).clip(*sesf_bounds)


def fit_SESFs_YELT():
    return None


def calc_deltas(EP_adj, EP_targ, min_loss=1):
    """Calculate deltas between adjusted EPs and target EP curves.

    Parameters
    ----------
        EP_adj : DataFrame
            Adjusted EP losses.
        EP_targ : DataFrame
            Target EP losses.
        min_loss : float, optional
            Minimum loss to consider - clip very small losses to this value
            to avoid large relative differences between very small numbers.

    Returns
    -------
        deltas : DataFrame
            Table of differences, absolute and relativ (%), by RP.
    """

    # Interpolate adjusted RP losses to target RP values for comparison    
    loss_adj_interp = np.interp(EP_targ['RP'], EP_adj['RP'], 
                              EP_adj['Loss'].clip(min_loss))
    loss_targ_clip = EP_targ['Loss'].clip(min_loss).to_numpy()
    delta_abs = loss_adj_interp - loss_targ_clip
    delta_rel = (loss_adj_interp /loss_targ_clip - 1) * 100
    deltas = pd.DataFrame({'RP': EP_targ['RP'].to_numpy(),
                           'delta_abs': delta_abs,
                           'delta_rel': delta_rel})
    return deltas


def RPtable(df, RPs=None, AAL=None):
    """Generate a table of OEP losses at specific RPs given full EP DataFrame.
    """

    if RPs is None:
        RPs = [2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000]
    if AAL is None:
        AAL = np.nan

    RPs = [RP for RP in RPs if RP<=df['RP'].max() and RP>=df['RP'].min()]
    df = df.sort_values('RP')
    out = pd.DataFrame({'RP': RPs + ['AAL'],
                        'Loss': np.r_[np.interp(RPs, df['RP'], df['Loss']), AAL]}
                        ).set_index('RP')['Loss']
    return out
