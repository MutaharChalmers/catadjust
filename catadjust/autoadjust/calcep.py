#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import scipy.stats as st


def calcEP_ELT(elt, Nq=1, dist=None, method='oep'):
    """Calculate OEP or AEP curve from generic ELT.
    
    Parameters
    ----------
        elt : DataFrame
            Generic Event Loss Table (ELT) with minimum required columns
            EventID, MeanLoss, Rate. Columns needed for secondary uncertainty
            calculation include StdDevLoss and ExpValue.
        Nq : int, optional
            Number of child events (quantiles) each parent event is split into. 
            Defaults to 1 (expected mode).
        dist : string, optional
            Name of distribution used to model secondary uncertainty. Options
            are beta and lognormal. If None, uses expected mode.
        method : string, optional
            Calculate OEP (default) or AEP (not yet supported).

    Returns
    -------
        EP : DataFrame
            DataFrame of OEP or AEP losses and RPs, indexed by EventID.
        AAL : float
            Annual Average Loss.
    """

    # Set EventID as index
    elt = elt.set_index('EventID')

    if method.lower() == 'oep':
        # Expected mode
        if Nq == 1 or dist is None:
            print('Expected mode')
            cols = ['Loss','Rate']
            EP = elt.rename(columns={'MeanLoss': 'Loss'}
                            ).sort_values('Loss', ascending=False)[cols]
            EP['EEF'] = EP['Rate'].cumsum()
        # Distributed mode - beta and lognormal distributions only
        else:
            if 'beta' in dist.lower():
                # Calculate event CV
                elt['CV'] = elt['StdDevLoss']/elt['MeanLoss']

                # Fit beta distribution parameters
                elt['mu'] = elt['MeanLoss']/elt['ExpValue']
                elt['alpha'] = (1 - elt['mu'])/elt['CV']**2 - elt['mu']
                elt['beta'] = elt['alpha']*(1 - elt['mu'])/elt['mu']
                
                # Split ELT into "normal" and "weird" (small mu or CV events)
                norm_crit = (elt['mu']>=1e-6) & (elt['CV']>=1e-3)
                elt_normal = elt[norm_crit]
                elt_weird = elt[~norm_crit].rename(columns={'MeanLoss':'Loss'})

                # Scipy distribution object
                scipy_dist = st.beta(elt_normal['alpha'], elt_normal['beta'])
            elif 'logn' in dist.lower():
                # Calculate log-mean and log-standard deviations for scipy
                elt['CV'] = elt['StdDevLoss']/elt['MeanLoss']
                elt['sig'] = np.sqrt(np.log(1+elt['CV']**2))
                elt['mu'] = np.log(elt['MeanLoss']) - elt['sig']**2/2
                
                # Split ELT into "normal" and "weird" (small mu or CV events)
                norm_crit = (elt['MeanLoss']>=1e-6) & (elt['CV']>=1e-3)
                elt_normal = elt[norm_crit]
                elt_weird = elt[~norm_crit].rename(columns={'MeanLoss':'Loss'})

                # Scipy distribution object - use scipy's parametrisation
                scipy_dist = st.lognorm(s=elt_normal['sig'], 
                                        scale=np.exp(elt_normal['mu']))
            else:
                print('dist must be beta or logn.')
                return None, None
                
            # Generate equiprobable cumulative probabilities
            cumprobs_raw = np.linspace(0, 1, Nq+1)
            cumprobs = 0.5*(cumprobs_raw[1:] + cumprobs_raw[:-1])

            # Calculate stochastic OEP
            exp = elt_normal['ExpValue'].values if 'beta' in dist.lower() else 1

            # quantiles has shape (N_quantiles, N_events)
            quantiles = scipy_dist.ppf(cumprobs[:,None])*exp
            quantile_rates = elt_normal['Rate']/cumprobs.size
            EP = pd.DataFrame({'Loss': quantiles.T.ravel(), 
                               'Rate': np.repeat(quantile_rates, Nq)})
            if elt_weird.shape[0] > 0:
                loss_weird = elt_weird['Loss'].repeat(Nq)
                rate_weird = elt_weird['Rate'].repeat(Nq)/Nq
                EP_weird = pd.concat([loss_weird, rate_weird], axis=1)
                EP = pd.concat([EP, EP_weird])
            EP = EP.sort_values('Loss', ascending=False).dropna()
            EP['EEF'] = EP['Rate'].cumsum()
        
        # Calculate EPs/RPs
        EP['EP'] = 1 - np.exp(-EP['EEF'])
        EP['RP'] = 1/EP['EP']
        EP = EP.sort_values('RP')
    else:
        print('AEP calculation from ELT not yet supported')
        return None, None
    
    # Calculate AAL
    AAL = (elt['MeanLoss']*elt['Rate']).sum()
    return EP, AAL


def calcEP_YELT(yelt, year_range=(1, None), method='oep',
                pp='median', manual_RP=False):
    """Generate OEP or AEP curve from a historic or stochastic YELT.
    
    Parameters
    ----------
        yelt : DataFrame
            Year-Event Loss Table (YELT) including columns Year, Loss, [RP]. 
            Column RP is optional - only used if argument manual_RP is True.
        year_range : (int, int), optional
            Start and end year (inclusive) for filtering YELT. Defaults to
            (1, None) assuming years are 1-indexed and the largest year in the
            YELT is the total nominal number of years.
        method : string, optional
            Calculate OEP (default) or AEP.
        pp : string
            Plotting position estimation method. Must be 'weibull' or 'median'.
        manual_RP : boolean, optional
            Non-null values in column `RP` override calculated RPs.

    Returns
    -------
        EP : DataFrame
            Table of OEP or AEP losses and RPs, including columns EP_Weibull,
            EP_median, Loss, RP_Weibull, RP_median, RP, EEF, RP 95% ci lower,
            RP 95% ci upper.
        AAL : float
            Annual Average Loss.
    """
    
    # Unpack year_range tuple and make assumption on largest year if necessary
    year_start, year_end = year_range
    if year_end is None:
        year_end = yelt['Year'].max()

    # Generate historic year range to use
    N = year_end - year_start+1
    years = range(year_start, year_end+1)
    
    # Filter years of interest
    yelt_filt = yelt[(yelt['Year']>=year_start)&(yelt['Year']<=year_end)]
    
    # Mapping from year to index of largest event loss in that year
    # Used to map year to manual event RP
    year_maxeventix = yelt_filt.groupby('Year')['Loss'].idxmax()

    # Aggregate losses and ranks - reindex to handle years with no losses
    if method.lower() == 'oep':
        aggloss = yelt_filt.groupby('Year')['Loss'].max().reindex(years,
                                                                  fill_value=0)
    else:
        aggloss = yelt_filt.groupby('Year')['Loss'].sum().reindex(years,
                                                                  fill_value=0)
    ranks = aggloss.rank(method='max')

    # Calculate EPs according to Weibull and median plotting positions
    EP = pd.DataFrame({'rank': ranks,
                       'EP_Weibull': 1-ranks/(N+1),
                       'EP_Median': 1-st.beta.ppf(0.5, a=ranks, b=N+1-ranks),
                       'Loss': aggloss}
                       ).sort_values('EP_Weibull', ascending=False)
    EP['RP_Weibull'] = 1/EP['EP_Weibull']
    EP['RP_Median'] = 1/EP['EP_Median']
    
    # Manual RP override if applicable - only for OEP calculation
    if 'RP' in yelt.columns and manual_RP and method.lower()=='oep':
        # Leaves nulls where no RP specified - fill in below
        EP['manualRP'] = year_maxeventix.map(yelt_filt['RP'])
        EP['RP'] = year_maxeventix.map(yelt_filt['RP'])
        
        # Weibull plotting position
        if pp.lower() == 'weibull':
            EP['RP'] = EP['RP'].fillna(EP['RP_Weibull'])
        # Median plotting position
        else:
            EP['RP'] = EP['RP'].fillna(EP['RP_Median'])
    else:
        EP['RP'] = EP[f'RP_{pp.title()}']

    # Calculate Event Exceedance Frequency (EEF) only for OEP
    if method.lower() == 'oep':
        EP['EEF'] = -np.log(1-1/EP['RP'])
    
    AAL = yelt_filt['Loss'].sum()/N
    
    # Calculate confidence intervals on historic RPs and append to DataFrame
    rp_ci_df = rp_ci(np.arange(1, N+1), N, eef=False)
    rp_ci_df.index = EP.index
    return pd.concat([EP, rp_ci_df], axis=1), AAL

def rp_ci(i, n, ci_width=0.95, eef=False):
    """Estimate confidence intervals.
        
    Parameters
    ----------
        i : int or ndarray
            Rank(s) of largest annual losses in ascending order, i.e. rank 1
            is the smallest, rank 2 is the second smallest, etc.
        n : int
            Number of years.
        eef : boolean, optional
            Return confidence interval in EEF terms or not.

    Returns
    -------
        df : DataFrame
            Table with lower and upper confidence intervals on RP and EEF.
    """

    # Calculate lower and upper probability thresholds for confidence interval
    ci_lp = (1 - ci_width)/2
    ci_up = 1 - ci_lp

    # Define labels
    ci_lo_str = f'{ci_width*100:.0f}% ci lower'
    ci_hi_str = f'{ci_width*100:.0f}% ci upper'
    
    # Calculate confidence intervals and median using beta quantile function
    rp_ci_lower = 1/(1 - st.beta.ppf(ci_lp, a=i, b=n+1-i))
    rp_ci_upper = 1/(1 - st.beta.ppf(ci_up, a=i, b=n+1-i))
    
    if eef:
        # Calculate EEF equivalents
        eef_ci_lower = -np.log(1-1/rp_ci_lower)
        eef_ci_upper = -np.log(1-1/rp_ci_upper)
        return pd.DataFrame.from_dict({f'RP {ci_lo_str}': rp_ci_lower,
                                       f'RP {ci_hi_str}': rp_ci_upper,
                                       f'EEF {ci_lo_str}': eef_ci_lower,
                                       f'EEF {ci_hi_str}': eef_ci_upper})
    else:
        return pd.DataFrame.from_dict({f'RP {ci_lo_str}': rp_ci_lower,
                                       f'RP {ci_hi_str}': rp_ci_upper})
