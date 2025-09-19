#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import pyodbc


def getELT_stoc(server, rdm_db, anlsid, perspcode, driver='SQL SERVER'):
    """Retrieve RMS RiskLink stochastic ELT from Aon SQL server.
    
    Parameters
    ----------
        server : string
            SQL Server with RDM.
        rdm_db : string
            Name of RDM.
        anlsid : int
            AnalysisID (ANLSID) required.
        perspcode : string
            Financial perspective code, e.g. GU, GR, CL.
        driver : string, optional
            SQL Server driver.

    Returns
    -------
        elt_stoc : DataFrame
            RMS ELT generated from RDM_PORT, RDM_ANLSEVENT and RDM_EVENTINFO
            with the following named columns: 
                ANLSID, PERSPCODE, PERSPVALUE, STDDEVC, STDDEVI, EXPVALUE, 
                RATE, TYPE, NAME, DESCRIPTION, ACTIVE
    """
    
    # Connect to server and RDM database
    cnxn = pyodbc.connect(f'DRIVER={driver};SERVER={server};DATABASE={rdm_db};'
                          'Trusted_connection=yes')
    curs = cnxn.cursor()
    
    # Retrieve ELT
    query = f"""SELECT a.ANLSID, a.EVENTID, a.PERSPCODE, a.PERSPVALUE,
                a.STDDEVC, a.STDDEVI, a.EXPVALUE, b.RATE, c.TYPE, c.NAME,
                c.DESCRIPTION, c.ACTIVE, d.NAME as ANLSNAME FROM RDM_PORT a 
                INNER JOIN RDM_ANLSEVENT b ON a.ANLSID=b.ANLSID
                AND a.EVENTID=b.EVENTID
                INNER JOIN RMS_EVENTINFO.dbo.event c ON a.EVENTID=c.ID
                INNER JOIN RDM_ANALYSIS d on a.ANLSID=d.ID
                WHERE a.PERSPCODE='{perspcode}' AND a.ANLSID='{anlsid}'
                AND c.ACTIVE=1"""
    curs.execute(query)
    columns = [column[0] for column in curs.description]
    elt_stoc = pd.DataFrame.from_records([rec for rec in curs], columns=columns)
    return elt_stoc

        
def getELT_ratescheme_stoc(server, rdm_db, anlsid, perspcode, rateschemeid,
                           driver='SQL SERVER'):
    """Retrieve RMS stochastic ELT from Aon SQL server using custom rates.

    Parameters
    ----------
        server : string
            SQL Server with RDM.
        rdm_db : string
            Name of RDM.
        anlsid : int
            AnalysisID (ANLSID) required.
        perspcode : string
            Financial perspective code, e.g. GU, GR, CL.
        rateschemeid : string
            Rate scheme ID.
        driver : string, optional
            SQL Server driver.

    Returns
    -------
        elt_stoc : DataFrame
            RMS ELT generated from RDM_PORT, RDM_ANLSEVENT and RDM_EVENTINFO
            with the following named columns: 
                ANLSID, PERSPCODE, PERSPVALUE, STDDEVC, STDDEVI, EXPVALUE, 
                RATE, TYPE, NAME, DESCRIPTION, ACTIVE
    """
    
    # Connect to server and RDM database
    cnxn = pyodbc.connect(f'DRIVER={driver};SERVER={server};DATABASE={rdm_db};'
                          'Trusted_connection=yes')
    curs = cnxn.cursor()

    # Retrieve ELT
    query = f"""SELECT a.ANLSID, a.ID, a.EVENTID, a.PERSPCODE, a.PERSPVALUE,
                a.STDDEVC, a.STDDEVI, a.EXPVALUE, e.RATE, c.TYPE, c.NAME,
                c.DESCRIPTION, c.ACTIVE, d.NAME as ANLSNAME FROM RDM_PORT a 
                INNER JOIN RDM_ANLSEVENT b ON a.ANLSID=b.ANLSID AND
                a.EVENTID=b.EVENTID
                INNER JOIN RMS_EVENTINFO.dbo.event c ON a.EVENTID=c.ID
                INNER JOIN RDM_ANALYSIS d on a.ANLSID=d.ID
                INNER JOIN RMS_EVENTINFO.dbo.eventrate e on a.EVENTID=e.EVENTID
                WHERE a.PERSPCODE='{perspcode}' AND a.ANLSID='{anlsid}'
                AND c.ACTIVE=1 AND e.RATESCHEMEID={rateschemeid}"""
    curs.execute(query)
    columns = [column[0] for column in curs.description]
    elt_stoc = pd.DataFrame.from_records([rec for rec in curs], columns=columns)
    cnxn.close()
    return elt_stoc
