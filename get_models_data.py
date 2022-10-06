"""
Functions to get data from file or 3MdB
"""
# !/usr/bin/env python3

import os
import pymysql
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


def get_bond_data(datafile='data_BOND.csv', force=False):
    """
    Get Bond data from file or 3MdB
    """
    if not os.path.exists(datafile) or force:
        sel = """SELECT
                 log10(O__3_500684A/H__1_486133A) as O3,
                 log10(N__2_658345A/H__1_486133A) as N2,
                 log10(BLND_372700A/H__1_486133A) as O2,
                 log10(BLND_436300A/H__1_486133A) as O3_4363,
                 log10(BLND_575500A/H__1_486133A) as N2_5755,
                 log10(BLND_436300A/O__3_500684A) as rO3,
                 log10(BLND_575500A/N__2_658345A) as rN2,
                 log10((O__3_500684A + BLND_372700A)/H__1_486133A) as O32,
                 NITROGEN - OXYGEN as NO,
                 OXYGEN as O,
                 12+OXYGEN as OH,
                 logU_mean as logU,
                 log10(-Ha_EW) as EWHa,
                 SUBSTRING(com2, 5) as fr,
                 SUBSTRING(com3, 6) as age,
                 HbFrac,
                 A_OXYGEN_vol_1 as Op,
                 A_OXYGEN_vol_2 as Opp,
                 A_NITROGEN_vol_1 as Np,
                 A_NITROGEN_vol_2 as Npp,
                 log10(A_NITROGEN_vol_1/A_OXYGEN_vol_1) as Np_Op
                 FROM tab_17, abion_17
                 WHERE tab_17.ref like 'BOND_2'
                 AND tab_17.N = abion_17.N"""
        database = pymysql.connect(host=os.environ['MdB_HOST'],
                                   user=os.environ['MdB_USER'],
                                   passwd=os.environ['MdB_PASSWD'],
                                   database=os.environ['MdB_DB_17'])
        res = pd.read_sql(sel, con=database)
        res.fr = res.fr.astype('float64')
        res.age = res.age.astype('float64')/1e6
        database.close()
        res.to_csv(datafile, index=False)
    else:
        res = pd.read_csv(datafile)
    return res


def get_califa_data(datafile='data_CALIFA.csv', force=False):
    """
    Get CALIFA data from file or 3MdB
    """
    if not os.path.exists(datafile) or force:
        sel = """SELECT
                 log10(O__3_500684A/H__1_486133A) as O3,
                 log10(N__2_658345A/H__1_486133A) as N2,
                 log10(BLND_372700A/H__1_486133A) as O2,
                 log10(BLND_372700A/H__1_486133A) as O2,
                 log10(NE_3_386876A/H__1_486133A) as Ne3,
                 log10(AR_3_713579A/H__1_486133A) as Ar3,
                 log10(HE_1_587564A/H__1_486133A) as He1,
                 log10(BLND_436300A/H__1_486133A) as O3_4363,
                 log10(BLND_575500A/H__1_486133A) as N2_5755,
                 log10(BLND_436300A/O__3_500684A) as rO3,
                 log10(BLND_575500A/N__2_658345A) as rN2,
                 log10((O__3_500684A + BLND_372700A)/H__1_486133A) as O32,
                 NITROGEN - OXYGEN as NO,
                 OXYGEN as O,
                 12+OXYGEN as OH,
                 logU_mean as logU,
                 log10(-Ha_EW) as EWHa,
                 SUBSTRING(com2, 5) as fr,
                 SUBSTRING(com3, 6) as age,
                 HbFrac,
                 A_OXYGEN_vol_1 as Op,
                 A_OXYGEN_vol_2 as Opp,
                 A_NITROGEN_vol_1 as Np,
                 A_NITROGEN_vol_2 as Npp,
                 log10(A_NITROGEN_vol_1/A_OXYGEN_vol_1) as Np_Op
                 FROM tab_17, abion_17
                 WHERE tab_17.ref like 'CALIFA_2'
                 AND tab_17.N = abion_17.N"""
        database = pymysql.connect(host=os.environ['MdB_HOST'],
                                   user=os.environ['MdB_USER'],
                                   passwd=os.environ['MdB_PASSWD'],
                                   database=os.environ['MdB_DB_17'])
        res = pd.read_sql(sel, con=database)
        res.fr = res.fr.astype('float64')
        res.age = res.age.astype('float64')*1000
        database.close()
        res.to_csv(datafile, index=False)
    else:
        res = pd.read_csv(datafile)
    return res
