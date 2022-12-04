import os
import numpy as np
import pandas as pd
from scipy.constants import c
from misc import kewley
c_cm_s = c * 100

CALIFA_DATA = os.getenv('CALIFA_DATA',
                        '/home/espinosa/GoogleDrive/cespinosa/CALIFA/data/')

def get_data_emission_lines() -> pd.DataFrame:
    """
    Read emission lines table file
    """
    path_in = '/home/espinosa/GoogleDrive/cespinosa/CALIFA/data/'
    data = pd.read_pickle(path_in + 'HII_reg_catalog_cor.pickle')
    return data


def get_data_physical_properties() -> pd.DataFrame:
    """
    Read abundance table file
    """
    path_in = '/home/espinosa/data/CALIFA_DATA/tables/'
    data = pd.read_pickle(path_in + 'AbundanceTablev2_final.pkl')
    IZI_columns = data.columns[data.columns.str.startswith('q_IZI')]
    for col in IZI_columns:
        name = col[5:]
        data['U_IZI{}'.format(name)] = np.log10(10**(data[col])
                                                / c_cm_s)
    return data


def assign_Mass(galname, l_mass_a):
    """
    A simple function to return mass galaxy
    """
    mass = l_mass_a.loc[galname]
    return mass


def get_data_stellar_properties() -> pd.DataFrame:
    """
    Read stellar properties files
    """
    path_in = '/home/espinosa/GoogleDrive/cespinosa/CALIFA/data/'
    # SFH
    dataframe_sfh_extended = pd.read_pickle(
        path_in + 'df_reg_raw_sfh_eCALIFA.p')
    dataframe_sfh_extended = dataframe_sfh_extended[
        ~dataframe_sfh_extended.index.str.startswith('UGC12250')]
    dataframe_sfh_extended = dataframe_sfh_extended[
        ~dataframe_sfh_extended.index.str.startswith('UGC9837')]
    df_sfh_pilot = pd.read_pickle(path_in + 'df_reg_raw_sfh_pCALIFA.p')
    df_sfh_all = dataframe_sfh_extended.append(df_sfh_pilot, sort=False)
    # SSP
    dataframe_ssp_e = pd.read_pickle(
        path_in + 'df_reg_raw_ssp_eCALIFA.p')
    dataframe_ssp_e = dataframe_ssp_e[dataframe_ssp_e['GALNAME'] != 'UGC12250']
    dataframe_ssp_e = dataframe_ssp_e[dataframe_ssp_e['GALNAME'] != 'UGC9837']
    dataframe_ssp_p = pd.read_pickle(
        path_in + 'df_reg_raw_ssp_pCALIFA.p')
    df_ssp_all = dataframe_ssp_e.append(dataframe_ssp_p, sort=False)
    return df_sfh_all, df_ssp_all

def calc_incl(name, ELIP):
    """Get Inclination of the galaxy"""
    # print(f'name={name}\t ELIP={ELIP}')
    if np.isfinite(ELIP):
        if ELIP < 0.99:
            cos_incl_p2 = (1-ELIP**2-0.13**2)/(1-0.13**2)
            if cos_incl_p2 >= 0:
                median_incl = np.arccos(np.sqrt(cos_incl_p2))
                median_incl_deg = 180 * (median_incl/np.pi)
                return median_incl_deg
            else:
                print('Error on inclination calcuation for {}'.format(name))
                return np.nan
        else:
            return 0
    else:
        return 90

def get_data(only_hii=True) -> pd.DataFrame:
    """
    Get data function
    """
    data_emission_lines = get_data_emission_lines()
    data_properties = get_data_physical_properties()
    data_output = pd.concat([data_emission_lines, data_properties],
                            axis=1, sort=False)
    if only_hii:
        data_output = data_output[data_output['HIIregion'] == 1]
    data_sfh, df_ssp_all = get_data_stellar_properties()
    df_ssp_all.drop('GALNAME', axis='columns', inplace=True)
    data_output = pd.concat([data_output, df_ssp_all], axis=1, join='inner')
    sum_frac = data_sfh.xs(0.001, level='age', axis=1).sum(axis=1)
    mask_sfh = data_sfh.columns.levels[0] <= 0.3548
    frac_levels = data_sfh.columns.levels[0][mask_sfh]
    for age in frac_levels[1:]:
        sum_frac += data_sfh.xs(age, level='age', axis=1).sum(axis=1)
    #  Making index intersection
    data_output = data_output.loc[
        data_output.index.intersection(sum_frac.index)]
    sum_frac = sum_frac.loc[data_output.index.intersection(sum_frac.index)]
    data_output['f_y'] = sum_frac
    data_output['O3Hb_dd'] = data_output['OIII5006_cor'] / \
        data_output['Hb4861_cor']
    data_output['N2Ha_dd'] = data_output['NII6583_cor'] / \
        data_output['Ha6562_cor']
    data_output['S2Ha_dd'] = (data_output['SII6716_cor'] +
                              data_output['SII6730_cor']) / \
        data_output['Ha6562_cor']
    data_output['O1Ha_dd'] = data_output['OI6300_cor'] / \
        data_output['Ha6562_cor']
    data_output['O2Hb_dd'] = data_output['OII3727_cor'] / \
        data_output['Hb4861_cor']
    data_output = data_output[data_output['EWHa6562'] < 0]
    data_output['EW'] = np.log10(-data_output.loc[:, 'EWHa6562'])
    file_path = CALIFA_DATA + '/get_proc_elines_CALIFA_plus_PILOT.all_good.csv'
    geo_dt = pd.read_csv(file_path, comment='#', usecols=[0, 126, 130, 131],
                         names=['GALNAME', 'Re_arc', 'PA', 'ELIP'],
                         index_col=[0])
    geo_dt = geo_dt[~geo_dt.index.duplicated(keep='first')]
    tmp = pd.DataFrame({'GALNAME': ['CGCG229-009', 'CGCG97072b',
                                    'FGC1287a', 'sn2008gt'],
                        'Re_arc': [None, None, None, None],
                        'PA': [None, None, None, None],
                        'ELIP': [None, None, None, None]})
    tmp.set_index('GALNAME', inplace=True)
    geo_dt = pd.concat([geo_dt, tmp], ignore_index=False, axis=0)
    data_output['incl'] = data_output.apply(lambda row:
                                            calc_incl(str(row['GALNAME']),
                                                      geo_dt.loc[
                                                          str(row[
                                                              'GALNAME'])][
                                                                  'ELIP']),
                                            axis=1)
    if only_hii:
        data_output = data_output[data_output['GALNAME'] != 'CGCG229-009']
        data_output = data_output[data_output['GALNAME'] != 'CGCG97072b']
        data_output = data_output[data_output['GALNAME'] != 'FGC1287a']
        data_output = data_output[data_output['GALNAME'] != 'UGC12250']
        data_output = data_output[data_output['GALNAME'] != 'UGC9837']
        califa_data_path = os.getenv(
            'CALIFA_DATA', default='/home/espinosa/GoogleDrive/cespinosa/CALIFA/data')
        file_path = califa_data_path \
            + '/get_proc_elines_CALIFA_plus_PILOT.all_good.csv'
        log_mass = pd.read_csv(file_path, comment='#', usecols=[0, 1],
                               names=['GALNAME', 'log_Mass'], index_col=[0])
        data_output['l_Mass'] = data_output.apply(lambda row:
                                                  assign_Mass(row.GALNAME,
                                                              log_mass),
                                                  axis=1)
    # data_output = clean_emission_lines(data_output, 3)
    return data_output

def clean_emission_lines(data: pd.DataFrame, snr_limit: int) -> pd.DataFrame:
    """
    Clean data by emission line SNRs
    """
    mask_o3 = data['OIII5006_cor'] / data['eOIII5006_cor'] > snr_limit
    mask_n2 = data['NII6583_cor'] / data['eNII6583_cor'] > snr_limit
    mask_s2_16 = data['SII6716_cor'] / data['eSII6716_cor'] > snr_limit
    mask_s2_30 = data['SII6730_cor'] / data['eSII6730_cor'] > snr_limit
    # mask_o1 = data['OI6300_cor'] / data['eOI6300_cor'] > snr_limit
    mask_o2 = data['OII3727_cor'] / data['eOII3727_cor'] > snr_limit
    # mask_ha = data['Ha6562_cor'] / data['eHa6562_cor'] > snr_limit
    mask_hb = data['Hb4861_cor'] / data['eHb4861_cor'] > snr_limit
    mask = mask_o3 & mask_n2 & mask_s2_16 & mask_s2_30 & mask_o2 \
        & mask_hb
    print((~mask).sum())
    return data[mask]


def clean_data_diagnostic_diagrams(data: pd.DataFrame) -> pd.DataFrame:
    """
    Clean data function
    """
    mask_o3hb = (data['O3Hb_dd'] > 0) & (np.isfinite(data['O3Hb_dd']))
    mask_n2ha = (data['N2Ha_dd'] > 0) & (np.isfinite(data['N2Ha_dd']))
    mask_s2ha = (data['S2Ha_dd'] > 0) & (np.isfinite(data['S2Ha_dd']))
    mask_o1ha = (data['O1Ha_dd'] > 0) & (np.isfinite(data['O1Ha_dd']))
    mask_o2hb = (data['O2Hb_dd'] > 0) & (np.isfinite(data['O2Hb_dd']))
    print((~mask_o3hb).sum(), (~mask_n2ha).sum(), (~mask_s2ha).sum(),
          (~mask_o1ha).sum(), (~mask_o2hb).sum())
    mask = mask_o3hb & mask_n2ha & mask_s2ha & mask_o1ha & mask_o2hb
    #print(f'fracc: {(~mask).sum()}/{len(mask)}')
    # data = data[mask]
    # mask = data['Ha6562_cor']/data['eHa6562_cor'] > 10
    return data[mask]


def clean_data_kewley_curve(data: pd.DataFrame) -> pd.DataFrame:
    logN2Ha = np.log10(data['N2Ha_dd'])
    O3Hb_kewley = kewley(logN2Ha) + 0.1*kewley(logN2Ha)
    mask = np.log10(data['O3Hb_dd']) < O3Hb_kewley
    print((~mask).sum())
    return data[mask]


def clean_data_physical_properties(data: pd.DataFrame) -> pd.DataFrame:
    """
    Clean data by physical properties
    """
    labels = ['f_y', 'EW', 'U_Mor16_O23_ts', 'OH_Ho', 'NO_Pil16_Ho_R',
              'Ne_Oster_S', 'Av']
    #          , 'U_Mor16_O23_fs']
    # ,
    #           'OH_IZI_J_dop_k_20', 'OH_IZI_mean_dop_k_20', 'OH_IZI_max_dop_k_20',
    #           'OH_IZI_J_dop_k_inf', 'OH_IZI_mean_dop_k_inf', 'OH_IZI_max_dop_k_inf',
    #           'OH_IZI_J_levesque', 'OH_IZI_mean_levesque', 'OH_IZI_max_levesque',
    #           'OH_IZI_J_byler', 'OH_IZI_mean_byler', 'OH_IZI_max_byler',
    #           'OH_IZI_J_byler_CSFR', 'OH_IZI_mean_byler_CSFR',
    #           'OH_IZI_max_byler_CSFR']
    # 'U_Dors_O32', 'U_Dors_S', 'U_Mor16_O23_ts',
    # 'U_Mor16_S23', 'U_NB', 'U_HCm_no_interp', 'U_HCm_interp']
    mask = np.isfinite(data[labels[0]])
    masks = {label: np.isfinite(data[label]) for label in labels[1:]}
    for _, item in masks.items():
        mask = np.logical_and(mask, item)
    print((~mask).sum())
    # data = data[mask]
    # mask_lower = data['OH_Ho'] > 7.4  # min=7.5, min+0.1
    # mask_upper = data['OH_Ho'] < 8.9  # max=8.7, max-0.1
    # mask = mask_lower & mask_upper
    return data[mask]


def clean_data_physical_properties2(data: pd.DataFrame) -> pd.DataFrame:
    """
    Clean data by physical properties
    """
    labels = ['U_Dors_O32', 'U_Dors_S', 'U_Mor16_O23_fs',
              'U_NB', 'U_HCm_interp', 'U_IZI_mean_levesque']
    mask = np.isfinite(data[labels[0]])
    masks = {label: np.isfinite(data[label]) for label in labels[1:]}
    for _, item in masks.items():
        mask = np.logical_and(mask, item)
    # data = data[mask]
    # mask_lower = data['OH_Ho'] > 7.4  # min=7.5, min+0.1
    # mask_upper = data['OH_Ho'] < 8.9  # max=8.7, max-0.1
    # mask = mask_lower & mask_upper
    return data[mask]


def clean_data_dist(data: pd.DataFrame) -> pd.DataFrame:
    """
    Clean data by physical properties
    """
    mask = np.isfinite(data['dist'])
    return data[mask]


def clean_data_incl(data: pd.DataFrame, incl=70, incl_l=0) -> pd.DataFrame:
    """
    Clean data by physical properties
    """
    mask_h = data['incl'] < incl
    mask_l = data['incl'] > incl_l
    mask = mask_h & mask_l
    print((~mask).sum(), len(data[~mask].GALNAME.unique()))
    return data[mask]


def clean_data_physical_properties_complete(data: pd.DataFrame) -> pd.DataFrame:
    """
    Clean data by physical properties
    """
    mask_OH = data.columns.str.startswith('OH')
    mask_OH_IZI = data.columns.str.startswith('OH_IZI')
    labels = data.columns[mask_OH & ~mask_OH_IZI].values
    mask = np.isfinite(data[labels[0]])
    masks = {label: np.isfinite(data[label]) for label in labels[1:]}
    for _, item in masks.items():
        mask = np.logical_and(mask, item)
    # data = data[mask]
    # mask_lower = data['OH_Ho'] > 7.4  # min=7.5, min+0.1
    # mask_upper = data['OH_Ho'] < 8.9  # max=8.7, max-0.1
    # mask = mask_lower & mask_upper
    return data[mask]


def main() -> None:
  """
  Main function
  """
  data = get_data()
  data = clean_data_diagnostic_diagrams(data)
  data = clean_data_physical_properties(data)
  data = clean_data_kewley_curve(data)
  data = clean_data_incl(data, 70)
  osolar = 8.69
  data['OZ'] = (data['OH_Ho']-data['ZH_LW'])-osolar