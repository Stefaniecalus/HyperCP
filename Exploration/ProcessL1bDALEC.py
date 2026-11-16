''' Process L1AQC to L1B '''
import os
import re
import glob
import datetime as dt
from datetime import datetime
import numpy as np
import pandas as pd

from Source import PATH_TO_CONFIG
from Source import PACKAGE_DIR as CODE_HOME
from Source.ProcessL1b import ProcessL1b
# from Source.ProcessL1b_FactoryCal import ProcessL1b_FactoryCal
from Source.ProcessL1b_FRMCal import ProcessL1b_FRMCal
from Source.ConfigFile import ConfigFile
from Source.CalibrationFileReader import CalibrationFileReader
from Source.ProcessL1b_Interp import ProcessL1b_Interp
from Source.GetAnc import GetAnc
from Source.GetAnc_ecmwf import GetAnc_ecmwf
import Source.utils.loggingHCP as logging
import Source.utils.dating as dating
import Source.utils.filing as filing
from Source.utils.uncertainties import unc_management as um
from Source.utils.loggingHCP import writeLogFileAndPrint

class ProcessL1bDALEC:
    '''L1B for DALEC'''

    @staticmethod
    def read_unc_coefficient_factory(root):
        ''' SeaBird or TriOS'''
        # Read Uncertainties_new_char from provided files
        gp = root.addGroup("RAW_UNCERTAINTIES")
        gp.attributes['FrameType'] = 'NONE'  # add FrameType = None so grp passes a quality check later

        # Unc dataset renaming
        um.RenameUncertainties_Class(root)

        for sensor in ['LI','LT','ES']:
            dsname = sensor+'_RADCAL_UNC'
            gp.addDataset(dsname)
            ds = gp.getDataset(dsname)
            ds.columns["wvl"] = [400]
            ds.columns["unc"] = [0.0]
            ds.columnsToDataset()

        # interpolate unc to full wavelength range, depending on class based or full char
        um.interpUncertainties_Factory(root)

        # # generate temperature coefficient
       #Utilities.UncTempCorrection(root)

        return root

    # NOTE: These three methods could be combined into one.
    @staticmethod
    def processSensor(node,s):
        specialNames = {
            'ES':['Delta_T_Ed','Tempco_Ed','d0','d1','a0'],
            'LT':['Delta_T_Lu','Tempco_Lu','e0','e1','b0'],
            'LI':['Delta_T_Lsky','Tempco_Lsky','f0','f1','c0']}

        gp=node.getGroup(s)
        gpc=node.getGroup('CAL_COEF')
        delta_t=float(gpc.attributes[specialNames[s][0]])
        tref=float(gpc.attributes['Tref'])
        def0=float(gpc.attributes[specialNames[s][2]])
        def1=float(gpc.attributes[specialNames[s][3]])
        ds=gp.datasets[s]
        dc=gp.datasets['DARK_CNT'].data[s].tolist()
        inttime=gp.datasets['INTTIME'].data[s].tolist()
        temp=gp.datasets['SPECTEMP'].data['NONE'].tolist()
        cd=gpc.datasets[f'Cal_{s}']
        abc0 = cd.data[specialNames[s][4]].tolist()
        tempco=cd.data[specialNames[s][1]].tolist()

        # K1=d0*(V-DC)+d1
        # Ed=a0*((V-DC)/(Inttime+DeltaT_Ed)/K1)/(Tempco_Ed*(Temp-Tref)+1)

        for i in range(ds.data.shape[0]):
            c1=inttime[i]+delta_t
            for j in range(cd.data.shape[0]):
                ds.data[i][j] = 100.0*abc0[j]*((ds.data[i][j]-dc[i])/c1
                /(def1*(ds.data[i][j]-dc[i])+def0))/(tempco[j]*(temp[i]-tref)+1)

        # As in the calibration step above, this presumes UV bands (initial pixels) are always calibrated and
        #   any truncation in calibration is on the NIR end
        gp.attributes['CAL_START'] = str(0)
        gp.attributes['CAL_STOP'] = str(j)

        gp_l1aqc = node.getGroup(f'{s}_L1AQC')
        calDS = gp_l1aqc.addDataset('CAL_COEF')
        gp_l1aqc.datasets.move_to_end('CAL_COEF',last=False)
        calDS.data = cd.data
        calDS.datasetToColumns()
        gp_l1aqc.attributes['Tref'] = tref
        for key, attr in gp.attributes.items():
            if key in ['CAL_START','CAL_STOP']:
                gp_l1aqc.attributes[key] = attr
        for key in specialNames[s]:
            if key.lower().startswith('d') or key.lower().startswith('e') or key.lower().startswith('f'):
                gp_l1aqc.attributes[key] = gpc.attributes[key]
            elif key.startswith('Tempco'):
                gp_l1aqc.attributes[key] = tempco
            else:
                gp_l1aqc.attributes[key] = abc0

    # ======================================================================================================
    #  FRM (sensor-specific, "FRM-Full") calibration of the DALEC
    # ======================================================================================================
    # DALEC channel -> HyperCP sensor type mapping and the keys needed from the factory calibration file.
    # The spectrometer module serial numbers (Ed / Lu / Lsky) are read at L1A from the factory .cal file
    # and carried as root attributes, e.g. root.attributes['Ed_spectrometer_module'] = ' 159825'.
    FRM_CHANNELS = {
        'ES': {'channel': 'Ed',   'module_attr': 'Ed_spectrometer_module',   'deltaT': 'Delta_T_Ed',
               'cal': 'Cal_ES', 'a': 'a0', 'tempco': 'Tempco_Ed',   'lin': ('d0', 'd1')},
        'LT': {'channel': 'Lu',   'module_attr': 'Lu_spectrometer_module',   'deltaT': 'Delta_T_Lu',
               'cal': 'Cal_LT', 'a': 'b0', 'tempco': 'Tempco_Lu',   'lin': ('e0', 'e1')},
        'LI': {'channel': 'Lsky', 'module_attr': 'Lsky_spectrometer_module', 'deltaT': 'Delta_T_Lsky',
               'cal': 'Cal_LI', 'a': 'c0', 'tempco': 'Tempco_Lsky', 'lin': ('f0', 'f1')},
    }
    # FidRadDB cal/char file types per sensor. POLAR is read and stored in RAW_UNCERTAINTIES for the
    # uncertainty budget, but, as for SeaBird/TriOS, no polarisation correction is applied (and it is optional).
    FRM_FILES = {
        'ES': ['RADCAL', 'STRAY', 'THERMAL', 'ANGULAR'],
        'LI': ['RADCAL', 'STRAY', 'THERMAL', 'POLAR'],
        'LT': ['RADCAL', 'STRAY', 'THERMAL', 'POLAR'],
    }
    FRM_OPTIONAL_FILES = ('POLAR',)

    # The DALEC factory calibration carries an integration-time offset (Delta_T_*, ~4 ms) so the effective
    # exposure is INTTIME + Delta_T. When True the same effective time is used for the S1/S2/cal_int times of
    # the characterisation and for each measurement, so the offset is applied consistently on both sides.
    FRM_USE_INTTIME_OFFSET = True
    # Number of inner-band pixels excluded from the straylight (Zong et al. 2006) scattering distribution
    FRM_ZONG_N_IB = 3
    # Warn when the median FRM/Factory ratio (450-700 nm) is further than this from unity
    FRM_FACTORY_RATIO_TOL = 0.15

    # ------------------------------------------------------------------------------------------------------
    #  Small helpers
    # ------------------------------------------------------------------------------------------------------
    @staticmethod
    def _frm_table(ds):
        '''FidRadDB dataset (recarray with columns '0','1',...) -> 2-D float array.'''
        if ds is None:
            raise KeyError('Characterisation dataset not found')
        return np.asarray(pd.DataFrame(ds.data), dtype=float)

    @staticmethod
    def _frm_pixel_rows(tab, n_pix, what):
        '''Align a FidRadDB per-pixel table (RADCAL, THERMAL, ANGULAR, POLAR) with the DALEC pixels.

        These tables have one row per *detector pixel number* (first column, 0 ... 255 for a 256-pixel array), of which
        only some pixels carry data (the others have wavelength 0). DALEC output column k is detector pixel k, so row k
        of the result is the table row whose pixel number is k. No re-indexing of the valid rows and no interpolation:
        the wavelengths of the file coincide with the factory wavelengths of the same pixel (checked at calibration time).
        Pixels missing from the table are NaN. Returns an (n_pix x n_columns) array.'''
        pix = np.rint(tab[:, 0]).astype(int)
        sel = (pix >= 0) & (pix < n_pix)
        if sel.sum() == 0:
            raise ValueError(f'{what}: no row with pixel number in 0..{n_pix - 1}')
        out = np.full((n_pix, tab.shape[1]), np.nan)
        out[pix[sel]] = tab[sel]
        return out

    @staticmethod
    def get_module_serials(node):
        '''Spectrometer module serial numbers per sensor type, from the factory cal file attributes.'''
        serials = {}
        for sensor, info in ProcessL1bDALEC.FRM_CHANNELS.items():
            val = node.attributes.get(info['module_attr'])
            digits = re.findall(r'\d+', str(val)) if val is not None else []
            if not digits:
                raise ValueError(f"Root attribute '{info['module_attr']}' not found or has no serial number. "
                                 f"Was the DALEC factory .cal file read at L1A?")
            serials[sensor] = digits[0]
        return serials

    @staticmethod
    def _fidrad_dalec_dir():
        base = os.path.join(CODE_HOME, 'Data', 'FidRadDB')
        if os.path.isdir(base):
            for name in os.listdir(base):
                if name.lower() == 'dalec' and os.path.isdir(os.path.join(base, name)):
                    return os.path.join(base, name)
        raise FileNotFoundError(f'FidRadDB DALEC directory not found under {base}')

    @staticmethod
    def _select_char_file(fidrad_dir, module, charType, acq_time_seconds):
        '''Pick the CP_DAL_<dalec>_<module>_<TYPE>_<yyyymmddHHMMSS>.TXT file: most recent prior to acquisition,
        otherwise closest in time (the same fallback as read_FidRadDB_cal_char_files).'''
        files = np.array(sorted(glob.glob(os.path.join(fidrad_dir, f'CP_DAL_*_{module}_{charType}_*'))))
        if len(files) == 0:
            return None, None
        stamps = [os.path.basename(f).split('_')[-1].split('.')[0] for f in files]
        secs = np.array([datetime.strptime(t, '%Y%m%d%H%M%S').timestamp() for t in stamps])
        chosen, idx = ProcessL1b.choose_cal_char_per_time(acq_time_seconds, secs, files,
                                                          rule='most_recent_prior_acquisition')
        if chosen is None:
            chosen, idx = ProcessL1b.choose_cal_char_per_time(acq_time_seconds, secs, files, rule='closest_in_time')
            writeLogFileAndPrint(f'  No {charType} file prior to acquisition for module {module}; '
                                 f'using closest in time: {os.path.basename(chosen)}')
        return chosen, stamps[idx]

    @staticmethod
    def _read_char_for_sensor(gp, path, sensor):
        '''Read one FidRadDB file with filing.read_char and rename the datasets it created from
        <DEVICE>_<TYPE>_<DATA> to <ES|LI|LT>_<TYPE>_<DATA> (the DEVICE tag in the file is not relied upon).'''
        before = set(gp.datasets.keys())
        filing.read_char(path, gp)
        for name in [n for n in gp.datasets if n not in before]:
            m = re.search(r'(RADCAL|STRAYDATA|TEMPDATA|ANGDATA|POLDATA)_.*$', name)
            if m is None:
                raise ValueError(f'Unexpected dataset name {name} read from {os.path.basename(path)}')
            ds = gp.datasets[name]
            newDS = gp.addDataset(f'{sensor}_{m.group(0)}')
            if newDS is None:
                raise ValueError(f'Duplicate dataset {sensor}_{m.group(0)} reading {os.path.basename(path)}')
            newDS.copy(ds)
            newDS.datasetToColumns()
            gp.removeDataset(name)

    # ------------------------------------------------------------------------------------------------------
    #  Loader: FidRadDB cal/char files -> RAW_UNCERTAINTIES
    # ------------------------------------------------------------------------------------------------------
    @staticmethod
    def read_unc_coefficient_frm(node):
        '''Read the sensor-specific FidRadDB characterisation of the three DALEC spectrometer modules into a
        RAW_UNCERTAINTIES group, using the dataset naming of the SeaBird/TriOS FRM branch:
            <S>_RADCAL_CAL / _RADCAL_LAMP / _RADCAL_PANEL (L only), <S>_STRAYDATA_LSF, <S>_TEMPDATA_CAL,
            <S>_ANGDATA_COSERROR[_AZ90] (ES), <S>_POLDATA_CAL (LI, LT)            with S in ES, LI, LT.
        LAMP (and PANEL) are interpolated onto the DALEC pixel wavelengths, as interpUncertainties_FullChar
        does for the other platforms.'''
        try:
            serials = ProcessL1bDALEC.get_module_serials(node)
            fidrad_dir = ProcessL1bDALEC._fidrad_dalec_dir()
        except (ValueError, FileNotFoundError) as err:
            writeLogFileAndPrint(f'read_unc_coefficient_frm (DALEC): {err}')
            return None
        writeLogFileAndPrint(f'DALEC FRM: modules {serials}; FidRadDB: {fidrad_dir}')

        gp = node.addGroup('RAW_UNCERTAINTIES')
        if gp is None:
            gp = node.getGroup('RAW_UNCERTAINTIES')
        gp.attributes['FrameType'] = 'NONE'  # so the group passes a quality check later

        acq_time_seconds = node.getGroup('ES').datasets['DATETIME'].data[0].timestamp()

        for sensor in ['ES', 'LI', 'LT']:
            n_pix = len(node.getGroup(sensor).getDataset(sensor).data.dtype.names)
            for charType in ProcessL1bDALEC.FRM_FILES[sensor]:
                path, stamp = ProcessL1bDALEC._select_char_file(fidrad_dir, serials[sensor], charType,
                                                                acq_time_seconds)
                if path is None:
                    msg = f'Missing FidRadDB file CP_DAL_*_{serials[sensor]}_{charType}_*.TXT for {sensor} in {fidrad_dir}'
                    if charType in ProcessL1bDALEC.FRM_OPTIONAL_FILES:
                        writeLogFileAndPrint('  WARNING: ' + msg + ' (optional, only needed for the uncertainty budget)')
                        continue
                    writeLogFileAndPrint('  ERROR: ' + msg)
                    return None
                writeLogFileAndPrint(f'  {sensor} {charType}: {os.path.basename(path)}')
                try:
                    ProcessL1bDALEC._read_char_for_sensor(gp, path, sensor)
                except Exception as err:
                    writeLogFileAndPrint(f'  ERROR reading {os.path.basename(path)}: {err}')
                    return None
                gp.attributes[f'{sensor}_{charType}_file'] = os.path.basename(path)
                if charType == 'RADCAL':
                    l1aqc = node.getGroup(f'{sensor}_L1AQC')
                    if l1aqc is not None:
                        l1aqc.attributes['CalibrationDate'] = stamp

            # LAMP / PANEL onto the pixel wavelengths (pixel k of the table = DALEC column k)
            try:
                cal = ProcessL1bDALEC._frm_pixel_rows(ProcessL1bDALEC._frm_table(gp.getDataset(f'{sensor}_RADCAL_CAL')),
                                                      n_pix, f'{sensor}_RADCAL_CAL')
                wvl_pix = cal[:, 1]
                valid = np.isfinite(wvl_pix) & (wvl_pix > 0)
                for dtype in (['_RADCAL_LAMP'] if sensor == 'ES' else ['_RADCAL_LAMP', '_RADCAL_PANEL']):
                    ds = gp.getDataset(sensor + dtype)
                    ds.datasetToColumns()
                    x = np.asarray(ds.columns['0'], dtype=float)
                    outside = int(np.sum(valid & ((wvl_pix < x.min()) | (wvl_pix > x.max()))))
                    if outside:
                        writeLogFileAndPrint(f'  WARNING: {outside} {sensor} pixels lie outside the {dtype[1:]} wavelength range '
                                             f'({x.min():.0f}-{x.max():.0f} nm); the nearest tabulated value is used there.')
                    for idx in range(1, len(ds.columns)):
                        y = np.asarray(ds.columns[str(idx)], dtype=float)
                        ynew = np.full(n_pix, np.nan)
                        ynew[valid] = np.interp(wvl_pix[valid], x, y)
                        ds.columns[str(idx)] = ynew.tolist()
                    ds.columns['0'] = np.where(valid, wvl_pix, 0.0).tolist()
                    ds.columnsToDataset()
            except Exception as err:
                writeLogFileAndPrint(f'  ERROR preparing {sensor} lamp/panel data: {err}')
                return None
        return node

    # ------------------------------------------------------------------------------------------------------
    #  Cosine (angular) response correction for Ed
    # ------------------------------------------------------------------------------------------------------
    @staticmethod
    def cosine_error_correction_dalec(unc_grp, sensor, wvl_pix):
        '''As ProcessL1b_FRMCal.cosine_error_correction, but for any number of pixels (that function hard-codes 255).
        Returns (avg_coserror [n_pix x n_zenith, %], full_hemi_coserror [n_pix, %], zenith_ang [deg]).'''
        n_pix = len(wvl_pix)
        ds0 = unc_grp.getDataset(f'{sensor}_ANGDATA_COSERROR')
        ds90 = unc_grp.getDataset(f'{sensor}_ANGDATA_COSERROR_AZ90')
        zenith_ang = np.asarray([float(x) for x in ds0.attributes['COLUMN_NAMES'].split('\t')[2:]])
        ce0 = ProcessL1bDALEC._frm_pixel_rows(ProcessL1bDALEC._frm_table(ds0), n_pix, 'ANGULAR COSERROR')[:, 2:]
        if ds90 is not None:
            ce90 = ProcessL1bDALEC._frm_pixel_rows(ProcessL1bDALEC._frm_table(ds90), n_pix, 'ANGULAR COSERROR AZ90')[:, 2:]
            azi_avg = (ce0 + ce90) / 2.
        else:
            writeLogFileAndPrint('  Only one azimuth plane in ANGULAR file; using it alone.')
            azi_avg = ce0
        zen_avg = (azi_avg + azi_avg[:, ::-1]) / 2.   # average symmetric zenith angles

        # No correction outside 300-1000 nm (and none for pixels without wavelength)
        zen_avg[~np.isfinite(wvl_pix) | (wvl_pix < 300) | (wvl_pix > 1000), :] = 0

        # Full-hemispherical error, uniform sky:  integral of eps(theta) sin(2 theta) d theta
        zen0 = np.argmin(np.abs(zenith_ang))
        zen90 = np.argmin(np.abs(zenith_ang - 90))
        dZen = zenith_ang[1:] - zenith_ang[:-1]
        w = np.sin(2 * np.pi * zenith_ang[zen0:zen90] / 180) * dZen[zen0:zen90] * np.pi / 180
        full_hemi = np.nansum(zen_avg[:, zen0:zen90] * w[None, :], axis=1)
        return zen_avg, full_hemi, zenith_ang

    # ------------------------------------------------------------------------------------------------------
    #  Factory result, used only as a sanity check on the FRM result
    # ------------------------------------------------------------------------------------------------------
    @staticmethod
    def _factory_reference(node, sensor, raw, dark_cnt, int_time, spec_temp):
        info = ProcessL1bDALEC.FRM_CHANNELS[sensor]
        gpc = node.getGroup('CAL_COEF')
        cd = gpc.datasets[info['cal']]
        a0 = np.asarray(cd.data[info['a']], dtype=float)
        tempco = np.asarray(cd.data[info['tempco']], dtype=float)
        s0, s1 = [float(gpc.attributes[k]) for k in info['lin']]   # slope-like 'x0' and 'x1' (see .cal usage)
        dT = float(gpc.attributes[info['deltaT']])
        tref = float(gpc.attributes['Tref'])
        v = raw - dark_cnt[:, None]
        k = s1 * v + s0
        return 100.0 * a0[None, :] * (v / (int_time[:, None] + dT) / k) / (tempco[None, :] * (spec_temp[:, None] - tref) + 1)

    # ------------------------------------------------------------------------------------------------------
    #  FRM calibration
    # ------------------------------------------------------------------------------------------------------
    @staticmethod
    def processL1b_FRM(node):
        '''
        FRM sensor-specific calibration of ES, LI and LT, following the FRM4SOC-2 processing used for TriOS
        (ProcessL1bTriOS.processDarkCorrection_FRM) and SeaBird (ProcessL1b_FRMCal.processL1b_SeaBird).

        For every measurement, per pixel:
          1. dark            x = V - DARK_CNT                    (the DALEC reports one dark count per record)
          2. non-linearity   x = x (1 - alpha x)                 alpha from S1/S2 of the RADCAL file
          3. straylight      x = C x                             C = Zong matrix from the STRAY LSF
          4. integration     x = x (t_cal / t_meas)              (t + Delta_T when FRM_USE_INTTIME_OFFSET)
          5. absolute cal    x = x / gain                        gain from RADCAL: lamp (Ed) or lamp+panel (Lu, Lsky)
          6. thermal         x = x (1 + cT (T_spec - T_cal))     cT from THERMAL, T_spec from SPECTEMP per record
          7. cosine (Ed)     x = r x eps_dir + (1-r) x eps_hemi  r from the 6S direct ratio, eps from ANGULAR
        Pixels without calibration are dropped. Output is in uW/cm^2/nm (/sr for radiance), like the factory path.

        Requires node to contain: RAW_UNCERTAINTIES (read_unc_coefficient_frm), SIXS_MODEL, CAL_COEF.
        Returns True on success, False otherwise.
        '''
        unc = node.getGroup('RAW_UNCERTAINTIES')
        sixs = node.getGroup('SIXS_MODEL')
        gpc = node.getGroup('CAL_COEF')
        if unc is None or sixs is None or gpc is None:
            writeLogFileAndPrint('DALEC FRM: RAW_UNCERTAINTIES, SIXS_MODEL and CAL_COEF groups are required.')
            return False

        for sensor in ['ES', 'LI', 'LT']:
            info = ProcessL1bDALEC.FRM_CHANNELS[sensor]
            writeLogFileAndPrint(f'FRM Processing: {sensor} ({info["channel"]})')
            grp = node.getGroup(sensor)
            raw_ds = grp.getDataset(sensor)

            # ---------------- measurements ----------------
            col_names = raw_ds.data.dtype.names
            wvl_factory = np.asarray([float(c) for c in col_names])
            raw = np.array(raw_ds.data.tolist(), dtype=float)               # n_meas x n_pix
            n_meas, n_pix = raw.shape
            dark_cnt = np.asarray(grp.getDataset('DARK_CNT').data[sensor], dtype=float)
            int_time = np.asarray(grp.getDataset('INTTIME').data[sensor], dtype=float)
            spec_temp = np.asarray(grp.getDataset('SPECTEMP').data['NONE'], dtype=float)

            dT = float(gpc.attributes[info['deltaT']]) if ProcessL1bDALEC.FRM_USE_INTTIME_OFFSET else 0.0

            try:
                # ---------------- RADCAL ----------------
                # Table rows = detector pixel numbers 0..255; DALEC column k = pixel k. Pixel 0 carries the setup
                # (calibration integration time, t1, t2), is not a data pixel, and is excluded below.
                pf = ProcessL1bDALEC._frm_pixel_rows
                cal = pf(ProcessL1bDALEC._frm_table(unc.getDataset(f'{sensor}_RADCAL_CAL')), n_pix, f'{sensor}_RADCAL_CAL')
                if cal.shape[1] < 9:
                    raise ValueError(f'{sensor}_RADCAL_CAL has {cal.shape[1]} columns; 10 expected '
                                     f'(px, wl, resp, unc, dark1, dark2, S1, sdS1, S2, sdS2).')
                cal_int, t1, t2 = cal[0, 2], cal[0, 6], cal[0, 8]
                if not np.all(np.isfinite([cal_int, t1, t2])):
                    raise ValueError(f'{sensor}_RADCAL_CAL: no setup row (pixel 0 with calibration integration time, t1 and t2) found.')
                wvl_f, resp, S1, S2 = cal[:, 1].copy(), cal[:, 2], cal[:, 6], cal[:, 8]
                wvl_f[0] = 0.0
                lamp = ProcessL1bDALEC._frm_table(unc.getDataset(f'{sensor}_RADCAL_LAMP'))[:, 2]
                if sensor != 'ES':
                    panel = ProcessL1bDALEC._frm_table(unc.getDataset(f'{sensor}_RADCAL_PANEL'))[:, 2]

                # Pixels without a wavelength (unused pixels, and pixel 0 whose row holds the setup values) carry no data:
                # zero them, otherwise the setup numbers would give garbage S12/alpha that the straylight matrix spreads everywhere.
                good_w = np.isfinite(wvl_f) & (wvl_f > 0)
                S1, S2, resp = np.where(good_w, S1, 0.0), np.where(good_w, S2, 0.0), np.where(good_w, resp, 0.0)

                # Pixel alignment check: the file wavelength of pixel k must equal the factory wavelength of column k
                if not good_w.any():
                    raise ValueError('no pixel with a wavelength in the RADCAL table')
                dwvl = np.nanmax(np.abs(wvl_f[good_w] - wvl_factory[good_w]))
                if dwvl > 10:
                    raise ValueError(f'RADCAL and factory wavelengths differ by up to {dwvl:.1f} nm: pixel misalignment.')
                if dwvl > 2:
                    writeLogFileAndPrint(f'  WARNING: RADCAL and factory wavelengths differ by up to {dwvl:.1f} nm.')

                # ---------------- STRAYLIGHT ----------------
                lsf = ProcessL1bDALEC._frm_table(unc.getDataset(f'{sensor}_STRAYDATA_LSF'))
                if lsf.shape[0] < n_pix or lsf.shape[1] < n_pix:
                    raise ValueError(f'{sensor}_STRAYDATA_LSF is {lsf.shape}, smaller than {n_pix} x {n_pix}.')
                lsf = np.nan_to_num(lsf[:n_pix, :n_pix])        # row = source pixel, column = detector pixel (DALEC pixels only)
                n_empty = int(np.sum(good_w & (lsf.sum(axis=1) <= 0)))
                if n_empty:
                    writeLogFileAndPrint(f'  WARNING: {n_empty} calibrated pixels have an empty LSF row (no straylight correction for them).')
                C = ProcessL1b_FRMCal.Zong_SL_correction_matrix(lsf, ProcessL1bDALEC.FRM_ZONG_N_IB)

                # ---------------- THERMAL ----------------
                th_ds = unc.getDataset(f'{sensor}_TEMPDATA_CAL')
                cT = pf(ProcessL1bDALEC._frm_table(th_ds), n_pix, f'{sensor}_TEMPDATA_CAL')[:, 2]
                T_cal = th_ds.attributes.get('AMBIENT_TEMP', th_ds.attributes.get('REFERENCE_TEMP', None))
                if T_cal is None:
                    T_cal = float(gpc.attributes['Tref'])
                    writeLogFileAndPrint(f'  WARNING: no AMBIENT_TEMP in THERMAL file; using factory Tref = {T_cal} C')
                T_cal = float(T_cal)

                # ---------------- ANGULAR (Ed) ----------------
                if sensor == 'ES':
                    zen_avg, full_hemi, zenith_ang = ProcessL1bDALEC.cosine_error_correction_dalec(unc, sensor, wvl_f)
            except Exception as err:
                writeLogFileAndPrint(f'  ERROR in {sensor} characterisation: {err}')
                return False

            # ---------------- non-linearity, gain ----------------
            tce, t1e, t2e = cal_int + dT, t1 + dT, t2 + dT
            with np.errstate(divide='ignore', invalid='ignore'):
                if t2e == t1e:
                    writeLogFileAndPrint('  WARNING: S1 and S2 integration times are equal; non-linearity set to 0.')
                    S12 = S1.copy()
                else:
                    k = t1e / (t2e - t1e)
                    S12 = (1 + k) * S1 - k * S2
                S12_sl = np.nan_to_num(C @ np.nan_to_num(S12))           # straylight-corrected S12
                alpha = np.where(S12 ** 2 > 0, (S1 - S12) / S12 ** 2, 0.0)
                alpha[~np.isfinite(alpha)] = 0.0
                if sensor == 'ES':
                    gain = (S12_sl / lamp) * (10 * tce / t1e)
                else:
                    gain = (np.pi * S12_sl) / (lamp * panel) * (10 * tce / t1e)
            ok = np.isfinite(gain) & (gain > 0) & (resp > 0) & good_w & np.isfinite(cT)
            if sensor == 'ES':
                ok &= np.isfinite(zen_avg).all(axis=1) & np.isfinite(full_hemi)
            if not ok.any():
                writeLogFileAndPrint(f'  ERROR: no calibrated pixel found for {sensor}.')
                return False
            kk = np.where(ok)[0]
            writeLogFileAndPrint(f'  {ok.sum()} of {n_pix} pixels calibrated (columns {kk.min()}-{kk.max()}, '
                                 f'{wvl_f[kk.min()]:.1f}-{wvl_f[kk.max()]:.1f} nm; file-vs-factory wavelength max diff {dwvl:.3f} nm); '
                                 f'alpha in [{alpha[ok].min():.2e}, {alpha[ok].max():.2e}]')
            if ok.sum() < n_pix:
                writeLogFileAndPrint(f'  {n_pix - ok.sum()} DALEC columns have no characterisation and are dropped '
                                     f'(factory wavelength of the first / last dropped column: {wvl_factory[~ok][0]:.0f} / {wvl_factory[~ok][-1]:.0f} nm).')

            # ---------------- working temperature ----------------
            T_work = spec_temp.copy()
            bad = ~np.isfinite(T_work) | (T_work == 0)
            if bad.all():
                anc = node.getGroup('ANCILLARY_METADATA')
                if anc is not None and 'AIRTEMP' in anc.datasets:
                    air = np.asarray(anc.getDataset('AIRTEMP').columns['AIRTEMP'], dtype=float)
                    T_work[:] = np.nanmean(air) + 5.0
                    writeLogFileAndPrint('  WARNING: no internal temperature; using air temperature + 5 C.')
                else:
                    writeLogFileAndPrint('  ERROR: no working temperature available.')
                    return False
            elif bad.any():
                T_work[bad] = np.nanmedian(T_work[~bad])
                writeLogFileAndPrint(f'  WARNING: {bad.sum()} records without valid SPECTEMP set to the median.')

            # ---------------- measurement chain ----------------
            x = raw - dark_cnt[:, None]                                        # 1 dark
            x = x * (1 - alpha[None, :] * x)                                   # 2 non-linearity
            x = x @ C.T                                                        # 3 straylight (C x for each record)
            x = x * (tce / (int_time + dT))[:, None]                           # 4 integration time
            x = x / np.where(ok, gain, 1.0)[None, :]                           # 5 absolute calibration
            x = x * (1 + cT[None, :] * (T_work - T_cal)[:, None])              # 6 thermal

            if sensor == 'ES':                                                 # 7 cosine
                try:
                    direct = np.asarray(pd.DataFrame(sixs.getDataset('direct_ratio').data), dtype=float)
                    sza = np.asarray(sixs.getDataset('solar_zenith').data['solar_zenith'], dtype=float)
                except Exception as err:
                    writeLogFileAndPrint(f'  ERROR in cosine correction: {err}')
                    return False
                if direct.shape != x.shape:
                    writeLogFileAndPrint(f'  ERROR: 6S direct ratio {direct.shape} does not match Es data {x.shape}.')
                    return False
                idx = np.argmin(np.abs(zenith_ang[None, :] - sza[:, None]), axis=1)
                cos_corr = 1 - zen_avg[:, idx].T / 100.                        # n_meas x n_pix
                fh_corr = (1 - full_hemi / 100.)[None, :]
                x = direct * x * cos_corr + (1 - direct) * x * fh_corr

            # ---------------- sanity check against factory calibration ----------------
            try:
                fac = ProcessL1bDALEC._factory_reference(node, sensor, raw, dark_cnt, int_time, spec_temp)
                with np.errstate(divide='ignore', invalid='ignore'):
                    ratio = np.nanmedian(x / fac, axis=0)
                band = ok & (wvl_f >= 450) & (wvl_f <= 700)
                if band.any():
                    r = float(np.nanmedian(ratio[band]))
                    msg = f'  FRM/Factory median ratio 450-700 nm: {r:.3f}'
                    if abs(r - 1) > ProcessL1bDALEC.FRM_FACTORY_RATIO_TOL:
                        msg += '   <-- WARNING: large difference, check units/pixel alignment of the FidRadDB files'
                    writeLogFileAndPrint(msg)
                # sign convention of the thermal coefficients: HCP multiplies by (1 + cT dT), i.e. cT is a
                # *correction* coefficient, expected to be anti-correlated with the factory sensitivity Tempco.
                tc = np.asarray(gpc.datasets[info['cal']].data[info['tempco']], dtype=float)
                both = ok & np.isfinite(cT) & np.isfinite(tc)
                if both.sum() > 10 and np.std(cT[both]) > 0 and np.std(tc[both]) > 0:
                    rho = np.corrcoef(cT[both], tc[both])[0, 1]
                    if rho > 0.5:
                        writeLogFileAndPrint(f'  WARNING: THERMAL cT correlates +{rho:.2f} with factory Tempco_*; '
                                             f'check the sign convention of cT.')
            except Exception as err:
                writeLogFileAndPrint(f'  (factory sanity check skipped: {err})')

            # ---------------- write back (drop uncalibrated pixels) ----------------
            names = [str(w) for w in wvl_f[ok]]
            ds_dt = np.dtype({'names': names, 'formats': [np.float64] * len(names)})
            raw_ds.data = np.rec.fromarrays(x[:, ok].T, dtype=ds_dt)

            first, last = int(np.where(ok)[0].min()), int(np.where(ok)[0].max())
            for g in (grp, node.getGroup(f'{sensor}_L1AQC')):
                if g is not None:
                    g.attributes['CAL_START'] = str(first)
                    g.attributes['CAL_STOP'] = str(last)

            # keep SIXS_MODEL on the same wavelength labels as the calibrated Es
            if sensor == 'ES':
                for dsname in ['sixS_irradiance', 'direct_ratio', 'diffuse_ratio']:
                    ds = sixs.getDataset(dsname)
                    arr = np.asarray(pd.DataFrame(ds.data), dtype=float)[:, ok]
                    ds.data = np.rec.fromarrays(arr.T, dtype=ds_dt)

        return True

    @staticmethod
    def processL1b(node, outFilePath):
        '''
        Non-TriOS path. ProcessL1b_Interp.processL1b_Interp will be common to both platforms
        Apply dark shutter correction to light data. Then apply either default factory cals
        or full instrument characterization. Introduce uncertainty group.
        Match timestamps and interpolate wavebands.
        '''
        node.attributes["PROCESSING_LEVEL"] = "1B"
        now = dt.datetime.now()
        timestr = now.strftime("%d-%b-%Y %H:%M:%S")
        node.attributes["FILE_CREATION_TIME"] = timestr
        if ConfigFile.settings["fL1bCal"] == 1:
            node.attributes['CAL_TYPE'] = 'Factory'
        elif ConfigFile.settings["fL1bCal"] == 2:
            node.attributes['CAL_TYPE'] = 'FRM-Class'
        elif ConfigFile.settings["fL1bCal"] == 3:
            node.attributes['CAL_TYPE'] = 'FRM-Full'
        node.attributes['WAVE_INTERP'] = str(ConfigFile.settings['fL1bInterpInterval']) + ' nm'

        logging.writeLogFileAndPrint(f"ProcessL1bDALEC.processL1b: {timestr}")

        # Add a dataset to each group for DATETIME, as defined by TIMETAG2 and DATETAG
        node  = dating.rootAddDateTime(node)

        # Introduce a new group for carrying L1AQC data forward.
        esGroup = node.addGroup('ES_L1AQC')
        liGroup = node.addGroup('LI_L1AQC')
        ltGroup = node.addGroup('LT_L1AQC')

        for gp in node.groups:
            if gp.id == 'ES':
                esGroup.copy(gp)
            elif gp.id == 'LI':
                liGroup.copy(gp)
            elif gp.id == 'LT':
                ltGroup.copy(gp)

        ######### PLACEHOOLDER: when DALEC class-based and full-FRM become available #########
        # classbased_dir needed for FRM whilst pol is handled in class-based way
        # classbased_dir = os.path.join(PATH_TO_DATA, 'Class_Based_Characterizations', ConfigFile.settings['SensorType'] + "_initial")
        #
        # # The radCalDir is now the same for all cal/char regimes and regardless of whether files were downloaded from FidRadDB or not
        # radcal_dir = ConfigFile.settings['calibrationPath']
        #
        # # Add Class-based characterization files if needed (RAW_UNCERTAINTIES)
        # if ConfigFile.settings['fL1bCal'] == 1:
        #     print("Factory DALEC - no uncertainty computation")
        #
        # # Add Class-based characterization files + RADCAL files
        # elif ConfigFile.settings['fL1bCal'] == 2:
        #
        #     print("Class-Based - uncertainty computed from class-based and RADCAL")
        #     print('Class-Based:', classbased_dir)
        #     print('RADCAL:', radcal_dir)
        #     node = ProcessL1b.read_unc_coefficient_class(node, classbased_dir)
        #     if node is None:
        #         logging.writeLogFileAndPrint('Error running class based uncertainties.')
        #         return None
        #
        # # Or add Full characterization files (RAW_UNCERTAINTIES)
        # elif ConfigFile.settings['fL1bCal'] == 3:
        #
        #     node = ProcessL1b.read_unc_coefficient_frm(node)
        #     if node is None:
        #         logging.writeLogFileAndPrint('Error loading FRM characterization files. Check directory.')
        #         return None
        ############################################################################################################

        if not ProcessL1b_Interp.interp_Anc(node, outFilePath):
            logging.writeLogFileAndPrint('Error interpolating ancillary data')
            return None

        # Need to fill in with model data here. This had previously been run on the GPS group, but now shifted to Ancillary group
        ancGroup = node.getGroup("ANCILLARY_METADATA")
        # Retrieve MERRA2 model ancillary data
        if ConfigFile.settings["bL1bGetAnc"] ==1:
            logging.writeLogFileAndPrint('MERRA2 data for Wind and AOD may be used to replace blank values. Reading in model data...')
            modRoot = GetAnc.getAnc(ancGroup)
        # Retrieve ECMWF model ancillary data
        elif ConfigFile.settings["bL1bGetAnc"] == 2:
            logging.writeLogFileAndPrint('ECMWF data for Wind and AOD may be used to replace blank values. Reading in model data...')
            modRoot = GetAnc_ecmwf.getAnc_ecmwf(ancGroup)
        else:
            modRoot = None

        # if modRoot is not None:
        # Regardless of whether SolarTracker/pySAS is used, Ancillary data will have been already been
        # interpolated in L1B as long as the ancillary file was read in at L1AQC. Regardless, these need
        # to have model data and/or default values incorporated.

        # If GMAO modeled data is selected in ConfigWindow, and an ancillary field data file
        # is provided in Main Window, then use the model data to fill in gaps in the field
        # record. Otherwise, use the selected default values from ConfigWindow

        # This step is only necessary for the ancillary datasets that REQUIRE
        # either field or GMAO or GUI default values. The remaining ancillary data
        # are culled from datasets in groups in L1B
        ProcessL1b.includeModelDefaults(ancGroup, modRoot)

        # Calculate 6S model
        print('Running sixS')

        sensortype = "ES"
        # Irradiance direct and diffuse ratio
        res_sixS = ProcessL1b_FRMCal.get_direct_irradiance_ratio(node, sensortype)

        # Store sixS results in new group
        grp = node.getGroup(sensortype)
        solar_zenith = res_sixS['solar_zenith']
        # ProcessL1b_FRMCal.get_direct_irradiance_ratio uses Es bands to run 6S and then works around bands that
        #  don't have values from Tartu for full FRM. Here, use all the Es bands.
        direct_ratio = res_sixS['direct_ratio']
        diffuse_ratio = res_sixS['diffuse_ratio']
        # sixS model irradiance is in W/m^2/um, scale by 10 to match HCP units
        # model_irr = (res_sixS['direct_irr']+res_sixS['diffuse_irr']+res_sixS['env_irr'])[:,ind_raw_data]/10
        model_irr = (res_sixS['direct_irr']+res_sixS['diffuse_irr']+res_sixS['env_irr'])/10
        # model_irr = (res_sixS['direct_irr']+res_sixS['diffuse_irr']+res_sixS['env_irr'])[:,ind_nocal==False]/10

        sixS_grp = node.addGroup("SIXS_MODEL")
        for dsname in ["DATETAG", "TIMETAG2", "DATETIME"]:
            # copy datetime dataset for interp process
            ds = sixS_grp.addDataset(dsname)
            ds.data = grp.getDataset(dsname).data

        ds = sixS_grp.addDataset("sixS_irradiance")

        irr_grp = node.getGroup('ES_L1AQC')
        str_wvl = np.asarray(pd.DataFrame(irr_grp.getDataset(sensortype).data).columns)
        ds_dt = np.dtype({'names': str_wvl,'formats': [np.float64]*len(str_wvl)})
        rec_arr = np.rec.fromarrays(np.array(model_irr).transpose(), dtype=ds_dt)
        ds.data = rec_arr

        ds = sixS_grp.addDataset("direct_ratio")
        ds_dt = np.dtype({'names': str_wvl,'formats': [np.float64]*len(str_wvl)})
        rec_arr = np.rec.fromarrays(np.array(direct_ratio).transpose(), dtype=ds_dt)
        ds.data = rec_arr

        ds = sixS_grp.addDataset("diffuse_ratio")
        ds_dt = np.dtype({'names': str_wvl,'formats': [np.float64]*len(str_wvl)})
        rec_arr = np.rec.fromarrays(np.array(diffuse_ratio).transpose(), dtype=ds_dt)
        ds.data = rec_arr

        ds = sixS_grp.addDataset("solar_zenith")
        ds.columns["solar_zenith"] = solar_zenith
        ds.columnsToDataset()

        # Calibration
        #   Depending on the Configuration, process either the factory calibration, class-based characterization,
        #   or the complete instrument characterizations.
        if ConfigFile.settings['fL1bCal'] == 1:
            # TODO: Combine these three methods into one and pass the sensor type
            for sensor in ['ES','LI','LT']:
                ProcessL1bDALEC.processSensor(node,sensor)
                # ProcessL1bDALEC.processES(node)
                # ProcessL1bDALEC.processLT(node)
                # ProcessL1bDALEC.processLI(node)

        elif ConfigFile.settings['fL1bCal'] == 2:
            # NOTE: Class-based processing is not available for DALEC. Previously this branch fell through with
            #   uncalibrated counts labelled as 'FRM-Class'.
            writeLogFileAndPrint('FRM-Class processing for DALEC is not available. Use Factory or FRM-Full. Halting. *****')
            return None
        elif ConfigFile.settings['fL1bCal'] == 3:
            # FRM sensor-specific calibration. SIXS_MODEL (above) provides the direct/diffuse ratio for the cosine correction.
            node = ProcessL1bDALEC.read_unc_coefficient_frm(node)
            if node is None:
                writeLogFileAndPrint('Error loading FRM characterisation files for DALEC. Check the FidRadDB/Dalec directory.')
                return None
            if not ProcessL1bDALEC.processL1b_FRM(node):
                writeLogFileAndPrint('Error in ProcessL1bDALEC.processL1b_FRM')
                return None

        # Interpolation
        # Match instruments to a common timestamp (slowest shutter, should be Lt) and
        # interpolate to the chosen spectral resolution.
        node = ProcessL1b_Interp.processL1b_Interp(node, outFilePath)

        # Datetime format is not supported in HDF5; already removed in ProcessL1b_Interp.py
        node.attributes["LI_UNITS"] = 'uW/cm^2/nm/sr'
        node.attributes["LT_UNITS"] = 'uW/cm^2/nm/sr'
        node.attributes["ES_UNITS"] = 'uW/cm^2/nm'

        return node
