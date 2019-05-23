import numpy as np
import pandas as pd
from scipy.signal import find_peaks

def cid_ce(x, normalize=True):
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    if normalize:
        s = np.std(x)
        if s!=0:
            x = (x - np.mean(x))/s
        else:
            return 0.0

    x = np.diff(x)
    return np.sqrt(np.dot(x, x))

def _get_length_sequences_where(x):
    if len(x) == 0:
        return [0]
    else:
        res = [len(list(group)) for value, group in itertools.groupby(x) if value == 1]
        return res if len(res) > 0 else [0]

def longest_strike_above_mean(x):
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    return np.max(_get_length_sequences_where(x >= np.mean(x)))/x.size if x.size > 0 else 0


def make_peak_features(x, alpha=1, t=150, debug=False):
    if np.mean(x['flux'].values < -1) >= 0.51:
        if debug:
            print('Scaled')
        x = x.copy()
        med = x['flux'].median()
        x.loc[:, 'flux'] -= med
        
    _mean = np.mean(x['flux'].values)
    _max = np.max(x['flux'].values)
    _size = x['flux'].shape[0]

    d = _size // 3 if _size // 3 > 1 else 2
    arr_vh, d_vh = find_peaks(x=np.concatenate([[_mean], x['flux'].values, [_mean]], axis=0), distance=d, 
                              height=(_max / 2, None))
    
    d = _size // 6 if _size // 6 > 1 else 2
    arr_h, d_h = find_peaks(x=np.concatenate([[_mean], x['flux'].values, [_mean]], axis=0), distance=d, 
                            height=(_mean, None)) 
    
    temp = []
    for i in arr_vh:
        if x['flux'].iloc[i-1] > (alpha + 1.5) * x['flux_err'].iloc[i-1]:
            temp.append(i) 
    arr_vh = np.array(temp)

    
    temp = []
    for i in arr_h:
        if x['flux'].iloc[i-1] > (alpha + 0.35) * x['flux_err'].iloc[i-1]:
            temp.append(i) 
    arr_h = np.array(temp)


    lhw, rhw, lw, rw, peak_time, lrd, rrd = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    if len(arr_vh) == 1:
        next_value, next_time, prev_value, prev_time = np.nan, np.nan, np.nan, np.nan
        i = arr_vh[0]
        peak_value, peak_time = x[['flux', 'mjd']].iloc[i-1]
        # allmost complete height of peak with correction of mean flux
        full_height = peak_value / 4
        
        while i < _size:
            next_value, next_time = x[['flux', 'mjd']].iloc[i]
            if  next_time - peak_time > t:
                # too big gap
                break
            if  next_value >= peak_value / 2:
                # make candidate
                rhw = next_time - peak_time
                rrd = (peak_value - next_value) / (next_time - peak_time + 0.001)
            elif next_value < full_height:
                rw = next_time - peak_time
                rrd = (peak_value - next_value) / (next_time - peak_time + 0.001)
                if rhw is np.nan:
                    rhw = next_time - peak_time
                break
            elif next_value > full_height:
                rw = next_time - peak_time
                rrd = (peak_value - next_value) / (next_time - peak_time + 0.001)
                if rhw is np.nan:
                    rhw = next_time - peak_time
            i += 1

        i = arr_vh[0]
        i -= 2
        while i >= 0:
            prev_value, prev_time = x[['flux', 'mjd']].iloc[i]
            if  peak_time - prev_time > t:
                # too big gap
                break      
            if  prev_value >= peak_value / 2:
                # make candidate
                lhw = peak_time - prev_time
                lrd = (peak_value - prev_value) / (peak_time - prev_time + 0.001) 
            elif prev_value < full_height:
                lw = peak_time - prev_time
                lrd = (peak_value - prev_value) / (peak_time - prev_time + 0.001) 
                if lhw is np.nan:
                    lhw = peak_time - prev_time
                break
            elif prev_value > full_height:
                lw = peak_time - prev_time     
                lrd = (peak_value - prev_value) / (peak_time - prev_time + 0.001) 
                if lhw is np.nan:
                    lhw = peak_time - prev_time
            i -= 1
            
    period = np.nan
    if len(arr_h) <= 5 and len(arr_h) > 1:
        values = []
        for i in arr_h:
            values.append(x['mjd'].iloc[i-1])
            
        lenght = len(arr_h)
        period = 0    
        count = 0
        for i in range(5):
            for j in range(0, lenght-1-i):
                count += 1
                period += (values[-1-j] - values[-2-j-i]) / (i+1)
        period = period / count
        
    if debug:
        return arr_vh, arr_h, lhw, rhw, lw, rw, lrd, rrd, peak_time, period
    else:
        return len(arr_vh), len(arr_h), lhw, rhw, lw, rw, lrd, rrd, peak_time, period
    
def make_peak_features2(x, alpha=1, t=150, debug=False, col='mag'):
        
    _mean = np.mean(x[col].values)
    _flux_median = np.median(x['flux'].values)
    _max = np.max(x[col].values)
    _size = x[col].shape[0]

    d = _size // 2 if _size // 2 > 1 else 2
    arr_vh, d_vh = find_peaks(x=np.concatenate([[_mean], x[col].values, [_mean]], axis=0), distance=d, 
                              height=(_mean, None))
    
    d = _size // 10 if _size // 10 > 1 else 2
    arr_h, d_h = find_peaks(x=np.concatenate([[_mean], x[col].values, [_mean]], axis=0), distance=d, 
                            height=(_mean, None)) 
    
    temp = []
    for i in arr_vh:
        if x['flux'].iloc[i-1] - _flux_median > (alpha + 1.5) * x['flux_err'].iloc[i-1]:
            temp.append(i) 
    arr_vh = np.array(temp)

    
    temp = []
    for i in arr_h:
        if x['flux'].iloc[i-1] - _flux_median > (alpha + 0.35) * x['flux_err'].iloc[i-1]:
            temp.append(i) 
    arr_h = np.array(temp)


    lhw, rhw, lw, rw, peak_time, lrd, rrd = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    if len(arr_vh) == 1:
        next_value, next_time, prev_value, prev_time = np.nan, np.nan, np.nan, np.nan
        i = arr_vh[0]
        peak_value, peak_time = x[[col, 'mjd']].iloc[i-1]
        peak_value -= _mean

        full_height = peak_value / 4
        
        while i < _size:
            next_value, next_time = x[[col, 'mjd']].iloc[i]
            next_value -= _mean
            if  next_time - peak_time > t:
                # too big gap
                break
            if  next_value >= peak_value / 2:
                # make candidate
                rhw = next_time - peak_time
                rrd = (peak_value - next_value) / (next_time - peak_time + 0.001)
            elif next_value < full_height:
                rw = next_time - peak_time
                rrd = (peak_value - next_value) / (next_time - peak_time + 0.001)
                if rhw is np.nan:
                    rhw = next_time - peak_time
                break
            elif next_value > full_height:
                rw = next_time - peak_time
                rrd = (peak_value - next_value) / (next_time - peak_time + 0.001)
                if rhw is np.nan:
                    rhw = next_time - peak_time
            i += 1

        i = arr_vh[0]
        i -= 2
        while i >= 0:
            prev_value, prev_time = x[[col, 'mjd']].iloc[i]
            prev_value -= _mean
            if  peak_time - prev_time > t:
                # too big gap
                break      
            if  prev_value >= peak_value / 2:
                # make candidate
                lhw = peak_time - prev_time
                lrd = (peak_value - prev_value) / (peak_time - prev_time + 0.001) 
            elif prev_value < full_height:
                lw = peak_time - prev_time
                lrd = (peak_value - prev_value) / (peak_time - prev_time + 0.001) 
                if lhw is np.nan:
                    lhw = peak_time - prev_time
                break
            elif prev_value > full_height:
                lw = peak_time - prev_time     
                lrd = (peak_value - prev_value) / (peak_time - prev_time + 0.001) 
                if lhw is np.nan:
                    lhw = peak_time - prev_time
            i -= 1 

            
    period = np.nan
    if len(arr_h) <= 5 and len(arr_h) > 1:
        values = []
        for i in arr_h:
            values.append(x['mjd'].iloc[i-1])
            
        lenght = len(arr_h)
        period = 0    
        count = 0
        for i in range(5):
            for j in range(0, lenght-1-i):
                count += 1
                period += (values[-1-j] - values[-2-j-i]) / (i+1)
        period = period / count
        
    if debug:
        return arr_vh, arr_h, lhw, rhw, lw, rw, lrd, rrd, peak_time, period
    else:
        return len(arr_vh), len(arr_h), lhw, rhw, lw, rw, lrd, rrd, peak_time, period    


def smoth_light_curve(x):
    x = x.set_index('mjd')
    window = x.shape[0] // 20 if x.shape[0] // 20 >= 6 else 5
    x = x.rolling(window, min_periods=0).mean().reset_index()
    t0, t1, t2, t3, t4, t5, t6, t7, t8, t9 = make_peak_features(x, alpha=0.75, t=125) 
    return t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, cid_ce(x['flux'])

def just_smoth(x):
    x = x.set_index('mjd')
    window = x.shape[0] // 20 if x.shape[0] // 20 >= 6 else 5
    x = x.rolling(window, min_periods=0).mean().reset_index()
    return x


def smoth_light_curve2(x, col):
    x = x.set_index('mjd').dropna()
    window = x.shape[0] // 20 if x.shape[0] // 20 >= 6 else 5
    x = x.rolling(window, min_periods=0).mean().reset_index()
    
    if x.size !=0:
        return make_peak_features2(x, alpha=0.75, t=125, col=col)
    else:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    
def polyfit_agg(df):
    if df.shape[0] <= 10:
        return np.nan
    
    f1 = np.polyfit(x=df['mjd'], y=df['flux'], deg=1, full=False)[0]
    return f1