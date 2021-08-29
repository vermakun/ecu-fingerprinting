import numpy as np
from scipy import signal
from scipy.interpolate import splev, splrep

class Record:

  sr = 20000  # Sample Rate - 20 kHz

  def __init__(self,array):
    self.array = array

    ## Instance Variables
    # Control System Features
    self.dom_pp = []
    self.rec_pp = []
    self.dom_bp = []
    self.rec_bp = []
    self.dom_pt = []
    self.rec_pt = []
    self.dom_ssv = []
    self.rec_ssv = []
    self.dom_sse = []
    self.rec_sse = []
    self.dom_po = []
    self.rec_po = []
    self.dom_st_s = []
    self.rec_st_s = []
    self.dom_rt_s = []
    self.rec_rt_s = []
    self.dom_dt_s = []
    self.rec_dt_s = []

    # Spectral Analysis Features
    self.dom_pulse_data = []
    self.rec_pulse_data = []
    self.dom_sd = []
    self.rec_sd = []
    self.dom_snr = []
    self.rec_snr = []
    self.dom_mdfr = []
    self.rec_mdfr = []
    self.dom_mnfr = []
    self.rec_mnfr = []

    # Outlier Check Results
    self.outlier_count = []

    ## Instance Methods
    # Control System Processing
    self.PeakDetection()
    self.PeakTime()
    self.SteadyStateValErr()
    self.PercentOvershoot()
    self.SettlingTime()
    self.RiseTime()
    self.DelayTime()

    # Spectral Analysis Processing
    self.RunSpectralAnalysis()

    self.total_rec = np.min((len(self.dom_sd), len(self.rec_sd)))

    # # Outlier Detection and Removal
    # self.OutlierCount()
    # self.RemoveOutliers()

    # # Build Feature Datastructure
    self.features    = self.GenerateFeatures()

    self.headers = []
    self.headers.append('Dom_Peak_Time')
    self.headers.append('Dom_Steady_State_Value')
    self.headers.append('Dom_Steady_State_Error')
    self.headers.append('Dom_Percent_Overshoot')
    self.headers.append('Dom_Settling_Time')
    self.headers.append('Dom_Rise_Time')
    self.headers.append('Dom_Delay_Time')
    for i in np.arange(len(self.dom_sd[0])):
      self.headers.append('Dom_Spectral_Bin_%d' % i)
    self.headers.append('Dom_SNR')
    self.headers.append('Dom_Mean_Freq')
    self.headers.append('Dom_Median_Freq')

    self.headers.append('Rec_Peak_Time')
    self.headers.append('Rec_Steady_State_Value')
    self.headers.append('Rec_Steady_State_Error')
    self.headers.append('Rec_Percent_Overshoot')
    self.headers.append('Rec_Settling_Time')
    self.headers.append('Rec_Rise_Time')
    self.headers.append('Rec_Delay_Time')
    for i in np.arange(len(self.rec_sd[0])):
      self.headers.append('Rec_Spectral_Bin_%d' % i)
    self.headers.append('Rec_SNR')
    self.headers.append('Rec_Mean_Freq')
    self.headers.append('Rec_Median_Freq')

  def PeakDetection(self):
    ##### PeakDetection
    # Input:    array  - raw signal data for record
    # Output:   dom_pp - dominant  Pulse Peak index
    #           dom_bp - dominant  Before Pulse peak index
    #           rec_pp - recessive Pulse Peak index
    #           rec_bp - recessive Before Pulse peak index

    ### Pulse Peak Detection ###

    # Calculate difference array
    arr_diff = np.diff(self.array, prepend=self.array[0])

    # Perform moving average filter, width=3, x2
    w = 3
    arr_diff = np.convolve(arr_diff, np.ones(w), 'valid') / w
    arr_diff = np.convolve(arr_diff, np.ones(w), 'valid') / w

    # Prepend zeros to offset processing delay
    arr_diff = np.insert(arr_diff, 0, np.zeros((w-1)*2), axis=0)

    # Crossing filter to detect dominant and recessive leading edge zones
    dom_pp_ts = (arr_diff >  0.2).astype(float)
    rec_pp_ts = (arr_diff < -0.2).astype(float)

    # Find peak for each zone (dominant)
    a = np.where(dom_pp_ts == 1)[0].astype(float)
    b = np.diff(a, prepend=0)
    c = np.where(b > 1)[0]
    dom_pp = a[c].astype(int)

    # Remove errant peaks (dominant)
    corr_idx = np.concatenate((np.diff(dom_pp),[np.average(np.diff(dom_pp))]))
    if np.min(np.diff(corr_idx)) < 100:
      corr_idx = np.where(corr_idx > np.average(corr_idx/4))[0]
      dom_pp = dom_pp[corr_idx]

    # Find peak for each zone (recessive)
    a = np.where(rec_pp_ts == 1)[0].astype(float)
    b = np.diff(a, prepend=0)
    c = np.where(b > 1)[0]
    rec_pp = a[c].astype(int)

    # Remove errant peaks (recessive)
    corr_idx = np.concatenate((np.diff(rec_pp),[np.average(np.diff(rec_pp))]))
    if np.min(np.diff(corr_idx)) < 15:
      corr_idx = np.where(corr_idx > np.average(corr_idx/4))[0]
      rec_pp = rec_pp[corr_idx]

    # Pair dom and rec indices

    dom_len = len(dom_pp)
    rec_len = len(rec_pp)

    dom_is_larger = []

    if dom_len > rec_len + 1:
      dom_is_larger = 1
    elif rec_len > dom_len + 1:
      dom_is_larger = 0

    if not dom_is_larger == []:
      len_min = np.min((dom_len, rec_len))
      len_dif = np.abs(dom_len - rec_len) + 1

      dif_amt = []

      for i in np.arange(len_dif):
        if dom_is_larger:
          temp = dom_pp[0:dom_len] - rec_pp[i:dom_len+i]
        else:
          temp = dom_pp[0:dom_len] - rec_pp[i:dom_len+i]
        temp = np.abs(temp)
        temp = np.sum(temp)
        dif_amt.append(temp)
        
      dif_loc = np.where(np.min(dif_amt) == dif_amt)[0]
      
      if dom_is_larger:
        dom_pp = dom_pp[dif_loc[0]:rec_len+dif_loc[0]+1]
      else:
        rec_pp = rec_pp[dif_loc[0]:dom_len+dif_loc[0]+1]

    # Create timestamps using indices
    dom_pp_ts = np.zeros(dom_pp_ts.size)
    dom_pp_ts[dom_pp] = 1
    self.dom_pp = np.where(dom_pp_ts == 1)[0]

    rec_pp_ts = np.zeros(rec_pp_ts.size)
    rec_pp_ts[rec_pp] = 1
    self.rec_pp = np.where(rec_pp_ts == 1)[0]

    ### Pre-Peak Detection ###

    # Crossing filter to detect pre-dominant steady state (Before Leading-edge)
    dom_bp_ts = np.abs(np.diff(self.array - 2.5, prepend = self.array[0]))
    w = 5
    dom_bp_ts = np.convolve(dom_bp_ts, np.ones(w), 'valid') / w
    dom_bp_ts = np.insert(dom_bp_ts, 0, np.zeros(w-1), axis=0)
    dom_bp_ts = 1-(dom_bp_ts > 0.05).astype(float)

    # Crossing filter to detect pre-recessive steady state (Before Leading-edge)
    rec_bp_ts = np.abs(np.diff(3.5 - self.array, prepend = self.array[0]))
    w = 5
    rec_bp_ts = np.convolve(rec_bp_ts, np.ones(w), 'valid') / w
    rec_bp_ts = np.insert(rec_bp_ts, 0, np.zeros(w-1), axis=0)
    rec_bp_ts = 1-(rec_bp_ts > 0.05).astype(float)

    ## Find the last instance of steady state prior to dominant peaks

    jj = np.zeros(dom_pp.size).astype(int)
    for k in np.arange(0,dom_pp.size):
      # "Dominant-low steady state" indices before peak
      j = np.where(dom_bp_ts[0:dom_pp[k]] == 1)
      j = j[0]
      

      # Find nearest index before dominant peak
      min_idx = j-dom_pp[k]
      min_idx = min_idx[np.where(np.min(np.abs(min_idx)) == np.abs(min_idx))[0]]
      jj[k] = ((min_idx + dom_pp[k])[0])

    # Dominant prior-to-peak steady-state indices
    dom_bp_ts2 = np.zeros(dom_bp_ts.size, dtype=int)
    dom_bp_ts2[jj] = 1
    self.dom_bp = jj

    ## Find the last instance of steady state prior to recessive peaks

    jj = np.zeros(rec_pp.size).astype(int)
    for k in np.arange(0,rec_pp.size):
      # "Recesive-low steady state" indices before peak
      j = np.where(rec_bp_ts[0:rec_pp[k]] == 1)
      j = j[0]

      # Find nearest index before recessive peak
      min_idx = j-rec_pp[k]
      min_idx = min_idx[np.where(np.min(np.abs(min_idx)) == np.abs(min_idx))[0]]
      jj[k] = ((min_idx + rec_pp[k])[0])

    # Recessive prior-to-peak steady-state indices
    rec_bp_ts2 = np.zeros(rec_bp_ts.size, dtype=int)
    rec_bp_ts2[jj] = 1
    self.rec_bp = jj

  def PeakTime(self):
    ##### PeakTime
    # Input:    dom_pp - dominant  Pulse Peak index
    #           dom_bp - dominant  Before Pulse peak index
    #           rec_pp - recessive Pulse Peak index
    #           rec_bp - recessive Before Pulse peak index
    #           sr - sample rate of the raw data
    # Output:   dom_pt - dominant  Peak Time
    #           rec_pt - recessive Peak Time

    self.dom_pt = (self.dom_pp-self.dom_bp)/Record.sr
    self.rec_pt = (self.rec_pp-self.rec_bp)/Record.sr

  def SteadyStateValErr(self):
    ##### Steady State Value and Error
    # Input:    array  - raw signal data for record
    #           dom_bp - dominant  Before Pulse peak index
    #           rec_bp - recessive Before Pulse peak index
    # Output:   dom_ssv - dominant  Steady State Value
    #           rec_ssv - recessive Steady State Value
    #           dom_sse - dominant  Steady State Error
    #           rec_sse - recessive Steady State Error

    # Perform moving average filter, width=19
    w = 19
    arr_avg = np.convolve(self.array, np.ones(w), 'valid') / w
    arr_avg = np.insert(arr_avg, 0, arr_avg[0]*np.ones(w-1), axis=0)

    # Extract Steady State Value from previous Steady State Index
    dom_ssv_idx = self.rec_bp
    rec_ssv_idx = self.dom_bp

    self.dom_ssv = arr_avg[dom_ssv_idx]
    self.rec_ssv = arr_avg[rec_ssv_idx]

    # Calculate Steady State Error
    self.dom_sse = arr_avg[dom_ssv_idx] - 3.5
    self.rec_sse = arr_avg[rec_ssv_idx] - 2.5

  def PercentOvershoot(self):
    ##### Percent Overshoot
    # Input:    array  - raw signal data for record
    #           dom_pp - dominant  Before Pulse peak index
    #           rec_pp - recessive Before Pulse peak index
    #           dom_ssv - dominant  Steady State Value
    #           rec_ssv - recessive Steady State Value
    # Output:   dom_po - dominant  Percent Overshoot
    #           rec_po - recessive Percent Overshoot

    dom_pv = self.array[self.dom_pp]
    rec_pv = self.array[self.rec_pp]

    try:
      self.dom_po = 100 * (dom_pv - self.dom_ssv) / self.dom_ssv
      self.rec_po = 100 * (self.rec_ssv - rec_pv) / self.rec_ssv
    except:
      self.dom_po = 100 * (dom_pv - np.average(self.dom_ssv)) / np.average(self.dom_ssv)
      self.rec_po = 100 * (np.average(self.rec_ssv) - rec_pv) / np.average(self.rec_ssv)

  def SettlingTime(self):
    ##### Settling Time
    # Input:    array  - raw signal data for record
    #           dom_pp - dominant  Before Pulse peak index
    #           rec_pp - recessive Before Pulse peak index
    #           dom_ssv - dominant  Steady State Value
    #           rec_ssv - recessive Steady State Value
    #           sr - sample rate of the raw data
    # Output:   dom_st_s - dominant  Settling Time (s)
    #           rec_st_s - recessive Settling Time (s)

    ss_rng = 0.05   # 5% Steady State Range of 1V Vpp design

    # Find index and time of settling point (dominant)
    w = 3
    arr_avg1 = np.convolve(np.abs(self.array-np.average(self.dom_ssv)), np.ones(w), 'valid') / w
    arr_avg1 = np.insert(arr_avg1, 0, arr_avg1[0]*np.ones(w-1), axis=0)

    arr_avg11 = np.abs(np.round(arr_avg1,decimals=2))
    dom_st_idx = np.where(arr_avg11 <= ss_rng)[0]
    dom_st = np.zeros(self.dom_pp.size)
    if dom_st_idx.size != 0:
      for i in np.arange(self.dom_pp.size):
          dom_st_idx[dom_st_idx <= self.dom_pp[i]] = -self.array.size
          j = np.where(
                np.min(np.abs(dom_st_idx - self.dom_pp[i])) 
                  == np.abs(dom_st_idx - self.dom_pp[i])
              )[0][-1]
          dom_st[i] = dom_st_idx[j]
      dom_st = dom_st.astype(int)
    else:
      self.dom_st = np.concatenate((self.dom_pp[1:],[self.array.size]))

    self.dom_st_s = (dom_st - self.dom_pp)/Record.sr

    # Find index and time of settling point (dominant)
    w = 3
    arr_avg2 = np.convolve(np.average(self.dom_ssv)-self.array, np.ones(w), 'valid') / w
    arr_avg2 = np.insert(arr_avg2, 0, arr_avg2[0]*np.ones(w-1), axis=0)

    arr_avg22 = np.abs(np.round(arr_avg2,decimals=2))
    rec_st_idx = np.where(arr_avg22 <= ss_rng)[0]
    rec_st = np.zeros(self.rec_pp.size)
    for i in np.arange(self.rec_pp.size):
        rec_st_idx[rec_st_idx <= self.rec_pp[i]] = -self.array.size
        j = np.where(
              np.min(np.abs(rec_st_idx - self.rec_pp[i])) 
                == np.abs(rec_st_idx - self.rec_pp[i])
            )[0][-1]
        rec_st[i] = rec_st_idx[j]
    rec_st = rec_st.astype(int)

    self.rec_st_s = (rec_st - self.rec_pp)/Record.sr

  def RiseTime(self):
    ##### Rise Time
    # Input:    array  - raw signal data for record
    #           dom_pp - dominant  Pulse Peak index
    #           rec_pp - recessive Pulse Peak index
    #           dom_bp - dominant  Before Pulse peak index
    #           rec_bp - recessive Before Pulse peak index
    #           dom_ssv - dominant  Steady State Value
    #           rec_ssv - recessive Steady State Value
    #           sr - sample rate of the raw data
    # Output:   dom_rt_s - dominant  Settling Time (s)
    #           rec_rt_s - recessive Settling Time (s)

    # Find index and time of rise point (dominant)
    dom_rt_ts = (self.array.copy() - np.average(self.rec_ssv) <= 1).astype(int)
    dom_rt_idx = np.where(dom_rt_ts == 1)[0]

    dom_rt = np.zeros(self.dom_pp.size)
    for i in np.arange(self.dom_pp.size):
      j = np.where(np.min(np.abs(dom_rt_idx - self.dom_pp[i])) 
        == np.abs(dom_rt_idx - self.dom_pp[i]))[0][-1]
      dom_rt[i] = dom_rt_idx[j]
    dom_rt = dom_rt.astype(int)

    self.dom_rt_s = (dom_rt - self.dom_bp)/Record.sr

    # Find index and time of rise point (recessive)
    rec_rt_ts = (-self.array.copy() + np.average(self.dom_ssv) <= 1).astype(int)
    rec_rt_idx = np.where(rec_rt_ts == 1)[0]

    rec_rt = np.zeros(self.rec_pp.size)
    for i in np.arange(self.rec_pp.size):
      j = np.where(np.min(np.abs(rec_rt_idx - self.rec_pp[i]))
        == np.abs(rec_rt_idx - self.rec_pp[i]))[0][-1]
      rec_rt[i] = rec_rt_idx[j]
    rec_rt = rec_rt.astype(int)

    self.rec_rt_s = (rec_rt - self.rec_bp)/Record.sr

  def DelayTime(self):
    ##### Delay Time
    # Input:    array  - raw signal data for record
    #           dom_pp - dominant  Pulse Peak index
    #           rec_pp - recessive Pulse Peak index
    #           dom_bp - dominant  Before Pulse peak index
    #           rec_bp - recessive Before Pulse peak index
    #           dom_ssv - dominant  Steady State Value
    #           rec_ssv - recessive Steady State Value
    #           sr - sample rate of the raw data
    # Output:   dom_rt_s - dominant  Settling Time (s)
    #           rec_rt_s - recessive Settling Time (s)

    # Find index and time of delay point (dominant)
    dom_dt_ts = (self.array.copy() - np.average(self.rec_ssv) <= 0.5).astype(int)
    dom_dt_idx = np.where(dom_dt_ts == 1)[0]

    dom_dt = np.zeros(self.dom_pp.size)
    for i in np.arange(self.dom_pp.size):
      j = np.where(np.min(np.abs(dom_dt_idx - self.dom_pp[i]))
        == np.abs(dom_dt_idx - self.dom_pp[i]))[0][-1]
      dom_dt[i] = dom_dt_idx[j]
    dom_dt = dom_dt.astype(int)

    self.dom_dt_s = (dom_dt - self.dom_bp)/Record.sr

    # Find index and time of delay point (recessive)
    rec_dt_ts = (-self.array.copy() + np.average(self.dom_ssv) <= 0.5).astype(int)
    rec_dt_idx = np.where(rec_dt_ts == 1)[0]

    rec_dt = np.zeros(self.rec_pp.size)
    for i in np.arange(self.rec_pp.size):
      j = np.where(np.min(np.abs(rec_dt_idx - self.rec_pp[i]))
        == np.abs(rec_dt_idx - self.rec_pp[i]))[0][-1]
      rec_dt[i] = rec_dt_idx[j]
    rec_dt = rec_dt.astype(int)

    self.rec_dt_s = (rec_dt - self.rec_bp)/Record.sr

  def RunSpectralAnalysis(self):
    ##### Spectral Analysis
    # Run the following methods:
    #
    #     + Spectral Density Binning
    #     + Signal-to-Noise Ratio
    #     + Median Frequency
    #     + Mean Frequency
    #
    #     Features will be processed for both
    #       Dominant and Recessive CAN High bits

    self.SpectralDensityBinning()
    self.SignalToNoiseRatio()
    self.MeanMedianFrequency()

  def SpectralDensityBinning(self):
    ##### Bin Spectral Density

    index_shift = -5  # Include some steady state info from prev pulse

    dom_pp_sd = self.dom_pp.copy() + index_shift
    rec_pp_sd = self.rec_pp.copy() + index_shift

    # Find the start/end pulse indices

    if self.dom_pp[0] <= self.rec_pp[0]:
      if len(self.dom_pp) > len(self.rec_pp):
        dom_pp_sd = dom_pp_sd[0:-1]
      idx_dom_se = np.array([dom_pp_sd,rec_pp_sd])
      idx_rec_se = np.array([rec_pp_sd[0:-1],dom_pp_sd[1:]])
    else:
      if len(self.rec_pp) > len(self.dom_pp):
        rec_pp_sd = rec_pp_sd[0:-1]
      idx_rec_se = np.array([rec_pp_sd,dom_pp_sd])
      idx_dom_se = np.array([dom_pp_sd[0:-1],rec_pp_sd[1:]])

    # Remove pulses that don't provide enough steady-state information from the prev pulse

    if idx_dom_se[0][0] < -index_shift:
      idx_dom_se = np.array([idx_dom_se[0][1:],idx_dom_se[1][1:]])

    if idx_rec_se[0][0] < -index_shift:
      idx_rec_se = np.array([idx_rec_se[0][1:],idx_rec_se[1][1:]])

    # Check for out-or-order index error

    if idx_dom_se[0][0] > idx_dom_se[1][0]:
      temp1 = np.array([idx_dom_se[1],idx_dom_se[0]])
      temp2 = np.array([idx_dom_se[0],idx_rec_se[1]])

      idx_dom_se = temp2
      idx_rec_se = temp1

    # Save dom pulse info to parent method variable dom_pulse_data
    for i in np.arange(idx_dom_se.shape[1]):
      self.dom_pulse_data.append(self.array[idx_dom_se[0][i]:idx_dom_se[1][i]])

    # Save dom pulse info to parent method variable rec_pulse_data
    for i in np.arange(idx_rec_se.shape[1]):
      self.rec_pulse_data.append(self.array[idx_rec_se[0][i]:idx_rec_se[1][i]])

    # Reset indices
    idx_dom_se = idx_dom_se - index_shift
    idx_rec_se = idx_rec_se - index_shift

    # Bin power densities

    def binned_sd(Pxx_den, nbins):

      bs = Pxx_den.size/nbins
      bs = round(bs)

      Pxx_hist = []
      for i in np.arange(nbins):
        idx_s = i*bs
        idx_e = (i+1)*bs

        if idx_e >= Pxx_den.size:
          idx_e = Pxx_den.size - 1

        Pxx_hist.append(np.average(Pxx_den[idx_s:idx_e]))

      Pxx_hist = np.nan_to_num(Pxx_hist)

      return Pxx_hist

    # Select bin sizes

    bin_sel = 2
    dom_nbin = [15,13,10]    # Bin size limited by pulse length

    # Perform binning of spectral density

    self.dom_sd = []
    for i in np.arange(len(self.dom_pulse_data)):
      f, pd = signal.welch(self.dom_pulse_data[i], Record.sr);
      self.dom_sd.append(binned_sd(pd, dom_nbin[bin_sel]))

    rec_nbin = [10, 8, 5]    # Bin size limited by pulse length

    self.rec_sd = []
    for i in np.arange(len(self.rec_pulse_data)):
      f, pd = signal.welch(self.rec_pulse_data[i], Record.sr);
      self.rec_sd.append(binned_sd(pd, rec_nbin[bin_sel]))


  def SignalToNoiseRatio(self):

    index_shift = -5

    self.dom_snr = []
    for i in np.arange(len(self.dom_pulse_data)):
      cur_array = self.dom_pulse_data[i]
      signl = (np.arange(len(cur_array)) > -index_shift-1).astype(float)*np.average(self.dom_ssv) + \
        (np.arange(len(cur_array)) <= -index_shift-1).astype(float)*np.average(self.rec_ssv)
      noise = signl - cur_array

      f, s_pd = signal.welch(signl, Record.sr);
      f, n_pd = signal.welch(noise, Record.sr);

      Ps = sum(s_pd)
      Pn = sum(n_pd)
      
      if Pn == 0:
        self.rec_snr.append(np.nan)
        continue
      self.dom_snr.append(10*np.log10(Ps/Pn))

    self.rec_snr = []
    for i in np.arange(len(self.rec_pulse_data)):
      cur_array = self.rec_pulse_data[i]
      signl = (np.arange(len(cur_array)) > -index_shift-2).astype(float)*np.average(self.rec_ssv) + \
        (np.arange(len(cur_array)) <= -index_shift-2).astype(float)*np.average(self.dom_ssv)
      noise = signl - cur_array
  
      f, s_pd = signal.welch(signl, Record.sr)
      f, n_pd = signal.welch(noise, Record.sr)

      Ps = sum(s_pd)
      Pn = sum(n_pd)

      if Pn == 0:
        self.rec_snr.append(np.nan)
        continue

      self.rec_snr.append(10*np.log10(Ps/Pn))

  def MeanMedianFrequency(self):

    self.dom_mdfr = []
    self.rec_mdfr = []
    self.dom_mnfr = []
    self.rec_mnfr = []


    self.dom_mnfr = []
    self.dom_mdfr = []
    for i in np.arange(len(self.dom_pulse_data)):
      cur_pulse = self.dom_pulse_data[i]
      f, pd = signal.welch(cur_pulse, Record.sr)

      spl = splrep(f, pd, k=1)
      x2 = np.arange(f[0], f[-1],0.01)
      y2 = splev(x2, spl)

      y21 = y2/np.sum(y2)                          # Normalize spectra
      y22 = np.cumsum(y21)                         # Cummulative sum (CDF for SPD)
      y23 = y22-0.5                                # Subtract 50% of energy
      y24 = abs(y23)                               # Abs value to create a minima
      y25 = np.where(np.min(y24) == y24)[0][-1]    # Locate minima index
      self.dom_mdfr.append(x2[y25])                # Retrieve minima frequency  

      self.dom_mnfr.append(np.sum(pd*f)/np.sum(pd))

    self.rec_mnfr = []
    self.rec_mdfr = []
    for i in np.arange(len(self.rec_pulse_data)):
      cur_pulse = self.rec_pulse_data[i]
      f, pd = signal.welch(cur_pulse, Record.sr)

      spl = splrep(f, pd, k=1)
      x2 = np.arange(f[0], f[-1],0.01)
      y2 = splev(x2, spl)

      y21 = y2/np.sum(y2)                          # Normalize spectra
      y22 = np.cumsum(y21)                         # Cummulative sum (CDF for SPD)
      y23 = y22-0.5                                # Subtract 50% of energy
      y24 = abs(y23)                               # Abs value to create a minima
      y25 = np.where(np.min(y24) == y24)[0][-1]    # Locate minima index
      self.rec_mdfr.append(x2[y25])                # Retrieve minima frequency  

      self.rec_mnfr.append(np.sum(pd*f)/np.sum(pd))

  def OutlierCount(self):
    ##### Outlier Count
    # Calculates the standard deviation for each feature and creates a binary
    # mask of pulses that exceed the standard deviation threshold
    # Binary masks are added to determine total number of deviations per pulse 
    # across all features
    
    std = 1.5   # Threshold

    def fix_size_disparity(in1, in2):
      if in1.size > in2.size:
        in2 = np.concatenate((in2,np.zeros(in1.size - in2.size))).astype(int)
      elif in2.size > in1.size:
        in1 = np.concatenate((in1,np.zeros(in2.size - in1.size))).astype(int)
      return in1, in2

    # Outlier check and size correction
    self.dom_pp, self.rec_pp = fix_size_disparity(self.dom_pp, self.rec_pp)
    self.dom_bp, self.rec_bp = fix_size_disparity(self.dom_bp, self.rec_bp)

    self.dom_pt, self.rec_pt = fix_size_disparity(self.dom_pt, self.rec_pt)
    dom_pt_out = (np.abs(self.dom_pt-np.average(self.dom_pt)) >
      std*np.std(self.dom_pt)).astype(int)
    rec_pt_out = (np.abs(self.rec_pt-np.average(self.rec_pt)) >
      std*np.std(self.rec_pt)).astype(int)
    pt_out = dom_pt_out + rec_pt_out

    self.dom_ssv, self.rec_ssv = fix_size_disparity(self.dom_ssv, self.rec_ssv)
    dom_ssv_out = (np.abs(self.dom_ssv-np.average(self.dom_ssv)) >
      std*np.std(self.dom_ssv)).astype(int)
    rec_ssv_out = (np.abs(self.rec_ssv-np.average(self.rec_ssv)) >
      std*np.std(self.rec_ssv)).astype(int)
    ssv_out = dom_ssv_out + rec_ssv_out

    self.dom_sse, self.rec_sse = fix_size_disparity(self.dom_sse, self.rec_sse)
    dom_sse_out = (np.abs(self.dom_sse-np.average(self.dom_sse)) >
      std*np.std(self.dom_sse)).astype(int)
    rec_sse_out = (np.abs(self.rec_sse-np.average(self.rec_sse)) >
      std*np.std(self.rec_sse)).astype(int)
    sse_out = dom_sse_out + rec_sse_out

    self.dom_po, self.rec_po = fix_size_disparity(self.dom_po, self.rec_po)
    dom_po_out = (np.abs(self.dom_po-np.average(self.dom_po)) >
      std*np.std(self.dom_po)).astype(int)
    rec_po_out = (np.abs(self.rec_po-np.average(self.rec_po)) >
      std*np.std(self.rec_po)).astype(int)
    po_out = dom_po_out + rec_po_out

    self.dom_st_s, self.rec_st_s = fix_size_disparity(self.dom_st_s, self.rec_st_s)
    dom_st_s_out = (np.abs(self.dom_st_s-np.average(self.dom_st_s)) >
      std*np.std(self.dom_st_s)).astype(int)
    rec_st_s_out = (np.abs(self.rec_st_s-np.average(self.rec_st_s)) >
      std*np.std(self.rec_st_s)).astype(int)
    st_s_out = dom_st_s_out + rec_st_s_out

    self.dom_rt_s, self.rec_rt_s = fix_size_disparity(self.dom_rt_s, self.rec_rt_s)
    dom_rt_s_out = (np.abs(self.dom_rt_s-np.average(self.dom_rt_s)) >
      std*np.std(self.dom_rt_s)).astype(int)
    rec_rt_s_out = (np.abs(self.rec_rt_s-np.average(self.rec_rt_s)) >
      std*np.std(self.rec_rt_s)).astype(int)
    rt_s_out = dom_rt_s_out + rec_rt_s_out

    self.dom_dt_s, self.rec_dt_s = fix_size_disparity(self.dom_dt_s, self.rec_dt_s)
    dom_dt_s_out = (np.abs(self.dom_dt_s-np.average(self.dom_dt_s)) >
      std*np.std(self.dom_dt_s)).astype(int)
    rec_dt_s_out = (np.abs(self.rec_dt_s-np.average(self.rec_dt_s)) >
      std*np.std(self.rec_dt_s)).astype(int)
    dt_s_out = dom_dt_s_out + rec_dt_s_out

    self.outlier_count = pt_out + ssv_out + sse_out + \
      po_out + st_s_out + rt_s_out + dt_s_out
    return self.outlier_count

  def RemoveOutliers(self):
    ##### Remove Outlier Pulses
    # Checks outlier count for each pulse and removes pulses that exceed
    # the deviation threshold

    dev = 6

    noutlier_idx = np.where(self.outlier_count < dev + 1)[0]

    self.dom_pp   = self.dom_pp[noutlier_idx]
    self.rec_pp   = self.rec_pp[noutlier_idx]
    self.dom_bp   = self.dom_bp[noutlier_idx]
    self.rec_bp   = self.rec_bp[noutlier_idx]
    self.dom_pt   = self.dom_pt[noutlier_idx]
    self.rec_pt   = self.rec_pt[noutlier_idx]
    self.dom_ssv  = self.dom_ssv[noutlier_idx]
    self.rec_ssv  = self.rec_ssv[noutlier_idx]
    self.dom_sse  = self.dom_sse[noutlier_idx]
    self.rec_sse  = self.rec_sse[noutlier_idx]
    self.dom_po   = self.dom_po[noutlier_idx]
    self.rec_po   = self.rec_po[noutlier_idx]
    self.dom_st_s = self.dom_st_s[noutlier_idx]
    self.rec_st_s = self.rec_st_s[noutlier_idx]
    self.dom_rt_s = self.dom_rt_s[noutlier_idx]
    self.rec_rt_s = self.rec_rt_s[noutlier_idx]
    self.dom_dt_s = self.dom_dt_s[noutlier_idx]
    self.rec_dt_s = self.rec_dt_s[noutlier_idx]

    self.OutlierCount()

  def summary(self):
      print('Peak Time (s):')
      print('  dom: ', self.dom_pt)
      print('    avg: ', np.average(self.dom_pt))
      print('    std: ', np.std(self.dom_pt))
      print('    dev: ', np.abs(self.dom_pt-np.average(self.dom_pt)))
      # print('    out: ', dom_pt_out)
      print('  rec: ', self.rec_pt)
      print('    avg: ', np.average(self.rec_pt))
      print('    std: ', np.std(self.rec_pt))
      print('    dev: ', np.abs(self.rec_pt-np.average(self.rec_pt)))
      # print('    out: ', rec_pt_out)
      print('')
      print('Steady State Value (V):')
      print('  dom: ', self.dom_ssv)
      print('    avg: ', np.average(self.dom_ssv))
      print('    std: ', np.std(self.dom_ssv))
      print('    dev: ', np.abs(self.dom_ssv-np.average(self.dom_ssv)))
      # print('    out: ', dom_ssv_out)
      print('  rec: ', self.rec_ssv)
      print('    avg: ', np.average(self.rec_ssv))
      print('    std: ', np.std(self.rec_ssv))
      print('    dev: ', np.abs(self.rec_ssv-np.average(self.rec_ssv)))
      # print('    out: ', rec_ssv_out)
      print('')
      print('Steady State Error (V):')
      print('  dom: ', self.dom_sse)
      print('    avg: ', np.average(self.dom_sse))
      print('    std: ', np.std(self.dom_sse))
      print('    dev: ', np.abs(self.dom_sse-np.average(self.dom_sse)))
      # print('    out: ', dom_sse_out)
      print('  rec: ', self.rec_sse)
      print('    avg: ', np.average(self.rec_sse))
      print('    std: ', np.std(self.rec_sse))
      print('    dev: ', np.abs(self.rec_sse-np.average(self.rec_sse)))
      # print('    out: ', rec_sse_out)
      print('')
      print('Percent Overshoot')
      print('  dom: ', self.dom_po)
      print('    avg: ', np.average(self.dom_po))
      print('    std: ', np.std(self.dom_po))
      print('    dev: ', np.abs(self.dom_po-np.average(self.dom_po)))
      # print('    out: ', dom_po_out)
      print('  rec: ', self.rec_po)
      print('    avg: ', np.average(self.rec_po))
      print('    std: ', np.std(self.rec_po))
      print('    dev: ', np.abs(self.rec_po-np.average(self.rec_po)))
      # print('    out: ', rec_po_out)
      print('')
      print('Settling Time (s)')
      print('  dom: ', self.dom_st_s)
      print('    avg: ', np.average(self.dom_st_s))
      print('    std: ', np.std(self.dom_st_s))
      print('    dev: ', np.abs(self.dom_st_s-np.average(self.dom_st_s)))
      # print('    out: ', dom_st_s_out)
      print('  rec: ', self.rec_st_s)
      print('    avg: ', np.average(self.rec_st_s))
      print('    std: ', np.std(self.rec_st_s))
      print('    dev: ', np.abs(self.rec_st_s-np.average(self.rec_st_s)))
      # print('    out: ', rec_st_s_out)
      print('')
      print('Rise Time (s)')
      print('  dom: ', self.dom_rt_s)
      print('    avg: ', np.average(self.dom_rt_s))
      print('    std: ', np.std(self.dom_rt_s))
      print('    dev: ', np.abs(self.dom_rt_s-np.average(self.dom_rt_s)))
      # print('    out: ', dom_rt_s_out)
      print('  rec: ', self.rec_rt_s)
      print('    avg: ', np.average(self.rec_rt_s))
      print('    std: ', np.std(self.rec_rt_s))
      print('    dev: ', np.abs(self.rec_rt_s-np.average(self.rec_rt_s)))
      # print('    out: ', rec_rt_s_out)
      print('')
      print('Delay Time (s)')
      print('  dom: ', self.dom_dt_s)
      print('    avg: ', np.average(self.dom_dt_s))
      print('    std: ', np.std(self.dom_dt_s))
      print('    dev: ', np.abs(self.dom_dt_s-np.average(self.dom_dt_s)))
      # print('    out: ', dom_dt_s_out)
      print('  rec: ', self.rec_dt_s)
      print('    avg: ', np.average(self.rec_dt_s))
      print('    std: ', np.std(self.rec_dt_s))
      print('    dev: ', np.abs(self.rec_dt_s-np.average(self.rec_dt_s)))
      # print('    out: ', rec_dt_s_out)
      print('\nOutlier Count: ', self.outlier_count)

  def GenerateFeatures(self):
    out = []
    out.append(self.dom_pt)
    out.append(self.dom_ssv)
    out.append(self.dom_sse)
    out.append(self.dom_po)
    out.append(self.dom_st_s)
    out.append(self.dom_rt_s)
    out.append(self.dom_dt_s)
    temp = np.array(self.dom_sd)
    temp = temp.reshape((len(self.dom_sd[0]), len(self.dom_sd)))
    for i in temp:
      out.append(i)
    out.append(self.dom_snr)
    out.append(self.dom_mnfr)
    out.append(self.dom_mdfr)

    out.append(self.rec_pt)
    out.append(self.rec_ssv)
    out.append(self.rec_sse)
    out.append(self.rec_po)
    out.append(self.rec_st_s)
    out.append(self.rec_rt_s)
    out.append(self.rec_dt_s)
    temp = np.array(self.rec_sd)
    temp = temp.reshape((len(self.rec_sd[0]), len(self.rec_sd)))
    for i in temp:
      out.append(i)
    out.append(self.rec_snr)
    out.append(self.rec_mnfr)
    out.append(self.rec_mdfr)

    return out

