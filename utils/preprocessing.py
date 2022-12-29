import numpy as np
import stft
import mne
from mne.preprocessing import ICA

def create_mne_raw(data, sfreq, chs=None):
    '''
    data: signal with shape (channel x samples)
    '''
    if chs is None:
        chs_ = ['ch{}'.format(i) for i in range(data.shape[0])]
    else:
        print (data.shape[0], len(chs))
        assert data.shape[0] == len(chs)
        chs_ = chs    
    ch_types = ['eeg' for i in range(len(chs_))]    
    info = mne.create_info(ch_names=chs_, sfreq=sfreq, ch_types=ch_types, verbose=False)
    raw = mne.io.RawArray(data*1e-6, info)
    return raw

def detect_interupted_data(sig_filt, freq):
    # sig_filt = butter_highpass_filter(sig, 0.5, freq, axis=-1, order=5)    
    # sig_filt = sig
    print ('sig_filt', sig_filt.shape)
    this_max_amp = 0
    normal_amp = 0
    try:
        normal_amp = np.min(np.max(np.abs(sig_filt), axis=0))
    except:
        return True
    for i in range(sig_filt.shape[1]):
        this_max_amp = np.max(np.abs(sig_filt[:,i]))
        # print (normal_amp, this_max_amp)
        if this_max_amp/normal_amp > 50: # overflow in signal recording
            print ('Signal overflow: {:.2f}, {:.2f}'.format(this_max_amp, normal_amp))
            return True
    
    # plot_eeg(sig)
    
    # stft_data = stft.spectrogram(sig_filt,framelength=freq,centered=False)        
    # stft_data = np.abs(stft_data)+1e-6        
    # dc_comp = np.sum(stft_data[0,:,:])
    # ac_comp = np.sum(stft_data[4:31,:,:])/27.0
    # print ('dc/ac ratio is {}'.format(dc_comp/ac_comp))   
    # if (dc_comp/ac_comp > 500): # signal is almost DC
    #     print ('Signal is almost DC')
    #     return True               

    n = 10
    _len = int(sig_filt.shape[0]/n)
    for i in range(n):
        _s = sig_filt[i*_len:(i+1)*_len]
        stds = np.std(_s, axis=0)
        # print (stds.shape, stds)
        if (stds<1).any():
            print ('Signal is almost DC')
            return True
        if (stds>5000).any():
            print ('Signal is unstable')
            return True

    return False

def ica_arti_remove(data, sfreq, chs=None):
    raw = create_mne_raw(data, sfreq, chs)
    # raw.plot(block=True, scalings=50e-6, remove_dc=False, lowpass=70)

    filt_raw = raw.copy()
    filt_raw.load_data().filter(l_freq=0.1, h_freq=None, verbose=False)

    ica = ICA(n_components=19, random_state=13)
    try:
        ica.fit(filt_raw, verbose=False)
    except:
        return None

    filt_raw.load_data()
    # ica.plot_sources(filt_raw)

    #ica.plot_components()

    ica.exclude = []
    # find which ICs match the EOG pattern
    eog_indices1, eog_scores1 = ica.find_bads_eog(filt_raw, threshold=2.0, ch_name='Fp1', verbose=False)
    print ('eog_indices', eog_indices1)
    eog_indices2, eog_scores2 = ica.find_bads_eog(filt_raw, threshold=2.0, ch_name='Fp2', verbose=False)
    print ('eog_indices', eog_indices2)

    # # find which ICs match the ECG pattern
    # ecg_indices, ecg_scores = ica.find_bads_ecg(raw, method='correlation') 
    
    if len(eog_indices1) > 0:
        ica.exclude.append(eog_indices1[0])
    if len(eog_indices2) > 0:
        ica.exclude.append(eog_indices2[0])

    print ('ica.exclude', ica.exclude)

    if len(ica.exclude) > 0:
        # ica.exclude.append(eog_indices[0])
        # ica.exclude = eog_indices[:2]
        reconst_raw = filt_raw.copy()
        reconst_raw.load_data()
        ica.apply(reconst_raw)
        # reconst_raw.plot(scalings=50, title='Filtered signals')
        print ('Reconstructing data from ICA components...')
        # reconst_raw.plot(block=True, scalings=50e-6, remove_dc=False, lowpass=70)
        return reconst_raw.get_data()*1e6
    
    return data   

