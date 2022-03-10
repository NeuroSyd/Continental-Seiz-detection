
# import glob
# import os
# import glob
# import datetime
# from datetime import timedelta
# import pandas as pd
# import scipy.io
import os
import numpy as np
from scipy.signal import resample
import json
import stft
from pyst import read_edf_elec
from preprocessing import detect_interupted_data, ica_arti_remove, create_mne_raw
import mne


def makedirs(dir):
    try:
        os.makedirs(dir)
        print('make the directory')
    except:
        print('fail to make the directory or already there')
        pass
def calc_stft(s_):
    # s_time = begin_rec + timedelta(seconds=int(i*numts))
    s = s_.transpose()

    stft_data = stft.spectrogram(s, framelength=250, centered=False)

    if stft_data.ndim ==2:
        stft_data = np.expand_dims(stft_data,-1)
    stft_data = np.transpose(stft_data, (1, 2, 0))
    stft_data = np.abs(stft_data) + 1e-6

    # if self.settings['dataset'] == 'FB':
    #     stft_data = np.concatenate((stft_data[:,:,1:47],
    #                                 stft_data[:,:,54:97],
    #                                 stft_data[:,:,104:]),
    #                                 axis=-1)
    # elif self.settings['dataset'] == 'CHBMIT':
    #     stft_data = np.concatenate((stft_data[:,:,1:57],
    #                                 stft_data[:,:,64:117],
    #                                 stft_data[:,:,124:]),
    #                                 axis=-1)
    # elif self.settings['dataset'] == 'EpilepsiaSurf':
    stft_data = stft_data[:,:,1:]
    stft_data = np.log10(stft_data)
    indices = np.where(stft_data <= 0)
    stft_data[indices] = 0

    # from matplotlib import cm
    # from matplotlib import pyplot as plt
    # plt.matshow(stft_data[0]/np.max(stft_data[0]))
    # plt.colorbar()
    # plt.show()

    stft_data = stft_data.reshape(-1, stft_data.shape[0],
                                  stft_data.shape[1],
                                  stft_data.shape[2])

    return stft_data


def get_ref_train_df_TUH(ref_train_file, all_files, cachedir,segement=None):
    print('using segement 12s ')
    window_len = 250 * segement  # 12 seconds
    with open(all_files, 'r') as f:
        all_filenames = f.readlines()
    print(len(all_filenames))
    # print (all_filenames[-1])
    count = 0
    with open(ref_train_file, 'r') as f:
        while True:

            # Get next line from file
            line = f.readline()

            # if line is empty
            # end of file is reached
            if not line:
                break
            # print("Line{}: {}".format(count, line.strip()))
            fn, st, sp, cl, _ = line.strip().split(' ')
            # if fn == '00010418_s016_t006': # (pyst: nedc_load_edf): failed to open
            #     continue
            count += 1
            st, sp = float(st), float(sp)
            # print (fn, st, sp, cl)
            fn_full = [name for name in all_filenames if fn in name]
            # print (fn_full)
            if len(fn_full) == 1:
                fn_full = fn_full[0].strip()
                print(fn_full)
                try:
                    fsamp, data = read_edf_elec(fn_full,parameters = "params_TUH_ECG.txt")
                    #fsamp, data = read_edf_elec(fn_full,parameters = "params_RPA_common_electrodes.txt")
                except:
                    print('can not read',fn_full )
                    with open('dev_ica_EEG_wrong.txt', 'a') as f1:
                        f1.write(fn_full+'\n')
                        f1.close()
                        continue

                print(fsamp, data.shape)

                # resample to 250 if sampling rate is higher
                if fsamp > 250:
                    print('Resampling data from {} to 250 Hz'.format(fsamp))
                    data = resample(data, int(data.shape[1] * 250.0 / fsamp), axis=1)

                i = 0
                chs = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']
                while (st + i) * 250 + window_len < sp * 250:
                    s = data[:, int((st + i) * 250): int((st + i) * 250) + window_len]

                    
                    # detect if signal is interupted, e.g., all dc, overflow
                    if detect_interupted_data(s.transpose(), 250):
                        print ('BAD DATA DETECTED! Skipping this {}-second segment due to interupted signals...'.format(segement))
                        i+=12
                        continue
                    else:
                        print ('GOOD DATA!')
                    
                    #raw = create_mne_raw(s, fsamp, chs)
                    #raw.plot(block=True, scalings=50e-6, remove_dc=True, lowpass=70, title='Raw - 0.5-70 Hz')
                

                    ica_filt_s = ica_arti_remove(s, 250, chs)

                    if ica_filt_s is None:
                        print ('Skipping this {}-second segment due to failed ICA...'.format(segement))
                        i+=12
                        continue

                    if cl == "seiz":
                        # train setting
                        # i+=1
                        # dev setting
                        i+=12
                    else:
                        # train setting
                        # i+=6
                        # dev setting
                        i+= 12


                    assert s.shape[1] == window_len

                    prep_s = calc_stft(ica_filt_s)
                    print('stft shape', prep_s.shape)
                    prep_fn = '{}/{}_{}_{}_{}.npy'.format(cachedir, fn, i, cl, st)


                    assert prep_s.shape == (1, 2*segement-1, 19, 125)
                    np.save(prep_fn, prep_s)

def get_ref_df_RPA(ref_test_files,all_files, cachedir,segement=None):
    print('using segement 12s ', cachedir)
    # window_len = 250 * segement  # 12 seconds
    with open(all_files, 'r') as f:
        all_filenames = f.readlines()
    print(len(all_filenames))
    # print (all_filenames[-1])
    count = 0
    f_prev='none'
    with open(ref_test_files, 'r') as f:
        while True:

            # Get next line from file
            line = f.readline()

            # if line is empty
            # end of file is reached
            if not line:
                break
            # print("Line{}: {}".format(count, line.strip()))
            fn, st, sp, cl, _ = line.strip().split(' ')
            # if fn == '00010418_s016_t006': # (pyst: nedc_load_edf): failed to open
            #     continue


            count += 1
            if sp!='END':
                st, sp = float(st), float(sp)
                # print (fn, st, sp, cl)
            else:
                st= float(st)

            if fn!=f_prev:
                fn_full = [name for name in all_filenames if fn in name]
                print (fn_full)
                if len(fn_full) == 1:
                    fn_full = fn_full[0].strip()
                    print(fn_full)

                    try:
                        fsamp, data = read_edf_elec(fn_full,parameters = "params_RPA_addECG.txt")
                        print(fsamp, data.shape)
                    except:
                        print('can not read', fn_full)
                        with open('RPA_ECG_wrong_1.txt', 'a') as f1:
                            f1.write(fn_full + '\n')
                            f1.close()
                            continue
                    # resample to 250 if sampling rate is higher
                    # if fsamp > 250:
                    #     print('Resampling data from {} to 250 Hz'.format(fsamp))
                    #     data = resample(data, int(data.shape[1] * 250.0 / fsamp), axis=1)

            else:
                print('same file')

            if sp != 'END':
                end = sp * fsamp
            else:
                end = data.shape[1]

            i = 0

            window_len = int(fsamp * segement)  # 12 seconds
            chs = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']
            while (st + i) * fsamp + window_len < end:
                s = data[:19, int((st + i) * fsamp): int((st + i) * fsamp) + window_len]
                s_eeg = data[19:, int((st + i) * fsamp): int((st + i) * fsamp) + window_len]*500

                # s_ap, chs_ap_ = convert_AP_montage(seg, chs)
                # chs_ap_ = [ch for ch in chs_ap_]
                
            
                

                # detect if signal is interupted, e.g., all dc, overflow
                if detect_interupted_data(s.transpose(), fsamp):
                    print ('BAD DATA DETECTED! Skipping this {}-second segment due to interupted signals...'.format(segement))
                    i+=12
                    continue
                else:
                    print ('GOOD DATA!')
                
                # raw = create_mne_raw(s, fsamp, chs)        
                # raw.plot(block=True, scalings=50e-6, remove_dc=True, lowpass=70, title='Raw - 0.5-70 Hz')
            

                ica_filt_s = ica_arti_remove(s, fsamp, chs)

                if ica_filt_s is None:
                    print ('Skipping this {}-second segment due to failed ICA...'.format(segement))
                    i+=12
                    continue

               
                # raw_ica = create_mne_raw(ica_filt_s, fsamp, chs)        
                # raw_ica.plot(block=True, scalings=50e-6, remove_dc=True, lowpass=70, title='ICA-denoised - 0.5-70')

                # resample to 250 if sampling rate is higher
                ica_filt_s = resample(ica_filt_s, int(ica_filt_s.shape[1] * 250.0 / fsamp), axis=1)
                s_eeg = resample(s_eeg, int(s_eeg.shape[1] * 250.0 / fsamp), axis=1)

                if cl == "seiz":
                    # train setting
                    #i+=1
                    # dev setting
                    i+=12
                else:
                    # train setting
                    #i+=6
                    # dev setting
                    i += 12

                #print(s.shape)
                assert s.shape[1] == window_len

                prep_s = calc_stft(ica_filt_s)
                ECG = calc_stft(s_eeg)
                #print(prep_s.shape, ECG.shape)
                prep_s = np.concatenate(np.concatenate([prep_s, ECG], axis=2))
                prep_s = np.expand_dims(prep_s,axis=0)
                print('stft shape', prep_s.shape)
                prep_fn = '{}/{}_{}_{}_{}.npy'.format(cachedir, fn, i, cl, st)
                print ('save to {}'.format(prep_fn))

                assert prep_s.shape == (1, 2*segement-1, 20, 125)
                np.save(prep_fn, prep_s)

            f_prev = fn


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--segement", type=int, default="12",help="12,60")
    parser.add_argument("--dataset", default="TUH",
                        help="EpilepsiaSurf, TUH, RPA")
    parser.add_argument("--ICA", default="None",
                        help="ICA,None")
    args = parser.parse_args()

    with open('SETTINGS_%s.json' % args.dataset) as f:
        settings = json.load(f)

    all_files = settings['all_files']



    if args.dataset =="TUH":
        # if args.ICA =="ICA":
        #     makedirs(settings['cachedir_ecg'] + str(args.segement) + 's')
        #     cachedir_train = settings['cachedir_ecg'] + str(args.segement) + 's'
        #     #all_files = settings['all_files_ICA']
        #     #make for the dev dataset
        #     makedirs(settings['cachedir_dev_ICA'] + str(args.segement) + 's')
        #     cachedir_dev = settings['cachedir_dev_ICA'] + str(args.segement) + 's'
        #     # make for the ecg dataset
        #
        # else:
        # makedirs(settings['cachedir_ecg'] + str(args.segement) + 's')
        # cachedir_train = settings['cachedir_ecg'] + str(args.segement) + 's'
        makedirs(settings['cachedir_dev'] + str(args.segement) + 's')
        cachedir_dev = settings['cachedir_dev'] + str(args.segement) + 's'

        # train file
        ref_train_file = settings['ref_train_file']
        # dev file
        ref_dev_file = settings['ref_dev_file']


        get_ref_train_df_TUH(ref_dev_file, all_files, cachedir_dev,segement=args.segement)

    if args.dataset =="RPA":
        year='2016'
        targets = [
            #'0',
            #'1',
             # '2',
             #    '3',
              #'4',
            #  '5',
            #    '6',
            #  '7',
            #'8',
                #'9',
               # '10',
            #  '11',
            #'12',
            #'13',
           # '14',
                '15',
                #'16',
             #'17',
                #'18',
                 #'19',
                 #'20',
            #'21',
            #'22',
              #'23',
               #'24',
              #'25',
               #'26',
              #'27',
              #'28',
             # '29',
            # '30',
             #'31'
        ]

        #targets = ['22']
        for target in targets:
            makedirs(settings["cachedir_randomECG"])
            makedirs(settings["cachedir_randomECG"]+'/' +year +'_'+ str(args.segement) + 's')
            makedirs(settings["cachedir_randomECG"]+'/' +year +'_' + str(args.segement) + 's' + '/pat'+target)
            cachedir = settings["cachedir_randomECG"]+'/' +year+'_' + str(args.segement) + 's' + '/pat'+ target
            ref_test_files = settings['ref_test_files']+'/ref_'+year+'/'+year+'_pat'+target+'.txt'
            get_ref_df_RPA(ref_test_files,all_files, cachedir, segement=args.segement)

