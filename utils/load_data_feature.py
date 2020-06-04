import numpy as np


def read_raw(path, chan_num):
    dt = np.dtype([('row', np.int32), ('feature1', np.float32), ('feature2', np.float32),
                   ('feature3', np.float32), ('feature4', np.float32),
                   ('feature5', np.float32), ('feature6', np.float32), ('feature7', np.float32),
                   ('feature8', np.float32),
                   ('feature9', np.float32), ('feature10', np.float32), ('feature11', np.float32),
                   ('feature12', np.float32),
                   ('feature13', np.float32), ('feature14', np.float32), ('feature15', np.float32),
                   ('feature16', np.float32),
                   ('feature17', np.float32), ('feature18', np.float32), ('feature19', np.float32),
                   ('feature20', np.float32),
                   ('feature21', np.float32), ('feature22', np.float32), ('feature23', np.float32),
                   ('feature24', np.float32),
                   ('feature25', np.float32), ('feature26', np.float32)
                   ])
    shape = np.fromfile(path, dtype=np.int32, count=3)
    if chan_num!=shape[0]:
        print(shape[0])
        chan_num = shape[0]
    rawData = np.fromfile(path, dtype=dt, offset=8)
    rawData = rawData.reshape(chan_num, -1)
    rawData = np.array(rawData.tolist())[:, :, 1:]

    print(shape)
    print(rawData.shape)
    return rawData

def get_ref_train_df(ref_train_file, all_files, cachedir):
    window_len = 30
    with open(all_files, 'r') as f:
        all_filenames = f.readlines()
    print(len(all_filenames))
    # print (all_filenames[-1])
    count = 0
    with open(ref_train_file, 'r') as f:
        while True:
            count += 1

            # Get next line from file
            line = f.readline()

            # if line is empty
            # end of file is reached
            if not line:
                break
            # print("Line{}: {}".format(count, line.strip()))
            fn, st, sp, cl, _ = line.strip().split(' ')
            st, sp = float(st), float(sp)
            # print (fn, st, sp, cl)
            fn_full = [name for name in all_filenames if fn in name]
            # print (fn_full)
            if len(fn_full) == 1:
                fn_full = fn_full[0].strip()
                print(fn)
                print(fn_full)
                if '03_tcp_ar_a' in fn_full:
                    print('this one is ar_a which only 20 channels')
                    chan_num = 20
                    rawData = read_raw(fn_full, chan_num)

                    i = 0
                    while ((st+i)*10 + window_len) < sp * 10:
                        s = rawData[:,int((st+i)*10):int((st+i)*10+window_len),:]
                        i += 1
                        # prep_s = calc_stft(s)
                        prep_fn = '{}/{}_{}_{}_{}.npy'.format(cachedir, fn, i, cl, st)
                        #print('Raw time-series shape', s.shape)
                        # print ('Preprocessed shape', prep_s.shape)
                        np.save(prep_fn, s)
                    print(i)
                    if cl == "seiz":
                        for i_a in range(2):
                            if st - i_a - 1 >= 0:
                                s = rawData[:20, int((st - i_a - 1) * 10): int((st - i_a - 1) * 10) + window_len,:]
                                print('Additional raw time-series shape', st - i_a - 1, s.shape,
                                      int((st - i_a - 1) * 10), int((st - i_a - 1) * 10) + window_len)
                                assert s.shape[1] == window_len
                                prep_fn = '{}/{}_{}_{}.npy'.format(cachedir, fn, -i_a - 1, cl)

                                print('Additional preprocessed shape', s.shape)
                                np.save(prep_fn, s)
                else:
                    print('this one is ar_a which 22 channels')
                    chan_num = 22
                    rawData = read_raw(fn_full, chan_num)

                    i = 0
                    while ((st+i)*10 + window_len) < sp * 10:
                        s = rawData[:20, int((st+i)*10):int((st+i)*10+window_len), :]
                        i += 1
                        # prep_s = calc_stft(s)
                        prep_fn = '{}/{}_{}_{}_{}.npy'.format(cachedir, fn, i, cl, st)
                        #print('only use 20')
                        #print('Raw time-series shape', s.shape)
                        # print ('Preprocessed shape', prep_s.shape)
                        np.save(prep_fn, s)
                    print(i)
                    if cl == "seiz":
                        for i_a in range(2):
                            if st - i_a - 1 >= 0:
                                s = rawData[:20, int((st - i_a - 1) * 10): int((st - i_a - 1) * 10) + window_len,:]
                                print('Additional raw time-series shape', st - i_a - 1, s.shape,
                                      int((st - i_a - 1) * 10), int((st - i_a - 1) * 10) + window_len)
                                assert s.shape[1] == window_len
                                prep_fn = '{}/{}_{}_{}.npy'.format(cachedir, fn, -i_a - 1, cl)

                                print('Additional preprocessed shape', s.shape)
                                np.save(prep_fn, s)

            else:
                print('can not find or find more')
                print(fn_full)
                print(fn)

def get_ref_train_test(ref_train_file, all_files, cachedir):
    # print("Line{}: {}".format(count, line.strip()))
    line = '00009455_s003_t004 0.0000 212.0000 bckg 1.0000'
    fn, st, sp, cl, _ = line.strip().split(' ')
    st, sp = float(st), float(sp)
    # print (fn, st, sp, cl)
    fn_full = ['/mnt/data4/datasets/seizure/TUH_EEG_Seizure/v1.5.1/feats/train/01_tcp_ar/094/00009455/s003_2012_10_16/00009455_s003_t004.raw']
    # print (fn_full)
    if len(fn_full) == 1:
        fn_full = fn_full[0].strip()
        print(fn)
        print(fn_full)
        if '03_tcp_ar_a' in fn_full:
            print('this one is ar_a which only 20 channels')
            chan_num = 20
            rawData = read_raw(fn_full, chan_num)

            i = 0
            while (st * 10 + i) < sp * 10:
                s = rawData[:, int(st * 10 + i), :]
                i += 1
                # prep_s = calc_stft(s)
                prep_fn = '{}/{}_{}_{}_{}.npy'.format(cachedir, fn, i, cl, st)
                # print('Raw time-series shape', s.shape)
                # print ('Preprocessed shape', prep_s.shape)
                np.save(prep_fn, s)
            print(i)
        else:
            print('this one is ar_a which 22 channels')
            chan_num = 22
            rawData = read_raw(fn_full, chan_num)

            i = 0
            while (st * 10 + i) < sp * 10:
                s = rawData[:20, int(st * 10 + i), :]
                i += 1
                # prep_s = calc_stft(s)
                if i ==1899:
                    print(s)
                prep_fn = '{}/{}_{}_{}_{}.npy'.format(cachedir, fn, i, cl, st)
                # print('only use 20')
                # print('Raw time-series shape', s.shape)
                # print ('Preprocessed shape', prep_s.shape)
                np.save(prep_fn, s)
            print(i)
    else:
        print('can not find or find more')
        print(fn_full)
        print(fn)


if __name__ == "__main__":
    ref_train_file = '/mnt/data4/datasets/seizure/TUH_EEG_Seizure/v1.5.1/_DOCS/ref_train.txt'
    #ref_train_file ='/home/yikai/code/competition/szpred_tuh-master 2/utils/ref_train.txt'
    all_files = './all_feature_files.txt'
    #cachedir = '/mnt/data7_M2/tempdata/tuh_feature_1s'
    cachedir = '/mnt/data7_M2/tempdata/tuh_feature'

    #cachedir = '/Users/yikai/Documents/szpred_tuh-master 2/test_data/feature'
    get_ref_train_df(ref_train_file, all_files, cachedir)
    #get_ref_train_test(ref_train_file, all_files, cachedir)
    #train1: begin ---'00009455_s003_t004
    #train2: 00009455_s003_t005--01_tcp_ar/131/00013145/s004_2015_09_01/00013145_s004_t006
    #teain3: ------00013112_s002_t001_00013112_s002_t001_7271_bckg_0.0
    #train4:--end