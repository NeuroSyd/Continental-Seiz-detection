import numpy as np
import glob
import os

from utils.pyst import read_edf,read_edf_elec
from scipy.signal import resample
from sklearn import preprocessing
from utils.load_data_feature import read_raw
from scipy.signal import resample
import stft



def predict(predictions1,threhold):
    for i in range(len(predictions1)):
        if predictions1[i]>=threhold:
            predictions1[i] =int(1)
        else:
            predictions1[i] =int(0)

    return predictions1

def write(predictions,file_name,threhold):

    print('begin vote predciction')

    begin = predictions[0]
    index_begin = 0
    print('begin write ' + file_name)
    for i in range(len(predictions)):
        if i == len(predictions) - 1:
            index_end = (i + 1)
            print('---------')
            print(index_begin, index_end)
            write_file(file_name, index_begin, index_end, begin,threhold)
        if predictions[i] != begin:
            index_end = i
            print('---------')
            print(index_begin, index_end)
            write_file(file_name, index_begin, index_end, begin,threhold)
            begin = predictions[i]
            index_begin = i
    print('end write ' + file_name)

def vote_wirte(predictions3,predictions5,predictions7,file_name,threhold):

    print('begin vote predciction')

    predictions = []
    for m in range(len(predictions3)):
        predictions.append(np.bincount([predictions3[m], predictions5[m], predictions7[m]]).argmax())
    begin = predictions[0]
    index_begin = 0
    print('begin write ' + file_name)
    for i in range(len(predictions)):
        if i == len(predictions) - 1:
            index_end = (i + 1)
            print('---------')
            print(index_begin, index_end)
            write_file(file_name, index_begin, index_end, begin,threhold)
        if predictions[i] != begin:
            index_end = i
            print('---------')
            print(index_begin, index_end)
            write_file(file_name, index_begin, index_end, begin,threhold)
            begin = predictions[i]
            index_begin = i
    print('end write ' + file_name)


def write_file(file_name, index_begin, index_end, begin,threhold):
    f = open("results_final/result_eval_2ch_f3f7_3s"+str(threhold)+".txt", 'a')

    if begin == 1:
        seiz_name = 'seiz'
        confidence = threhold
        print(file_name + ' ' + str(round(index_begin, 4)) + ' ' + str(round(index_end, 4)) + ' ' + str(
            round(confidence, 4))+' '+str(2)
              + '\n')
        f.write(file_name + ' ' + str(round(index_begin, 4)) + ' ' + str(round(index_end, 4)) + ' ' + str(
            round(confidence, 4))+' '+str(2)
                + '\n')

    f.close()


def calc_stft(s_):
    # s_time = begin_rec + timedelta(seconds=int(i*numts))
    s = s_.transpose()

    stft_data = stft.spectrogram(s,framelength=250,centered=False)
    stft_data = np.transpose(stft_data,(1,2,0))
    stft_data = np.abs(stft_data)+1e-6

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


def get_ref_evalu_stft(single_path,space_time):
    window_len = 250 * space_time  # 7 seconds
    #single_path='/mnt/data4/datasets/seizure/TUH_EEG_Seizure/v1.5.1/edf/train/03_tcp_ar_a/131/00013145/s003_2015_09_01/00013145_s003_t000.edf'
    fsamp, data = read_edf_elec(single_path)
    if fsamp > 250:
        print('Resampling data from {} to 250 Hz'.format(fsamp))
        data = resample(data, int(data.shape[1] * 250.0 / fsamp), axis=1)
    print('--------------------')
    print(data.shape)
    time = data.shape[1]/250
    number = time-space_time+1
    i=0
    print(number)
    s = np.empty([int(number), 2*space_time-1,2,125,1])
    while (i + space_time) <= time:
        data1 = data[:,int(i*250) : int(i*250)+window_len]
        assert data1.shape[1] == window_len
        diff1 = data1[3:4, :] - data1[2:3, :]
        # print(i,'done')
        diff2 = data1[13:14, :] - data1[17:18, :]
        data1 = np.concatenate((diff1, diff2), axis=0)
        s[i] = np.moveaxis(calc_stft(data1), 0, -1)
        i += 1
    print(i)
    print(s.shape)
    return s,int(time)



def get_ref_evalu_df(single_path):
    # single_path = os.path.join('./',single_path)
    fsamp, data = read_edf(single_path)
    # print(fsamp, data.shape)
    window_len = 250  # 1 seconds
    # resample to 250 if sampling rate is higher
    if fsamp > 250:
        print('Resampling data from {} to 250 Hz'.format(fsamp))
        data = resample(data, int(data.shape[1] * 250.0 / fsamp), axis=1)

    return data
def get_ref_evalu_lcc(single_path):
    rawData = read_raw(single_path, 20)
    data = rawData[:20,:,:]
    print(data[1])

    return data



def get_path(path):
    evaluate_list = []
    # path ='/mnt/data4/datasets/seizure/TUH_EEG_Seizure/v1.5.1/edf/train/'
    import os
    f = open("./name_feature.txt", 'w')
    for root, dirs, files in os.walk(path, topdown=False):
        # print(root)
        # print(root)
        # print(sorted(dirs))
        # print(sorted(files))
        for name in files:
            # print(name)
            dic = os.path.join(root, name)
            data = dic.split('.')

            # print(data[-1])
            if data[-1] == 'edf':
                # print(dic)
                # print(data[-1])
                evaluate_list.append(dic)
                f.write(dic + '\n')
    return evaluate_list


def extract_fn(raw_fn):
    base_fn = os.path.basename(raw_fn)
    fns = base_fn.strip().split('_')
    #print(fns)
    fns = '{}_{}_{}_{}'.format(fns[0], fns[1], fns[2], fns[5])
    #print(fns)
    #print(fns[:-4])
    return fns[:-4]


def get_image_path(train_dir):
    fl_sz = glob.glob(train_dir + "/*_seiz_*.npy")[:500]
    fl_bg = glob.glob(train_dir + "/*_bckg_*.npy")[:500]
    n_sz = len(fl_sz)
    n_bg = len(fl_bg)
    print('!{} seiz samples, {} bckg samples!'.format(n_sz, n_bg))
    fl_sz_uniq = [extract_fn(fn) for fn in fl_sz]
    fl_sz_uniq = list(set(fl_sz_uniq))

    fl_bg_uniq = [extract_fn(fn) for fn in fl_bg]
    fl_bg_uniq = list(set(fl_bg_uniq))

    print('!{} seiz files, {} bckg filess!'.format(len(fl_sz_uniq), len(fl_bg_uniq)))
    fl_sz_test = [fn for fn in fl_sz if extract_fn(fn) in fl_sz_uniq]
    fl_bg_test = [fn for fn in fl_bg if extract_fn(fn) in fl_bg_uniq]
    print('{} seiz training samples, {} bg training samples'.format(len(fl_sz_test), len(fl_bg_test)))

    #fl_test=fl_sz_test +fl_bg_test
    fl_test = fl_sz_test
    X = np.empty((len(fl_test), 1,19,750))
    y = np.empty((len(fl_test)), dtype=int)
    filename = []
    for i, ID in enumerate(fl_test):
        x_ = np.load(ID)  # shape (1, 20, 250)
        # print ('__data_generation', x_.shape)
        x_ = preprocessing.normalize(x_)
        X[i,] = np.expand_dims(x_, axis=0)
        label = ID.strip().split('_')

        Id = ID.split('/')[-1]
        if i==0:
            print(ID)
            print(ID.split('/'))
            print(Id)
        # label = label[-1].strip().split('.')[0]
        label = label[-2]

        if label == 'seiz':
            y[i] = 1
        elif label == 'bckg':
            y[i] = 0
        filename.append(Id)
    return X,y,filename

def train_val_get_file_list(train_dir, dev_dir):
    # Count number of seiz and bckg samples

    # this is train file
    fl_sz = glob.glob(train_dir + "/*_seiz_*.npy")
    fl_bg = glob.glob(train_dir + "/*_bckg_*.npy")
    n_sz = len(fl_sz)
    n_bg = len(fl_bg)
    print('!{} seiz samples, {} bckg samples!'.format(n_sz, n_bg))
    fl_sz_uniq = [extract_fn(fn) for fn in fl_sz]

    fl_sz_uniq = list(set(fl_sz_uniq))

    fl_bg_uniq = [extract_fn(fn) for fn in fl_bg]
    fl_bg_uniq = list(set(fl_bg_uniq))

    print('!{} seiz files, {} bckg filess!'.format(len(fl_sz_uniq), len(fl_bg_uniq)))

    fl_sz_uniq_train = fl_sz_uniq
    fl_bg_uniq_train = fl_bg_uniq

    fl_sz_train = [fn for fn in fl_sz if extract_fn(fn) in fl_sz_uniq_train]
    fl_bg_train = [fn for fn in fl_bg if extract_fn(fn) in fl_bg_uniq_train]

    print('{} seiz training samples, {} bg training samples'.format(len(fl_sz_train), len(fl_bg_train)))
    fl_train = fl_sz_train + fl_bg_train
    class_weights = {}
    class_weights[0] = len(fl_train) / (len(fl_bg_train) + 1e-6)
    class_weights[1] = len(fl_train) / (len(fl_sz_train) + 1e-6)

    # this is val file
    fl_sz_val = glob.glob(dev_dir + "/*_seiz_*.npy")
    fl_bg_val = glob.glob(dev_dir + "/*_bckg_*.npy")
    n_sz_val = len(fl_sz_val)
    n_bg_val = len(fl_bg_val)
    print('!{} val seiz samples, {} val bckg samples!'.format(n_sz_val, n_bg_val))
    fl_sz_uniq_val = [extract_fn(fn) for fn in fl_sz_val]
    fl_sz_uniq_val = list(set(fl_sz_uniq_val))

    fl_bg_uniq_val = [extract_fn(fn) for fn in fl_bg_val]
    fl_bg_uniq_val = list(set(fl_bg_uniq_val))

    print('!{} val seiz files, {} val bckg filess!'.format(len(fl_sz_uniq_val), len(fl_bg_uniq_val)))


    fl_sz_dev = [fn for fn in fl_sz_val if extract_fn(fn) in fl_sz_uniq_val]
    fl_bg_dev = [fn for fn in fl_bg_val if extract_fn(fn) in fl_bg_uniq_val]

    print('{} seiz training samples, {} bg training samples'.format(len(fl_sz_dev), len(fl_bg_dev)))

    # val_weighhted

    fl_sz_dev_dup = fl_sz_dev * int(len(fl_bg_dev) / len(fl_sz_dev))
    fl_dev = fl_sz_dev_dup + fl_bg_dev

    print('------after upsampling the minority class for val  ')
    print('{} seiz val samples, {} bg val samples'.format(len(fl_sz_dev_dup), len(fl_bg_dev)))
    np.random.shuffle(fl_train), np.random.shuffle(fl_dev)

    return fl_train, fl_dev,class_weights



def train_val_split_file_list(train_dir, val_ratio=0.25):
    # Count number of seiz and bckg samples
    #for i in range(len(train_dir)):
        #train_dir_=train_dir[i]
    fl_sz = glob.glob(train_dir + "/*_seiz_*.npy")
    fl_bg = glob.glob(train_dir + "/*_bckg_*.npy")
    n_sz = len(fl_sz)
    n_bg = len(fl_bg)
    print('!{} seiz samples, {} bckg samples!'.format(n_sz, n_bg))
    # down_spl = int(np.floor(n_bg / n_sz))
    # if down_spl > 1:
    # fl_bg = fl_bg[::down_spl]

    # print('downsample', down_spl)

    fl_sz_uniq = [extract_fn(fn) for fn in fl_sz]
    fl_sz_uniq = list(set(fl_sz_uniq))

    fl_bg_uniq = [extract_fn(fn) for fn in fl_bg]
    fl_bg_uniq = list(set(fl_bg_uniq))

    print('!{} seiz files, {} bckg filess!'.format(len(fl_sz_uniq), len(fl_bg_uniq)))

    fl_sz_uniq_train = fl_sz_uniq[:-int(val_ratio * len(fl_sz_uniq))]
    fl_sz_uniq_val = fl_sz_uniq[-int(val_ratio * len(fl_sz_uniq)):]

    fl_bg_uniq_train = fl_bg_uniq[:-int(val_ratio * len(fl_bg_uniq))]
    fl_bg_uniq_val = fl_bg_uniq[-int(val_ratio * len(fl_bg_uniq)):]

    fl_sz_train = [fn for fn in fl_sz if extract_fn(fn) in fl_sz_uniq_train]
    fl_bg_train = [fn for fn in fl_bg if extract_fn(fn) in fl_bg_uniq_train]

    fl_sz_val = [fn for fn in fl_sz if extract_fn(fn) in fl_sz_uniq_val]
    fl_bg_val = [fn for fn in fl_bg if extract_fn(fn) in fl_bg_uniq_val]

    print('{} seiz training samples, {} bg training samples'.format(len(fl_sz_train), len(fl_bg_train)))
    print('{} seiz val samples, {} bg val samples'.format(len(fl_sz_val), len(fl_bg_val)))
    fl_train = fl_sz_train + fl_bg_train
    # train weighted
    # weight_seiz = np.empty((750, 1), dtype=int)
    # weight_bckg = np.empty((750, 1), dtype=int)
    # for i in range(750):
    #weight_seiz=len(fl_train) / (len(fl_bg_train) + 1e-6)
    #weight_bckg=len(fl_train) / (len(fl_sz_train) + 1e-6)

    #class_weights_train = {}
    #class_weights_train[0] = weight_seiz
    #class_weights_train[1] = weight_bckg
    class_weights = {}
    class_weights[0] = len(fl_train) / (len(fl_bg_train) + 1e-6)
    class_weights[1] = len(fl_train) / (len(fl_sz_train) + 1e-6)
    # val_weighhted
    # fl_val = fl_sz_val + fl_bg_val
    # weight_seiz_val = np.empty((750, 1), dtype=int)
    # weight_bckg_val = np.empty((750, 1), dtype=int)
    # for i in range(750):
    # weight_seiz_val[i][0] = len(fl_val) / (len(fl_bg_val) + 1e-6)
    # weight_bckg_val[i][0] = len(fl_val) / (len(fl_sz_val) + 1e-6)

    # class_weights_val = {}
    # class_weights_val[0] = weight_seiz_val
    # class_weights_val[1] = weight_bckg_val
    # bcz class_weight did not support for three dimension so upsample for train
    #fl_sz_train_dup = fl_sz_train * int(len(fl_bg_train) / len(fl_sz_train))
    #fl_train = fl_sz_train_dup + fl_bg_train
    #print('------after upsampling the minority class for train  ')
    #print('{} seiz train samples, {} bg train samples'.format(len(fl_sz_train_dup), len(fl_bg_train)))

    # Because class_weight is only support during training, not for validation
    # we need to generate more minority class during validation manually
    fl_sz_val_dup = fl_sz_val * int(len(fl_bg_val) / len(fl_sz_val))
    fl_val = fl_sz_val_dup + fl_bg_val

    print('------after upsampling the minority class for val  ')
    print('{} seiz val samples, {} bg val samples'.format(len(fl_sz_val_dup), len(fl_bg_val)))
    np.random.shuffle(fl_train), np.random.shuffle(fl_val)

    return fl_train, fl_val,class_weights


def extract_fn_diff_patient(raw_fn):
    base_fn = os.path.basename(raw_fn)
    fns = base_fn.strip().split('_')
    #print(fns)
    fns = '{}'.format(fns[0])
    return fns





def train_val_split_diff_patient_file_list(train_dir, val_ratio=0.25):
    # Count number of seiz and bckg samples
    #for i in range(len(train_dir)):
        #train_dir_=train_dir[i]
    fl_sz = glob.glob(train_dir + "/*_seiz_*.npy")
    fl_bg = glob.glob(train_dir + "/*_bckg_*.npy")
    n_sz = len(fl_sz)
    n_bg = len(fl_bg)
    print('!{} seiz samples, {} bckg samples!'.format(n_sz, n_bg))
    # down_spl = int(np.floor(n_bg / n_sz))
    # if down_spl > 1:
    # fl_bg = fl_bg[::down_spl]

    # print('downsample', down_spl)

    fl_sz_uniq = [extract_fn_diff_patient(fn) for fn in fl_sz]
    fl_sz_uniq = list(set(fl_sz_uniq))

    fl_bg_uniq = [extract_fn_diff_patient(fn) for fn in fl_bg]
    fl_bg_uniq = list(set(fl_bg_uniq))
    #print(fl_bg_uniq)

    print('!{} seiz files, {} bckg filess!'.format(len(fl_sz_uniq), len(fl_bg_uniq)))

    fl_sz_uniq_train = fl_sz_uniq[:-int(val_ratio * len(fl_sz_uniq))]
    fl_sz_uniq_val = fl_sz_uniq[-int(val_ratio * len(fl_sz_uniq)):]

    fl_bg_uniq_train = fl_bg_uniq[:-int(val_ratio * len(fl_bg_uniq))]
    fl_bg_uniq_val = fl_bg_uniq[-int(val_ratio * len(fl_bg_uniq)):]

    fl_sz_train = [fn for fn in fl_sz if extract_fn_diff_patient(fn) in fl_sz_uniq_train]
    fl_bg_train = [fn for fn in fl_bg if extract_fn_diff_patient(fn) in fl_bg_uniq_train]

    fl_sz_val = [fn for fn in fl_sz if extract_fn_diff_patient(fn) in fl_sz_uniq_val]
    fl_bg_val = [fn for fn in fl_bg if extract_fn_diff_patient(fn) in fl_bg_uniq_val]

    print('{} seiz training samples, {} bg training samples'.format(len(fl_sz_train), len(fl_bg_train)))
    print('{} seiz val samples, {} bg val samples'.format(len(fl_sz_val), len(fl_bg_val)))
    fl_train = fl_sz_train + fl_bg_train
    # train weighted

    class_weights = {}
    class_weights[0] = len(fl_train) / (len(fl_bg_train) + 1e-6)
    class_weights[1] = len(fl_train) / (len(fl_sz_train) + 1e-6)

    # val_ weighted
    fl_sz_val_dup = fl_sz_val * int(len(fl_bg_val) / len(fl_sz_val))
    fl_val = fl_sz_val_dup + fl_bg_val

    print('------after upsampling the minority class for val  ')
    print('{} seiz val samples, {} bg val samples'.format(len(fl_sz_val_dup), len(fl_bg_val)))
    np.random.shuffle(fl_train), np.random.shuffle(fl_val)

    return fl_train, fl_val,class_weights


