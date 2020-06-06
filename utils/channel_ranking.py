import numpy as np
from sklearn.ensemble import RandomForestClassifier
from scipy.signal import resample
from pyst import read_edf_elec
import glob
    
ref_train_file = '/mnt/data4/datasets/seizure/TUH_EEG_Seizure/v1.5.1/_DOCS/ref_train.txt'
all_files = './all_files.txt'
cachedir = '/mnt/data7_M2/tempdata/tuh_1s_fft'
def get_fft_train(ref_train_file, all_files, cachedir):
    window_len = 250 # 1 second
    with open(all_files, 'r') as f:
        all_filenames = f.readlines() 
    print (len(all_filenames))
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
                print (fn_full)
                fsamp, data = read_edf_elec(fn_full)
                print (fsamp, data.shape)
                
                # resample to 250 if sampling rate is higher
                if fsamp > 250:
                    print ('Resampling data from {} to 250 Hz'.format(fsamp))
                    data = resample(data, int(data.shape[1]*250/fsamp), axis=1)
                i=0
                while (st+i)*250+window_len < sp*250: 
                    s = data[:, int((st+i)*250) : int((st+i)*250)+window_len]
                    print ('Raw time-series shape', s.shape)
                    i+=1
                    prep_s = np.fft.rfft(s, axis=-1)
                    print (prep_s.shape)
                    prep_s = prep_s[:,1:48]
                    prep_fn = '{}/{}_{}_{}_{}.npy'.format(cachedir,fn,i,cl,st)                    
                    print ('Preprocessed shape', prep_s.shape)
                    np.save(prep_fn, prep_s)
                    
def load_training_data(datadir):
    data_fns = glob.glob(datadir + '/*.npy')
    X = []
    y = []
    for fn in data_fns:
        x_ = np.load(fn) # shape (19,47)        
        X.append(x_)     

        # Store class           
        label = fn.strip().split('_')
        label = label[-2]
        if label=='seiz':
            y.append(1)
        elif label == 'bckg':
            y.append(0)
    return np.array(X), np.array(y)

def get_channels_importance(X,y):        
    
    importances_agg = [0.0]*X.shape[1]
        
    clf = RandomForestClassifier(n_estimators=300, min_samples_split=2, bootstrap=False, n_jobs=4, random_state=0)
    clf.fit(X.reshape(-1,X.shape[1]*X.shape[2]),y)
    X_test = X[::100]
    score = clf.score(X_test.reshape(-1,X_test.shape[1]*X_test.shape[2]),y[::100])
    print ('Score', score)
    importances = clf.feature_importances_.reshape(X.shape[1],X.shape[2])
    #print np.sum(importances,axis=1)
    importances_agg += np.sum(importances,axis=1)	
    print (importances_agg)			
    return importances_agg.argsort()[::-1]

if __name__ == "__main__":
    # get_fft_train(ref_train_file, all_files, cachedir) #(20, 47)
    X, y = load_training_data(cachedir)
    channels_ranking = get_channels_importance(X, y)
    print (channels_ranking)