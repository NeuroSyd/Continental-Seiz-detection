import glob
import argparse

def generate_file_list(root_dir, fn):
    with open (fn, "w+") as f:        
        for filename in glob.iglob(root_dir + '**/*.raw', recursive=True):
            # print(filename)
            f.write(filename + "\n")
        


if __name__ == "__main__":   
    #data_dir = "/mnt/data4/datasets/seizure/TUH_EEG_Seizure/v1.5.1/feats/"
    data_dir = "/Users/yikai/Documents/szpred_tuh-master 2/test_data//"
    fn = "./all_feature_files1.txt"
    generate_file_list(data_dir, fn)