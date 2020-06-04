import numpy as np

file1 ='/Users/yikai/Documents/result_final_train2_eval/overlap/3s_0.97_5s0.97.txt'
file2 ='/Users/yikai/Documents/result_final_train2_eval/7s/result_eval_2ch_f3f7_7s0.97.txt'
file_new = '/Users/yikai/Documents/result_final_train2_eval/overlap/3s_0.97_5s0.97_7s_0.97.txt'
with open(file1, 'r') as f:
    while True:
        # Get next line from file
        line = f.readline()
        if not line:
            break
        fn, st, sp, cl,channel = line.split(' ')

        with open(file2, 'r') as f2:
            while True:
                line2 = f2.readline()
                if not line2:
                    break
                fn2, st2, sp2, cl2,channel = line2.split(' ')
                if fn2==fn:
                    st_new = max(float(st2),float(st))
                    sp_new = min(float(sp2),float(sp))
                    if sp_new > st_new:
                        cl_new = np.mean([float(cl2),float(cl)])
                        with open(file_new, 'a') as f3:
                            f3.write(fn+' '+str(st_new)+' '+str(sp_new)+' '+str(cl_new)+' '+channel)






