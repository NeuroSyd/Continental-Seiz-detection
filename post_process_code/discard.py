

def account_number():

    file1 = '/Users/yikai/Documents/result_dev2/post_process/post_3s_0.99_5s_0.99_7s_0.99_v3.txt'
    file_new = '/Users/yikai/Documents/result_dev2/post_process/post_3s_0.99_5s_0.99_7s_0.99_v4.txt'

    with open(file1, 'r') as f:
        line = f.readlines()
        print(len(line))
    with open(file_new, 'r') as f1:
        line1 = f1.readlines()
        print(len(line1))

def discard():
    file1 = '/Users/yikai/Desktop/final_code /results/result_eval_dataset/overlap/3s_0.98_5s0.98_7s_0.98.txt'
    file_new = '/Users/yikai/Desktop/test_new/aaaa.txt'

    with open(file1, 'r') as f:
        # Get next line from file
        line = f.readlines()
        # print(line[-1])
        #print(line[0].split(' '))
        fn0, st0, sp0, cl0,channel = line[0].split(' ')
        with open(file_new, 'a') as f3:
            f3.write(fn0 + ' ' + str(st0) + ' ' + str(sp0) + ' ' + str(cl0)+' '+channel)
        for i in range(1,len(line)):

            fn_new, st_new, sp_new, cl_new,channel = line[i].split(' ')
            if fn_new==fn0:
                if float(st_new)-float(sp0)>10:
                    if float(sp_new)-float(st_new)>5:
                        with open(file_new, 'a') as f3:
                            f3.write(fn_new + ' ' + str(st_new) + ' ' + str(sp_new) + ' ' + str(cl_new)+' '+channel)
                else:
                    if float(sp_new) - float(st_new) > 5 and float(sp0) - float(st0) > 5:
                        with open(file_new, 'a') as f3:
                            f3.write(fn_new + ' ' + str(st0) + ' ' + str(sp_new) + ' ' + str(cl_new)+' '+channel)
            else:
                if float(sp_new) - float(st_new) > 5:
                    with open(file_new, 'a') as f3:
                        f3.write(fn_new + ' ' + str(st_new) + ' ' + str(sp_new) + ' ' + str(cl_new)+' '+channel)
            fn0 = fn_new
            st0 = st_new
            sp0 = sp_new
            cl0 = cl_new

if __name__ == "__main__":
    discard()
