
# coding: utf-8

# In[208]:
def clean(file_in, file_out):
    ans = []
    with open(file_in) as fp:
        Lines = fp.readlines()
        for line in Lines:
            ele = line.split(" ")
            ele[1]=float(ele[1])
            ele[2]=float(ele[2])
            ans.append(ele)


# In[210]:

    for d in range(len(ans)-1):
        count = 1
        while d+count<len(ans)-1 and ans[d+count][0]==ans[d][0]:
            if ans[d][2] > ans[d+count][1]:
                ans[d][2] = ans[d+count][2]
                del ans[d+count]
            else:
                count+=1


# In[217]:

    with open(file_out, 'a') as the_file:
        for i in ans:
            the_file.write(i[0]+' '+str(i[1])+' '+str(i[2])+' '+i[3]+" "+i[4])



# In[ ]:
if __name__ == "__main__":
    file_in = '/Users/yikai/Desktop/test_new/aaaa.txt'
    file_out = '/Users/yikai/Desktop/test_new/bbbbbb.txt'
    clean(file_in,file_out)



