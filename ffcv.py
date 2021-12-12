import pandas as pd
from sklearn.utils import shuffle
def ffold():
    dataset = pd.read_csv('data_set.csv')
    dataset = dataset[dataset.corona_result != 'other']
    list1=[]
    final=[]
    dataset = shuffle(dataset)
    # now we have a unordered dataset!
    ksize = int(len(dataset)/5) #indicates how many rows for each part
    j = 0
    k = ksize
    for i in range(0,5): #in this loop we divide our data frame into a list of 5 dataframes with 800 rows
        list1.append(dataset.iloc[j:k])
        j+=ksize
        k+=ksize
            
    j=0
    k=0 #now we will create a list which contains 5 list of train and test datasets
    temp = pd.DataFrame()
    for j in range(0,5):
        list2=[]
        for k in range(0,5):
            if j==k :
                list2.append(list1[k])
            else:
                temp = temp.append(list1[k], ignore_index = True)
        list2.append(temp)
        temp = pd.DataFrame()
        final.append(list2) # final list looks like this : [[first fold, rest],[2nd fold, rest],...]
    return final
#note that shuffle will work everytime we call the function again! which means we will-
#not get the same dataframes everytime we run a program which use this function
