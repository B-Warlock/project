import pandas as pd
from ffcv import ffold
from statistics import mean


dataframes = ffold() #geting the dataframes from ffold

                                        #start of train phase
step = 1   #step counter
recall_lst = []
precision_lst = []
fmeasure_lst = []
for inputdf in dataframes: 
    print(" step no. ",step, "\n")
    traindataframe = inputdf[1] #getting train data
    testdataframe = inputdf[0] # getting test data
    traindataframe.drop(traindataframe.columns[0], axis=1, inplace = True) #we dropped first col of dataset which we do not need
    testdataframe.reset_index(inplace = True) # we should reset the index, its not ordered but added 1 col with old indexes
    testdataframe.drop(testdataframe.columns[[0,1]], axis=1, inplace = True) #dropped first and second col which we do not need
    data=[]
    maj=[]
    t_err = {}         #for total_error of each feature
    cols = traindataframe.columns
    for column in cols:     #filling the null values with nan so we can count them
        traindataframe[column] = traindataframe[column].fillna("nan")
    clmns = testdataframe.columns
    for column in clmns:
        testdataframe[column] = testdataframe[column].fillna("nan")
        
                                            # creating a basic table for each feature
        
        
    for col in cols:
        test = traindataframe.groupby([col])['corona_result'].value_counts(dropna = False)
        df = pd.DataFrame(columns=['feature', 'val', 'count', 'maj', 'err', 'class'])
        ft = []
        val = []
        clss = []
        for i in test.index.values:    #here we create a list of each row values.
            ft.append(col)
            val.append(i[0])
            clss.append(i[1])
        df['count'] = test.values      #fill the dataset rows with lists
        df['feature'] = ft
        df['val'] = val
        df['class'] = clss
        maj.append(df.groupby('val', as_index = False)['count'].max())
        data.append(df)               #pushing each feature table in a list
    maj.pop()# we don't need The last elements of both maj and data so we pop them
    data.pop()
    
    
                                    #end of creating basic table for features now we have to calculate the majority class and err
        
                                    #start of filling maj column
    for i in range(0,len(data)): 
        mj = maj[i]
        df = data[i]
        for j in range(0, len(mj.index)):
            maj_cls = df[df['count'] == mj.loc[j,'count']]['class'].values[0]    #calc the class of max count
            maj_val = mj[mj['count'] == mj.loc[j,'count']]['val'].values[0]      #calc the value of max count 
            indx = df.loc[df['val'] == maj_val,['maj']].index.values      #calc the right index for assignment
            df.iloc[indx,3]= maj_cls        #assign
            
                                    #end of majority class assignment
                                    
                                    #start of total_err calculation
                                    
    for i in range(0,len(data)):
        count_list = []
        df = data[i]
        mj = maj[i]
        sum_all= df['count'].sum()    #The denominator of the total error of feature is calculated 
        cnt = pd.Series(dtype = 'float64')
        cnt = df.loc[df['maj'] != df['class']]['count'].sum() #sum of the frequency of wrong predictions
        fname = df['feature'].values[0]
        t_err[fname] = round(cnt/sum_all*100, 4)   #saving total Err of feature in a dictionary (in percent)
        
                                    #end of total err calc.
                                    
                                    #start of calculating error for each value of a feature.
                                    
    for i in range(0,len(data)):
        mj = maj[i]
        df = data[i]
        sum_err= df.groupby('val', as_index = False)['count'].sum() #The denominator of the error is calculated
        for k in range(0,len(sum_err)):       #now we calc err for each value-count and assign it to its cell
            se_val = sum_err.loc[k, 'val']
            se_count = sum_err.loc[k, 'count']
            for j in range(0, len(df)):
                if df.loc[j, 'val'] == se_val:    #if there is same val btw sum_err and df dataframes
                    err = int(df.loc[j, 'count'])/int(se_count)     #calc count/se_count
                    df.loc[j, 'err'] =  err       #error in percent
                                  #end of calculating error for each value of a feature.
                                 #selection of best feature to make rules and creating a rule table.

    rule_feature = min(t_err, key = t_err.get)#choosing the best rule
    print("The Rule is Based On : ", rule_feature, " with Total ERR of ", t_err[rule_feature]," % \n")
    for i in data:
        if i.loc[0,'feature'] == rule_feature:
            rule_tbl = pd.DataFrame(columns = ['value', 'class'])
            rule_val = []                  #creating a rule table
            rule_val = pd.unique(i['val']).tolist()
            rule_tbl['value'] = rule_val    #setting the values from feature table
            rule_cls = []
            for j in rule_tbl['value']:
                r_max = i[i['val'] == j]['count'].max()
                rule_cls.append(i[i['count'] == r_max]['class'].values[0]) #fetching right class for value
            rule_tbl['class'] = rule_cls
    print("RULE TABLE : \n")          #rule table is complete!!
    print(rule_tbl , '\n')
    
                                        #calculation of SUPPORT and CONFIDENCE
    
    for df in data:
        if df['feature'].values[0] == rule_feature:
            print("Confidence for each value in order : \n")
            for i in df[df['maj'] == df['class']]['err']: #confidence is the err cell of major frequency of each value in basic table
                print(i)               #F(x,y)/F(x)
            print("\n")
            print("Support for each value in order : \n")
            for j in df[df['maj'] == df['class']]['count']:
                print(j/df['count'].sum())  #F(x,y)/N
            print("\n")
            
                                           # start of test phase
            
            
    test_tbl = pd.DataFrame(columns = ['val','result', 'clss'])
    test_tbl['val'] = testdataframe[rule_feature] # filling the val column with data we need from test dataset
    test_tbl['clss'] = testdataframe['corona_result'] # filling table with right lables from dataset
    for j in range(0, len(rule_tbl)):
        for i in range(0, len(test_tbl)):
            if test_tbl.loc[i,'val'] == rule_tbl.loc[j,'value']:
                test_tbl.loc[i,'result'] = rule_tbl.loc[j, 'class']   #CLASSIFICATION
    
                                    #calculation of TP, TN, FP, FN AND RECALL , PRECISION , F_MEASURE
    TP = test_tbl[(test_tbl['result'] == test_tbl['clss']) & (test_tbl['clss'] == 'positive')]['clss'].count()
    TN = test_tbl[(test_tbl['result'] == test_tbl['clss']) & (test_tbl['clss'] == 'negative')]['clss'].count()
    FP = test_tbl[(test_tbl['result'] != test_tbl['clss']) & (test_tbl['result'] == 'positive')]['clss'].count()
    FN = test_tbl[(test_tbl['result'] != test_tbl['clss']) & (test_tbl['result'] == 'negative')]['clss'].count()
    print("True Positive : ", TP , "\nTrue Negative : ", TN ,"\nFalse Positive : ", FP , "\nFalse Negative : ", FN ,"\n")
    recall = TP / (TP + FN)
    recall_lst.append(recall)
    precision = TP / (TP + FP)
    precision_lst.append(precision)
    f_measure = (2*precision*recall) / (precision + recall)
    fmeasure_lst.append(f_measure)
    print("Recall : ", recall,"\nprecision : ", precision,"\nf_measure : ", f_measure , "\n-------------------------------------------------------------\n")
                                    #end of of TP, TN, FP, FN AND RECALL , PRECISION , F_MEASURE calculation
    step+=1      
print("**************************End of Test And Train Operation***************************** \n")
print("Recall AVG : ", mean(recall_lst), "\nprecision AVG : ", mean(precision_lst), "\nf_measure AVG : ", mean(fmeasure_lst),"\n")
