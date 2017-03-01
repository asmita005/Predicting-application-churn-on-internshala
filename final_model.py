import numpy as np
import pandas as pd
from sklearn.cross_validation import StratifiedKFold
from sklearn import ensemble, preprocessing, cross_validation
from sklearn import metrics 



if __name__ == '__main__':
    train = pd.read_csv('../input/train.csv',parse_dates= 'Earliest_Start_Date')#,nrows = 300)
    test = pd.read_csv('../input/test.csv',parse_dates= 'Earliest_Start_Date' )#,nrows = 300)
    internship = pd.read_csv('../input/Internship.csv',parse_dates= ['Start_Date','Internship_deadline'])#,nrows = 300)
    student = pd.read_csv('../input/student.csv')
    

    internship  = internship[internship.columns[0:13]]
    train = train.merge(internship, on = 'Internship_ID' ,how = 'left')
    test = test.merge(internship, on = 'Internship_ID' ,how = 'left')
    
    int_id = test.Internship_ID
    st_id = test.Student_ID
    
    
    

    col = ['Student_ID', 'Institute_Category', 'Institute_location', 'hometown', 'Degree', 'Stream', 'Current_year','Year_of_graduation', 'Performance_PG', 'PG_scale','Performance_UG', 'UG_Scale', 'Performance_12th','Performance_10th']
    
    c = student.groupby(col).agg('count').reset_index()    
    
    student = c[col]
    
    train = train.merge(student ,on = 'Student_ID', how = 'left' )
    test = test.merge(student ,on = 'Student_ID', how = 'left' )

   

    text_columns = []

    for f in train.columns:
        if train[f].dtype=='object':
            if f != 'loca':    
                text_columns.append(f)            
                lbl = preprocessing.LabelEncoder()
                lbl.fit(list(train[f].values) + list(test[f].values))
                train[f] = lbl.transform(list(train[f].values))
                test[f] = lbl.transform(list(test[f].values))
     
         
    train.replace(np.nan , -1 ,inplace = True)
    test.replace(np.nan , -1 ,inplace = True)
    
    target = train.Is_Shortlisted.values
    train = train.drop(['Is_Shortlisted','Earliest_Start_Date','Start_Date','Internship_deadline'],axis =1) 
    test = test.drop(['Earliest_Start_Date','Start_Date','Internship_deadline'],axis =1)
    k = train.columns
    
    
    
    gbm1 = ensemble.GradientBoostingClassifier(random_state = 42, learning_rate = 0.07 ,  min_samples_split = 10,subsample = 0.9 , max_features = 'sqrt' ,  n_estimators = 350 , max_depth =  6)               
    gbm1.fit(train,target)
    prob1 = gbm1.predict_proba(test)[:,1]
    
    
    
    gbm2 = ensemble.RandomForestClassifier(bootstrap=False, class_weight='auto', criterion='entropy',max_depth = None, max_features='sqrt', max_leaf_nodes=None,
            min_samples_leaf=1, min_samples_split=10,
            min_weight_fraction_leaf=0.0, n_estimators = 500, n_jobs=-1,
            oob_score=False, random_state=57, verbose=0,
            warm_start=False)
    gbm2.fit(train,target)
    prob2 = gbm2.predict_proba(test)[:,1]
    
    
    gbm3 = ensemble.GradientBoostingClassifier(random_state = 42, learning_rate = 0.07 ,  min_samples_split = 10,subsample = 0.9 , max_features = 'sqrt' ,  n_estimators = 350 , max_depth =  4)               
    gbm3.fit(train,target)
    prob3 = gbm3.predict_proba(test)[:,1]    
    
    
    
    prob = 0.6 * prob1 + 0.25 * prob2 + 0.15 *prob3
    
    
    submission = pd.DataFrame({'Internship_ID':int_id, 'Student_ID':st_id, 'Is_Shortlisted':prob})
    submission.to_csv('final_model.csv',index = False)
