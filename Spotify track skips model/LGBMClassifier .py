import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.decomposition import PCA
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score




tfdata=pd.read_csv(r'E:\projects\technocolabs\major project\tf_mini.csv')
logdata=pd.read_csv(r'E:\projects\technocolabs\major project\log_mini.csv')

combineddata = logdata.merge(tfdata,left_on='track_id_clean',right_on='track_id')

sns.countplot(combineddata.time_signature,hue = combineddata.skip_2)
#users slightly tend to listen to the track rather than skipping it if it has a 
#time signature of 4

sns.histplot(x=combineddata.duration,bins=100,hue = combineddata.skip_2)
#users tend to skip songs more if they exceed 250 seconds

sns.histplot(x=combineddata.acousticness,bins=10,hue = combineddata.skip_2)
#no.skips are bigger in low accoustics value

#sns.catplot(y='duration',x='context_type',data=combineddata,hue = 'skip_2',col='premium',alpha=0.5)
plt.xticks(rotation = 45)
#intersting insights 
"""
FOR PREMIUM USERS IN THE CONTEXT TYPE MORE SKIPS HAPPEN IN USER COLLECTIONS AND
CHARTS BUT LESS SKIPS IN EDITORIAL PLAYLISTS,CATALOG AND PERSONALIZED PLAYLIST
HOW EVER FOR NON PREIMIUM USERS MORE SKIPS HAPPEN IN CATALOG,RADIO AND CHARTS 
WHILE LESS SKIPS HAPPEN IN EDITORIAL PLAYLIST USER COLLECTION
"""
#sns.catplot(y='duration',x='hist_user_behavior_is_shuffle',data=combineddata,hue = 'skip_2',col='premium',alpha=0.5)

'''
FOR PREUMIUM USERS THEY TEND TO SKIP MORE ON SHUFFLE AND LESS WHEN NOT SHUFFLING BUT
FOR NON PREMIUM USERSTHEY TEND TO SKIP WHEN THEY ARE NOT SHUFFLING MORE THAN THEY RE ON SHUFFLE
'''

subset_data = combineddata.head()
data1 = combineddata
#we figure out that we dont need three skips so we drop them and revert the 
#not skipped col
combineddata['is_skipped'] = ~combineddata.not_skipped
combineddata=combineddata.drop(['skip_1','skip_2','skip_3','not_skipped',
                                'session_id','track_id'],axis=1)

#check the corelation between the attributes and the new target feature
corrleation = combineddata.corr()
#no correlation between attributes and is skipped feature 

#transfering boolen data types into int
boolen_dtypes=['hist_user_behavior_is_shuffle','premium','is_skipped']
combineddata[boolen_dtypes]=combineddata[boolen_dtypes].astype(int)

#onehot encoding for categorical variables 
cols = ['context_type','hist_user_behavior_reason_start','hist_user_behavior_reason_end']
combineddata = pd.get_dummies(combineddata,columns =cols )

#label encoding the mode attribute and fixing some attributes 
lencoder = LabelEncoder()
combineddata['mode']=lencoder.fit_transform(combineddata['mode'])

combineddata['duration']=combineddata['duration']/60 #convert to minutes
combineddata['us_popularity_estimate']=combineddata['us_popularity_estimate']/10 #make a scale from 1 to 10
combineddata['released_from']=2022 - combineddata['release_year']

#compacting the 3 pause attributes into 1 
combineddata['is_paused_before_play']= ~(combineddata['no_pause_before_play']
                                          .astype(bool))
combineddata['is_paused_before_play']=combineddata['is_paused_before_play'].astype(int)
combineddata=combineddata.drop(['no_pause_before_play','short_pause_before_play',
                   'long_pause_before_play','release_year'],axis = 1)

#double checking session position,length,HourOfDay,date before droping 
sns.barplot(y=combineddata.is_skipped,x=combineddata.session_position)
sns.barplot(y=combineddata.is_skipped,x=combineddata.session_length,ci=None)
sns.barplot(y=combineddata.is_skipped,x=combineddata.hour_of_day,ci=None)
#this plots shows that there is no relation between session length, session position,
#date and skips
combineddata.date=pd.to_datetime(combineddata.date)
daysofweek =[]
for i in combineddata.date:
    daysofweek.append(pd.Timestamp(i).day_name())
combineddata['daysofweek']=daysofweek
sns.barplot(y=combineddata.is_skipped,x=combineddata.daysofweek,ci=None)
#this shows that least songs are listened on monday but is this a valid suggestion
#we check how many songs are actually played on weekdays
sns.countplot(combineddata.daysofweek,hue=combineddata.is_skipped)
#days of the week might be a usefull attribute so we keep it and drop the rest
combineddata=combineddata.drop(['session_length','session_position','date',
                                'hour_of_day'],axis=1)
combineddata.daysofweek=lencoder.fit_transform(combineddata.daysofweek)

#checking corrleation one last time 
corrleation = combineddata.corr()
#found correlations between is skipped and the usage of fwd button


# new data pulled

new_columns =['track_id_clean','context_switch','is_paused_before_play','hist_user_behavior_is_shuffle',
              'hist_user_behavior_n_seekfwd','hist_user_behavior_n_seekback',
              'released_from','premium',"is_skipped"]
old_columns = ['track_id_clean','session_length','session_position','context_type',
               'hist_user_behavior_reason_start','hist_user_behavior_reason_end',
               'hour_of_day']
new_data = combineddata[new_columns]
new_data1= data1[old_columns]
data_14_input = pd.concat([new_data,new_data1],axis = 1)

data_14_input['context_type'] = lencoder.fit_transform(data_14_input['context_type'])
data_14_input['hist_user_behavior_reason_start'] = lencoder.fit_transform(data_14_input['hist_user_behavior_reason_start'])
data_14_input['hist_user_behavior_reason_end'] = lencoder.fit_transform(data_14_input['hist_user_behavior_reason_end'])
data_14_input_y = data_14_input.is_skipped

data_14_input.drop(['track_id_clean','is_skipped'],axis = 1,inplace = True)


target = combineddata.is_skipped
combineddata=combineddata.drop(['is_skipped','track_id_clean'],axis=1)
#implementing pca
#we can plot the explained variance to the dimensions to help us choose 
pca = PCA()
pca.fit(combineddata)
cumsum= np.cumsum(pca.explained_variance_ratio_)
sns.lineplot(data=cumsum)
#approxmitly we achieve 98% accuracy at 3 diminsions 

pca = PCA(n_components=0.98)
PCAdata = pd.DataFrame(pca.fit_transform(combineddata))
#using pca.explained_variance_ratio_ we find that our 4 components has 98% of
#our data, considering that we reduced from 59 cols to 4

#we can plot the explained variance to the dimensions to help us choose 

#model impelementation 
#for less input data write data_14_input, for normal write combined data

x = data_14_input
y = data_14_input_y

trainx ,valx,trainy,valy = train_test_split(x,y,random_state =42)


n_estimator = 50
model1 = LGBMClassifier(max_depth =2,learning_rate = 0.1,n_estimators=n_estimator)
model1.fit(trainx,trainy)
pred = model1.predict(valx)
accuracy1 = accuracy_score(valy,pred)


#bi lstm RNN
#data to import imto the RNN notebook
combineddata.to_csv(r'E:\projects\technocolabs\major project\data.csv')
PCAdata.to_csv(r'E:\projects\technocolabs\major project\PCAdata.csv')
target.to_csv(r'E:\projects\technocolabs\major project\target.csv')

data_14_input.to_csv(r'E:\projects\technocolabs\major project\modified_data.csv',index_label = False)
data_14_input_y.to_csv(r'E:\projects\technocolabs\major project\modified_target.csv',index_label = False)





