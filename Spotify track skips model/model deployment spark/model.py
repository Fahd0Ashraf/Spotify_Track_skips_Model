import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
import pickle
import re

data = pd.read_csv('E:/projects/technocolabs/model deployment spark/modified_data.csv')
target = pd.read_csv('E:/projects/technocolabs/model deployment spark/modified_target.csv')



data = data.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
trainx ,valx,trainy,valy = train_test_split(data,target,random_state =42)


model = LGBMClassifier(max_depth =2,learning_rate = 0.1,n_estimators=150)
model.fit(trainx,trainy)

#make pickle file of model 

pickle.dump(model,open('E:/projects/technocolabs/model deployment spark/model.pkl','wb'))