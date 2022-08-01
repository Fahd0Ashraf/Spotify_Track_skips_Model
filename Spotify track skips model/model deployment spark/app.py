import numpy as np
from flask import Flask,request ,jsonify,render_template
import pickle 

#create flask 
app = Flask(__name__)

#load pickle model 
model = pickle.load(open('E:/projects/technocolabs/model deployment spark/model.pkl','rb'))

#for haome page
@app.route("/")
def Home():
    return render_template("trial.html")

#for prediction from home page
@app.route("/predict",methods=['POST'])
def predict():
    float_feature = [float(x) for x in request.form.values()]
    features =[np.array(float_feature)]
    prediction = model.predict(features)
    if prediction > 0.5:
        assume = 'skipped'
    else: assume = 'not skipped'    
    return render_template("trial.html",prediction_text ="The track will be {}".format(assume))

if __name__ ==   "__main__":
    app.run(debug=True,use_reloader=False)
  