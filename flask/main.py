from flask import  Flask,render_template,request
# from flask_cors import CORS,cross_origin
import pickle
import json
import logging
from flask import jsonify
import pandas as pd
### Sandardization of data ###
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# with open('./xgb.pkl', 'rb') as f:
#     xgb = pickle.load(f)
# with open('./DataForML.pkl', 'rb') as f:
#     dataforml = pickle.load(f)

app = Flask(__name__)
# cors = CORS(app, resources={r"/predict/*": {"origins": "http://127.0.0.1"}})

if app.debug is not True:   
    from logging.handlers import RotatingFileHandler
    file_handler = RotatingFileHandler('info.log', maxBytes=1024 * 1024 * 100, backupCount=20)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)
    app.logger.addHandler(file_handler)

@app.route("/",methods=['GET','POST'])
def home():
    return render_template("genrepredict.html")

@app.route("/predict/",methods=['GET','POST'])
# @cross_origin()
def predict():
    try:
    # Getting the paramters from API call
        LSTAT_value = float(request.args.get('lstat'))
        RM_value=float(request.args.get('rooms'))
        PTRATIO_value=float(request.args.get('ptratio'))
        print(LSTAT_value,RM_value,PTRATIO_value)

        # Calling the funtion to get predictions
        print("In this function: FunctionGeneratePrediction")
        # Creating a data frame for the model input
        # print(inp_LSTAT,inp_RM,inp_PTRATIO)
        SampleInputData=pd.DataFrame(
        data=[[LSTAT_value , RM_value, PTRATIO_value]],
        columns=['LSTAT', 'RM', 'PTRATIO'])
        InputData = SampleInputData

        Num_Inputs=InputData.shape[0]
        # Making sure the input data has same columns as it was used for training the model
        # Also, if standardization/normalization was done, then same must be done for new input
        
        # Appending the new data with the Training data
        scaler=pd.read_pickle('scalerNew.pkl')
        # with open('./DataForML.pkl', 'rb') as f:
        #     DataForML = pickle.load(f)
        print(scaler)
        
        # Maintaining the same order of columns as it was during the model training
        Predictors=['LSTAT', 'RM', 'PTRATIO']
        
        # Generating the input values to the model
        X=InputData[Predictors].values[0:Num_Inputs]
        print(X.shape)
        # Choose between standardization and MinMAx normalization
        #PredictorScaler=StandardScaler()
        PredictorScaler=scaler

        # # Storing the fit object for later reference
        # PredictorScalerFit=PredictorScaler.fit(X)

        # Generating the standardized values of X since it was done while model training also
        X=scaler.transform(X)
        # Calling the function defined above using the input parameters
        # Predictions=predict.predict(model, InputData= SampleInputData)
        with open('./rfr.pkl', 'rb') as f:
            rfr = pickle.load(f)
        Prediction=rfr.predict(X)
        PredictionResult=pd.DataFrame(Prediction, columns=['Prediction'])
        print("Here is the result")
        # print(PredictionResult)
        print(PredictionResult)
        # Returning the predictions
        return(PredictionResult.to_json())
        # prediction_from_api=predictfunctions.FunctionGeneratePrediction(model=xgb,
        #                                             inp_LSTAT=LSTAT_value,
        #                                             inp_RM=RM_value,
        #                                             inp_PTRATIO=PTRATIO_value
        #                                         )
        # print(prediction_from_api)
        # return (prediction_from_api)

    except Exception as e:
        return('Something is not right!:'+str(e))
    
if __name__ =="__main__":
    app.run(host='0.0.0.0', port=80)