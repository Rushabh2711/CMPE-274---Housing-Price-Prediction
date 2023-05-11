
import pandas as pd
import pickle
### Sandardization of data ###
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# This Function can be called from any from any front end tool/website
def FunctionPredictResult(InputData):
    
    Num_Inputs=InputData.shape[0]
    
    # Making sure the input data has same columns as it was used for training the model
    # Also, if standardization/normalization was done, then same must be done for new input
    
    # Appending the new data with the Training data
    DataForML=pd.read_pickle('DataForML.pkl')
    # InputData=InputData.concat(DataForML)
    InputData = pd.concat([DataForML,InputData], axis=0)
    # Generating dummy variables for rest of the nominal variables
    InputData=pd.get_dummies(InputData)
            
    # Maintaining the same order of columns as it was during the model training
    Predictors=['LSTAT', 'RM', 'PTRATIO']
    
    # Generating the input values to the model
    X=InputData[Predictors].values[0:Num_Inputs]
    
    # Choose between standardization and MinMAx normalization
    #PredictorScaler=StandardScaler()
    PredictorScaler=MinMaxScaler()

    # Storing the fit object for later reference
    PredictorScalerFit=PredictorScaler.fit(X)

    # Generating the standardized values of X since it was done while model training also
    X=PredictorScalerFit.transform(X)
    
    # Loading the Function from pickle file
    # import pickle
    with open('xgb.pkl', 'rb') as fileReadStream:
        PredictionModel=pickle.load(fileReadStream)
        # Don't forget to close the filestream!
        fileReadStream.close()
            
    # Genrating Predictions
    Prediction=PredictionModel.predict(X)
    PredictionResult=pd.DataFrame(Prediction, columns=['Prediction'])
    print("Here is the rsult")
    print(PredictionResult)
    return(PredictionResult)

# Creating the function which can take inputs and return prediction
def FunctionGeneratePrediction(inp_LSTAT , inp_RM, inp_PTRATIO):
    print("In this function: FunctionGeneratePrediction")
    # Creating a data frame for the model input
    SampleInputData=pd.DataFrame(
     data=[[inp_LSTAT , inp_RM, inp_PTRATIO]],
     columns=['LSTAT', 'RM', 'PTRATIO'])

    # Calling the function defined above using the input parameters
    Predictions=FunctionPredictResult(InputData= SampleInputData)
    print(Predictions)
    # Returning the predictions
    return(Predictions.to_json())

# # Function call
# FunctionGeneratePrediction( inp_LSTAT=4.98,
#                            inp_RM=6.5,
#                            inp_PTRATIO=15.3
#                              )
