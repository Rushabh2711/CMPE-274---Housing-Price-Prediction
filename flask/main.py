from flask import  Flask,render_template,request
# from flask_cors import CORS,cross_origin
import pickle
import predictfunctions
import json
import logging
from flask import jsonify

# with open('./xgb.pkl', 'rb') as f:
#     xgb = pickle.load(f)
# with open('./DataForML.pkl', 'rb') as f:
#     datforml = pickle.load(f)

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
        prediction_from_api=predictfunctions.FunctionGeneratePrediction(
                                                    inp_LSTAT=LSTAT_value,
                                                    inp_RM=RM_value,
                                                    inp_PTRATIO=PTRATIO_value
                                                )
        print(prediction_from_api)
        return (prediction_from_api)

    except Exception as e:
        return('Something is not right!:'+str(e))
    
if __name__ =="__main__":
    app.run(host='0.0.0.0', port=80)