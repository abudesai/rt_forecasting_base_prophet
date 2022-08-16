import numpy as np, pandas as pd
import os
import sys

import algorithm.utils as utils
import algorithm.model.forecaster as forecaster
import algorithm.preprocessing.preprocessing_main as preprocessing
import algorithm.preprocessing.preprocess_utils as pp_utils


# get model configuration parameters 
model_cfg = utils.get_model_config()


class ModelServer:
    def __init__(self, model_path, data_schema): 
        self.model_path = model_path
        self.preprocessor = None
        self.model = None 
        self.pp_params = pp_utils.get_preprocess_params(data_schema)    
    
    
    def _get_preprocessor(self): 
        if self.preprocessor is None: 
            self.preprocessor = preprocessing.load_data_preprocessor(self.model_path)
            return self.preprocessor           
        else: return self.preprocessor
    
    
    def _get_model(self): 
        if self.model is None: 
            self.model = forecaster.load_model(self.model_path)
            return self.model
        else: return self.model
        
       
    def predict(self, data, ):    
        print("Making predictions ...")
                   
        preprocessor = self._get_preprocessor()
        model = self._get_model()   
        
        if preprocessor is None:  raise Exception("No preprocessor found. Did you train first?")
        # if model is None:  raise Exception("No model found. Did you train first?")
        
        # extract history and sp_events data
        history = data["history"] ; sp_events = data["sp_events"]  
            
        # transform data - returns tuple of X (array of word indexes) and y (None in this case)
        history, sp_events = preprocessor.transform(history, sp_events) 
                
        forecast_df = model.predict( history )
        
        forecast_df = self._postprocess_forecast_df(forecast_df=forecast_df)        
        return forecast_df
    
    
    def _postprocess_forecast_df(self, forecast_df):        
        forecast_df = self.preprocessor.inverse_transform_forecast_df(forecast_df)
        return forecast_df
        
