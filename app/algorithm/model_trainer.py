#!/usr/bin/env python

import os, warnings, sys
import pprint
warnings.filterwarnings('ignore') 

import numpy as np, pandas as pd

import algorithm.preprocessing.preprocess_utils as pp_utils
import algorithm.preprocessing.preprocessing_main as data_preprocessing

import algorithm.utils as utils
import algorithm.model.forecaster as forecaster
from algorithm.utils import get_model_config

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error



# get model configuration parameters 
model_cfg = get_model_config()


def get_trained_model(data, data_schema, hyper_params):      
    # set seeds
    utils.set_seeds(100)       
    # get preprocessing parameters
    pp_params = pp_utils.get_preprocess_params(data_schema)     
    # preprocess data (includes history and special_events if any)
    processed_history, processed_sp_events, data_preprocessor = preprocess_data(data, pp_params)    
    # print('processed_history \n', processed_history.head())    
    # Train model  - this will also do the preprocessing specific to the model
    model = train_model(processed_history, processed_sp_events, pp_params, hyper_params) 
    
    return model, data_preprocessor

    

def preprocess_data(data, pp_params):   
    print("Preprocessing data...")
    history = data["history"] ; sp_events = data["sp_events"]  
        
    data_preprocessor = data_preprocessing.DataPreprocessor(
            pp_params,  
            has_sp_events=True if sp_events is not None else False
        )    
    history, sp_events = data_preprocessor.fit_transform(history=history, sp_events=sp_events) 
     
    # del history["__exog__missing__"]
    return history, sp_events, data_preprocessor  



def train_model(processed_history, processed_sp_events, pp_params, hyper_params):       
    
    pprint.pprint(pp_params)
    
    model = forecaster.Forecaster(
        run_type="multi",
        series_id_cols=[pp_params["location_field"], pp_params["item_field"]],
        additional_regressors=[model_cfg["missing_val_field"]]
        ) 
    
    model.fit(
        history = processed_history, 
        sp_events=processed_sp_events,
        verbose=True
        )
    
    # -----------------------------------------------------
    # testing
    # forecast = model.predict(processed_history)    
    # print(forecast.tail())
    # print(processed_history.tail())
    # sys.exit()
    # -----------------------------------------------------
        
    return model
    



def save_training_artifacts(model, data_preprocessor, model_artifacts_path):   
    # save data_preprocessor
    data_preprocessing.save_data_preprocessor(data_preprocessor, model_artifacts_path)      
    # save the model artifacts
    forecaster.save_model(model, model_artifacts_path)