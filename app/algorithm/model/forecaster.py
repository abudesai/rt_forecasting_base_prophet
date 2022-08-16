
import numpy as np, pandas as pd
import math
import joblib
import json
import sys
import os, warnings
from prophet import Prophet
# Multi-processing
from multiprocessing import Pool, cpu_count
# Process bar
from tqdm import tqdm
from time import time

import logging
logging.getLogger('fbprophet').setLevel(logging.ERROR)


MODEL_NAME = "forecaster_base_Prophet"

model_fname = "model.save"

class Forecaster(): 
    
    def __init__(self, run_type, series_id_cols, additional_regressors):
        self.series_id_cols = series_id_cols
        self.run_type = run_type
        self.models = {}
        self.uniq_ids = None
        self.additional_regressors = additional_regressors
        self.print_period = 5
    
    
    def fit(self, history, sp_events, verbose=False):
                
        groups_by_ids = history.groupby(self.series_id_cols)        
        all_ids = list(groups_by_ids.groups.keys())        
        for id_ in tqdm(all_ids): 
            history_sub = groups_by_ids.get_group(id_).drop(columns=self.series_id_cols)            
            self._fit_on_series(
                key=id_,
                history=history_sub,
                sp_events=sp_events
            )
                
        
    def _fit_on_series(self, key, history, sp_events): 
        m = Prophet(holidays=sp_events)   
        
        for regressor in self.additional_regressors:            
            m.add_regressor(regressor)     
            
        m.fit(history)        
        
        self.models[key] = m
        
        
    def predict(self, future_df):        
        groups_by_ids = future_df.groupby(self.series_id_cols)        
        all_ids = list(groups_by_ids.groups.keys())        
        all_forecasts=[]
        for id_ in tqdm(all_ids): 
            future_df_sub = groups_by_ids.get_group(id_).drop(columns=self.series_id_cols)[['ds'] + self.additional_regressors]
            forecast = self._predict_on_series(
                key=id_,
                future_df=future_df_sub
            )
            
            for i, id_col in enumerate(self.series_id_cols):
                forecast.insert(i, id_col, id_[i])
            
            all_forecasts.append(forecast)

        forecast = pd.concat(all_forecasts, axis=0, ignore_index=True)
        return forecast
    
    
    def _predict_on_series(self, key, future_df): 
        forecast = self.models[key].predict(future_df)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        return forecast


def save_model(model, model_path):     
    joblib.dump(model, os.path.join(model_path, model_fname))  
    

def load_model(model_path): 
    model = joblib.load(os.path.join(model_path, model_fname))
    return model