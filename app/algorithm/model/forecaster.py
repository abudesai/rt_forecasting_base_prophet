
import numpy as np, pandas as pd
import joblib
import sys
import os, warnings
from prophet import Prophet
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


MODEL_NAME = "forecaster_base_Prophet"

model_fname = "model.save"

class Forecaster(): 
    
    def __init__(self, run_type, series_id_cols, additional_regressors, 
                 uncertainty_samples, interval_width, n_changepoints, seasonality_mode):
        self.series_id_cols = series_id_cols
        self.run_type = run_type
        self.models = {}
        self.uniq_ids = None
        self.additional_regressors = additional_regressors
        self.uncertainty_samples = uncertainty_samples
        self.interval_width = interval_width
        self.n_changepoints = n_changepoints
        self.seasonality_mode = seasonality_mode
        
        
        self.sp_events = None
        
        self.multi_min_count = 10
        self.print_period = 5
        self.max_cpus_to_use = 6
    
    def fit(self, history, sp_events, verbose=False):
                
        groups_by_ids = history.groupby(self.series_id_cols)        
        all_ids = list(groups_by_ids.groups.keys()) 
        all_series = [groups_by_ids.get_group(id_).drop(columns=self.series_id_cols)  
                      for id_ in all_ids]   
        
        self.sp_events = sp_events
        
        if self.sp_events is not None:  
            print(self.sp_events.head())
            self.sp_events['lower_window'] = - self.sp_events['lower_window']
        
        if self.run_type == "sequential" or len(all_ids) <= self.multi_min_count:
            for id_, series in tqdm(zip(all_ids, all_series)):        
                model = self._fit_on_series( history=series )
                self.models[id_] = model
                
        elif self.run_type == "multi":            
            # Spare 2 cpus if we have many, but use at least 1 and no more than 6.
            cpus = max(1, min(cpu_count()-2, self.max_cpus_to_use))
            print(f"Multi training with {cpus=}" )
            p = Pool(cpus)
            models = list(tqdm(p.imap(self._fit_on_series, all_series)))            
            
            for i, id_ in enumerate(all_ids):
                self.models[id_] = models[i]         
        else: 
            raise ValueError(f"Unrecognized run_type {self.run_type}. Must be one of ['sequential', 'multi'] ")
        
        
    def _fit_on_series(self, history): 
        model= Prophet(holidays=self.sp_events, 
                uncertainty_samples=self.uncertainty_samples,
                interval_width=self.interval_width,
                n_changepoints=self.n_changepoints,
                seasonality_mode=self.seasonality_mode,
            )    
               
        for regressor in self.additional_regressors:            
            model.add_regressor(regressor)      
                                
        model.fit(history)          
            
        return model
        
                
    def predict(self, future_df):        
        groups_by_ids = future_df.groupby(self.series_id_cols)        
        all_ids = list(groups_by_ids.groups.keys())   
        all_series = [ groups_by_ids.get_group(id_).drop(columns=self.series_id_cols)[['ds'] + self.additional_regressors]
                        for id_ in all_ids]   
        
        all_forecasts=[]  
        
        # for some reason, multi-processing takes longer! So use single-threaded. 
        for id_, series in tqdm(zip(all_ids, all_series)):   
            forecast = self._predict_on_series(key_and_future_df = (id_, series) )  
            if forecast is None: continue 
            for col_idx, id_col in enumerate(self.series_id_cols):
                forecast.insert(col_idx, id_col, id_[col_idx])             
            all_forecasts.append(forecast)
        
        # multi-processing logic - disabled because it takes much longer
        # if self.run_type == "sequential" or len(all_ids) <= self.multi_min_count:                              
        #     for id_, series in tqdm(zip(all_ids, all_series)):   
        #         forecast = self._predict_on_series(key_and_future_df = (id_, series) )                
        #         all_forecasts.append(forecast)
        # elif self.run_type == "multi":
        #     # Spare 2 cpus if we have many, but use at least 1 and no more than 6.
        #     cpus = max(1, min(cpu_count()-2, self.max_cpus_to_use))
        #     print(f"Multi predictions with {cpus=}", )
        #     p = Pool(cpus)
        #     all_forecasts = list(tqdm(p.imap(self._predict_on_series, zip(all_ids, all_series))))
        # else: 
        #     raise ValueError(f"Unrecognized run_type {self.run_type}. Must be one of ['sequential', 'multi'] ")
     
        # for i, id_ in enumerate(all_ids): 
        #     forecast = all_forecasts[i]
        #     if forecast is not None: 
        #         for col_idx, id_col in enumerate(self.series_id_cols):
        #             forecast.insert(col_idx, id_col, id_[col_idx])
                
        
        all_forecasts = pd.concat(all_forecasts, axis=0, ignore_index=True)
        all_forecasts['yhat'] = all_forecasts['yhat'].round(4)
        return all_forecasts
    
    
    def _predict_on_series(self, key_and_future_df): 
        key, future_df = key_and_future_df
        if self.models.get(key) is not None: 
            forecast = self.models[key].predict(future_df)            
            df_cols_to_use = ['ds', 'yhat', 'yhat_lower', 'yhat_upper']
            cols = [ c for c in df_cols_to_use if c in forecast.columns]
            forecast = forecast[cols]
        else: 
            # no model found - indicative of key not being in the history, so cant forecast for it. 
            forecast = None
        return forecast


def save_model(model, model_path):     
    joblib.dump(model, os.path.join(model_path, model_fname))  
    

def load_model(model_path): 
    model = joblib.load(os.path.join(model_path, model_fname))
    return model