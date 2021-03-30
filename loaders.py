import pandas as pd
import numpy as np
from pytorch_forecasting import TimeSeriesDataSet 

class PVData:
    def __init__(self, file_name: str):
        self.solar_df = pd.read_parquet(file_name).reset_index()
        ## Add time anchors as additional features to the dataset
        self.solar_df['date'] = pd.DatetimeIndex( self.solar_df['start_of_quarter'] ).date
        self.solar_df['month'] = pd.DatetimeIndex( self.solar_df['start_of_quarter'] ).month
        self.solar_df['year'] = pd.DatetimeIndex( self.solar_df['start_of_quarter'] ).year
        self.solar_df['week'] = pd.DatetimeIndex( self.solar_df['start_of_quarter'] ).isocalendar().week.reset_index()['week']
        self.solar_df['day_of_year'] = pd.DatetimeIndex( self.solar_df['start_of_quarter'] ).dayofyear
        ## bins that map day of year to seasons
        bins = [0, 91, 183, 275, 366]
        labels=['Winter', 'Spring', 'Summer', 'Fall']
        doy = pd.DatetimeIndex( self.solar_df['start_of_quarter'] ).dayofyear
        ## Add seasons as an attribute
        self.solar_df['seasons'] = np.array( pd.cut( (doy + 11)- 366*(doy > 355) , bins=bins, labels=labels) )
        ## Group consecutive days and add an index
        self.solar_df['time_idx'] = self.solar_df.groupby( pd.Grouper(key="start_of_quarter", freq="2D") ).cumcount()

    
    def scale_power_output(self, scaler: float):
        # Normalize the power outpt values between 0 and 1
        self.solar_df['power'] = self.solar_df['power'] * scaler
    
    def get_examples(self, 
        year: int, 
        encoder_length: dict,
        prediction_length: dict) -> TimeSeriesDataSet:
        
        solar_df_filtered = self.solar_df[ (self.solar_df['year'] == year) ]
        group_length =  2 * encoder_length['max'] - 1
        num_groups = int(np.floor( solar_df_filtered.shape[0] / group_length ))
        solar_df_filtered = solar_df_filtered[1 : (num_groups * group_length + 1)]
        solar_df_filtered['group'] = np.repeat(np.arange(num_groups), group_length)
        examples = TimeSeriesDataSet(
        solar_df_filtered,
        group_ids=["group"],
        target="power",
        time_idx="time_idx",
        min_encoder_length=encoder_length['min'],
        max_encoder_length=encoder_length['max'],
        min_prediction_length=prediction_length['min'],
        max_prediction_length=prediction_length['max'],
        time_varying_unknown_reals=["power"],
        time_varying_known_reals=["cloudcover_low","cloudcover_mid","cloudcover_high"],
        time_varying_known_categoricals=["seasons"],
        allow_missings=True,)

        return examples
    



