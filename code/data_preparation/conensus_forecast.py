import os
import pandas as pd
import numpy as np
from datetime import datetime


class ConsensusForecast:
    def __init__(self, quarterly_file_path, monthly_file_path, output_file_path=None):
        self.quarterly_file_path = quarterly_file_path
        self.monthly_file_path = monthly_file_path
        self.output_file_path = output_file_path

    def get_consensus_forecast(self, country_var='US GDP'):
        quarterly_file_path_latest = fr'{self.quarterly_file_path}\long_term_consensus_forecasts_20250117.xlsx'
        monthly_file_path_latest = fr'{self.monthly_file_path}\consensus_forecasts_20250117.xlsx'

        if country_var == 'EA STR':
            country_var_m = 'DE STR'
            country_var_q = country_var
        else:
            country_var_m = country_var
            country_var_q = country_var

        df_monthly = pd.read_excel(monthly_file_path_latest, sheet_name=country_var_m, header=4, index_col=0)
        df_quarterly = pd.read_excel(quarterly_file_path_latest, sheet_name=country_var_q, header=4, index_col=0)

        max_forecast_year = df_quarterly.index.max()
        df_monthly.index = pd.to_datetime(df_monthly.index, errors='coerce', format='%d.%m.%Y')
        full_date_range = pd.date_range(start=df_monthly.index.min(), end=max_forecast_year, freq='YE')
        df_monthly_expanded = df_monthly.reindex(full_date_range)

        df_monthly_melted = df_monthly_expanded.reset_index().melt(
            id_vars='index', var_name='forecast_date', value_name='monthly_forecast'
        )
        df_monthly_melted.rename(columns={'index': 'forecasted_year'}, inplace=True)

        df_quarterly_melted = df_quarterly.reset_index().melt(
            id_vars='index', var_name='forecast_date', value_name='quarterly_forecast'
        )
        df_quarterly_melted.rename(columns={'index': 'forecasted_year'}, inplace=True)

        df_monthly_melted['forecast_date'] = pd.to_datetime(df_monthly_melted['forecast_date'], dayfirst=True)
        df_monthly_melted['forecasted_year'] = pd.to_datetime(df_monthly_melted['forecasted_year']).dt.year
        df_quarterly_melted['forecast_date'] = pd.to_datetime(df_quarterly_melted['forecast_date'], dayfirst=True)
        df_quarterly_melted['forecasted_year'] = pd.to_datetime(df_quarterly_melted['forecasted_year']).dt.year

        df_full_forecast = pd.merge_asof(
            df_monthly_melted.sort_values(['forecast_date', 'forecasted_year']),
            df_quarterly_melted.sort_values(['forecast_date', 'forecasted_year']),
            on='forecast_date',
            by='forecasted_year',
            suffixes=('', '_filled'),
            direction='backward'
        )

        df_full_forecast['all_forecasts'] = df_full_forecast['monthly_forecast'].fillna(
            df_full_forecast['quarterly_forecast']
        )

        df_full_forecast_monthly_ann = df_full_forecast.copy()
        df_full_forecast_monthly_ann['forecast_date'] = pd.to_datetime(df_full_forecast_monthly_ann['forecast_date'])
        df_full_forecast_monthly_ann['forecasted_year_start'] = pd.to_datetime(
            df_full_forecast_monthly_ann['forecasted_year'].astype(str) + '-01-01'
        )
        df_full_forecast_monthly_ann['months_left'] = 12 - df_full_forecast_monthly_ann['forecast_date'].dt.month + 1

        df_full_forecast_monthly_ann['annualized_forecast'] = np.where(
            df_full_forecast_monthly_ann['forecast_date'].dt.year == df_full_forecast_monthly_ann['forecasted_year'],
            (df_full_forecast_monthly_ann['all_forecasts'] / 12) * df_full_forecast_monthly_ann['months_left'],
            df_full_forecast_monthly_ann['all_forecasts']
        )

        monthly_forecasts = []
        for _, row in df_full_forecast_monthly_ann.iterrows():
            start_date = max(row['forecast_date'], row['forecasted_year_start'])
            end_date = f"{row['forecasted_year']}-12-01"
            monthly_dates = pd.date_range(start=start_date, end=end_date, freq='MS')

            if row['forecast_date'].year == row['forecasted_year']:
                monthly_forecast = row['annualized_forecast'] / row['months_left']
            else:
                monthly_forecast = row['all_forecasts'] / 12

            for month in monthly_dates:
                monthly_forecasts.append({
                    'forecast_date': row['forecast_date'],
                    'forecasted_year': row['forecasted_year'],
                    'forecasted_month': month,
                    'monthly_forecast': monthly_forecast
                })

        df_monthly_series = pd.DataFrame(monthly_forecasts)

        return df_monthly_series, df_full_forecast