import os
os.chdir(r'C:\git\backtest-baam\code')

from modeling.time_series_modeling import AR1Model, ARXModel

models = [
    {
        "name": "AR(1)",
        "handler": AR1Model(),
        "params": {
            "output_gap_method": None,
            "inflation_method": None
        }
    },
    {
        "name": "AR(1) + Output Gap (Direct)",
        "handler": ARXModel(),
        "params": {
            "output_gap_method": "direct",
            "inflation_method": None
        }
    },
    {
        "name": "AR(1) + Output Gap (HP Filter)",
        "handler": ARXModel(),
        "params": {
            "output_gap_method": "hp_filter",
            "lamb": 1600000,
            "one_sided": "kalman",
            "inflation_method": None
        }
    },
    {
        "name": "AR(1) + Inflation (UCSV)",
        "handler": ARXModel(),
        "params": {
            "output_gap_method": None,
            "inflation_method": "ucsv"
        }
    },
    {
        "name": "AR(1) + Output Gap (HP Filter) + Inflation (UCSV)",
        "handler": ARXModel(),
        "params": {
            "output_gap_method": "hp_filter",
            "lamb": 1600000,
            "one_sided": "kalman",
            "inflation_method": "ucsv"
        }
    }
]