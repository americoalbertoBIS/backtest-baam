import os
os.chdir(r'C:\git\backtest-baam\code')

from modeling.time_series_modeling import AR1Model, ARXModel

# Model configurations for beta forecasting
models_configurations = {
    "AR_1": {
        "beta1": "AR_1",
        "beta2": "AR_1",
        "beta3": "AR_1"
    },
    "AR_1_Output_Gap_Direct_Inflation_UCSV": {
        "beta1": "AR_1_Output_Gap_Direct_Inflation_UCSV",
        "beta2": "AR_1_Output_Gap_Direct_Inflation_UCSV",
        "beta3": "AR_1_Output_Gap_Direct_Inflation_UCSV"
    },
    "Mixed_Model": {
        "beta1": "AR_1_Output_Gap_Direct_Inflation_UCSV",
        "beta2": "AR_1_Output_Gap_Direct",
        "beta3": "AR_1"
    }
}

models = [
    {
        "name": "AR(1)",
        "handler": AR1Model(),
        "params": {}
    },
    {
        "name": "AR(1) + GDP",
        "handler": ARXModel(),
        "params": {}
    },
    {
        "name": "AR(1) + Output Gap (Direct)",
        "handler": ARXModel(),
        "params": {
            "output_gap_method": "direct"
        }
    },
    {
        "name": "AR(1) + Output Gap (HP Filter)",
        "handler": ARXModel(),
        "params": {
            "output_gap_method": "hp_filter",
            "lamb": 1600000,
            "one_sided": "kalman"
        }
    },
    {
        "name": "AR(1) + Inflation",
        "handler": ARXModel(),
        "params": {
            "inflation_method": "default"
        }
    },
    {
        "name": "AR(1) + Inflation (UCSV)",
        "handler": ARXModel(),
        "params": {
            "inflation_method": "ucsv"
        }
    },
    {
        "name": "AR(1) + GDP + Inflation",
        "handler": ARXModel(),
        "params": {
            "inflation_method": "default"
        }
    },
    {
        "name": "AR(1) + GDP + Inflation (UCSV)",
        "handler": ARXModel(),
        "params": {
            "output_gap_method": "gdp",
            "inflation_method": "ucsv"
        }
    },
    {
        "name": "AR(1) + Output Gap (Direct) + Inflation",
        "handler": ARXModel(),
        "params": {
            "output_gap_method": "direct",
            "inflation_method": "default"
        }
    },
    {
        "name": "AR(1) + Output Gap (Direct) + Inflation (UCSV)",
        "handler": ARXModel(),
        "params": {
            "output_gap_method": "direct",
            "inflation_method": "ucsv"
        }
    },
    {
        "name": "AR(1) + Output Gap (HP Filter) + Inflation",
        "handler": ARXModel(),
        "params": {
            "output_gap_method": "hp_filter",
            "lamb": 1600000,
            "one_sided": "kalman",
            "inflation_method": "default"
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