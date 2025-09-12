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
    "AR_1_Output_Gap_Direct_Inflation_UCSV_MRM": {
        "beta1": "AR_1_Output_Gap_Direct_Inflation_UCSV_MRM",
        "beta2": "AR_1_Output_Gap_Direct_Inflation_UCSV_MRM",
        "beta3": "AR_1_Output_Gap_Direct_Inflation_UCSV_MRM"
    },
    "Mixed_Model": {
        "beta1": "AR_1_Output_Gap_Direct_Inflation_UCSV",
        "beta2": "AR_1_Output_Gap_Direct",
        "beta3": "AR_1"
    },
    "Mixed_Model_curvMacro": {
        "beta1": "AR_1_Output_Gap_Direct_Inflation_UCSV",
        "beta2": "AR_1_Output_Gap_Direct",
        "beta3": "AR_1_Output_Gap_Direct_Inflation_UCSV"
    },
    "Mixed_Model_MRM": {
        "beta1": "AR_1_Output_Gap_Direct_Inflation_UCSV_MRM",
        "beta2": "AR_1_Output_Gap_Direct_MRM",
        "beta3": "AR_1"
    }
}

selected_models = [
    {
        "name": "AR(1) + Inflation (UCSV)",
        "handler": ARXModel(),
        "params": {
            "inflation_method": "ucsv",
            "macro_forecast": "consensus",
            "exogenous_variables": ["inflation"]
        }
    },
    {
        "name": "AR(1) + Inflation (UCSV) - MRM",
        "handler": ARXModel(),
        "params": {
            "inflation_method": "ucsv",
            "macro_forecast": "ar_1",
            "exogenous_variables": ["inflation"]
        }
    },
]

models = [
    {
        "name": "AR(1)",
        "handler": AR1Model(),
        "params": {}
    },
    {
        "name": "AR(1) + Output Gap (Direct)",
        "handler": ARXModel(),
        "params": {
            "output_gap_method": "direct",
            "macro_forecast": "consensus",
            "exogenous_variables": ["output_gap"]
        }
    },
    {
        "name": "AR(1) + Output Gap (Direct) - MRM",
        "handler": ARXModel(),
        "params": {
            "output_gap_method": "direct",
            "macro_forecast": "ar_1",
            "exogenous_variables": ["output_gap"]
        }
    },
    {
        "name": "AR(1) + Inflation (UCSV)",
        "handler": ARXModel(),
        "params": {
            "inflation_method": "ucsv",
            "macro_forecast": "consensus",
            "exogenous_variables": ["inflation"]
        }
    },
    {
        "name": "AR(1) + Inflation (UCSV) - MRM",
        "handler": ARXModel(),
        "params": {
            "inflation_method": "ucsv",
            "macro_forecast": "ar_1",
            "exogenous_variables": ["inflation"]
        }
    },
    {
        "name": "AR(1) + Output Gap (Direct) + Inflation (UCSV)",
        "handler": ARXModel(),
        "params": {
            "output_gap_method": "direct",
            "inflation_method": "ucsv",
            "macro_forecast": "consensus",
            "exogenous_variables": ["output_gap", "inflation"]
        }
    },
    {
        "name": "AR(1) + Output Gap (Direct) + Inflation (UCSV) - MRM",
        "handler": ARXModel(),
        "params": {
            "output_gap_method": "direct",
            "inflation_method": "ucsv",
            "macro_forecast": "ar_1",
            "exogenous_variables": ["output_gap", "inflation"]
        }
    },
]

all_models = [
    {
        "name": "AR(1)",
        "handler": AR1Model(),
        "params": {}
    },
    {
        "name": "AR(1) + GDP",
        "handler": ARXModel(),
        "params": {
            "macro_forecast": "consensus",
            "exogenous_variables": ["gdp_yoy"]
        }
    },
    {
        "name": "AR(1) + Output Gap (Direct)",
        "handler": ARXModel(),
        "params": {
            "output_gap_method": "direct",
            "macro_forecast": "consensus",
            "exogenous_variables": ["output_gap"]
        }
    },
    {
        "name": "AR(1) + Output Gap (HP Filter)",
        "handler": ARXModel(),
        "params": {
            "output_gap_method": "hp_filter",
            "lamb": 1600000,
            "one_sided": "kalman",
            "macro_forecast": "consensus",
            "exogenous_variables": ["output_gap"]
        }
    },
    {
        "name": "AR(1) + Inflation",
        "handler": ARXModel(),
        "params": {
            "inflation_method": "default",
            "macro_forecast": "consensus",
            "exogenous_variables": ["inflation"]
        }
    },
    {
        "name": "AR(1) + Inflation (UCSV)",
        "handler": ARXModel(),
        "params": {
            "inflation_method": "ucsv",
            "macro_forecast": "consensus",
            "exogenous_variables": ["inflation"]
        }
    },
    {
        "name": "AR(1) + GDP + Inflation",
        "handler": ARXModel(),
        "params": {
            "inflation_method": "default",
            "macro_forecast": "consensus",
            "exogenous_variables": ["gdp_yoy", "inflation"]
        }
    },
    {
        "name": "AR(1) + GDP + Inflation (UCSV)",
        "handler": ARXModel(),
        "params": {
            "output_gap_method": "gdp",
            "inflation_method": "ucsv",
            "macro_forecast": "consensus",
            "exogenous_variables": ["gdp_yoy", "inflation"]
        }
    },
    {
        "name": "AR(1) + Output Gap (Direct) + Inflation",
        "handler": ARXModel(),
        "params": {
            "output_gap_method": "direct",
            "inflation_method": "default",
            "macro_forecast": "consensus",
            "exogenous_variables": ["output_gap", "inflation"]
        }
    },
    {
        "name": "AR(1) + Output Gap (Direct) + Inflation (UCSV)",
        "handler": ARXModel(),
        "params": {
            "output_gap_method": "direct",
            "inflation_method": "ucsv",
            "macro_forecast": "consensus",
            "exogenous_variables": ["output_gap", "inflation"]
        }
    },
    {
        "name": "AR(1) + Output Gap (HP Filter) + Inflation",
        "handler": ARXModel(),
        "params": {
            "output_gap_method": "hp_filter",
            "lamb": 1600000,
            "one_sided": "kalman",
            "inflation_method": "default",
            "macro_forecast": "consensus",
            "exogenous_variables": ["output_gap", "inflation"]
        }
    },
    {
        "name": "AR(1) + Output Gap (HP Filter) + Inflation (UCSV)",
        "handler": ARXModel(),
        "params": {
            "output_gap_method": "hp_filter",
            "lamb": 1600000,
            "one_sided": "kalman",
            "inflation_method": "ucsv",
            "macro_forecast": "consensus",
            "exogenous_variables": ["output_gap", "inflation"]
        }
    },
]