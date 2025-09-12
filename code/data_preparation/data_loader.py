import pandas as pd
import scipy.io
from datetime import datetime, timedelta
import numpy as np
import os
os.chdir(r'C:\git\backtest-baam\code')
# Define the global data path
DATA_PATH = r"\\msfsshared\bnkg\RMAS\Resources\BAAM\OpenBAAM\Private\Data"

class DataLoader:
    def __init__(self, country, variable_list, baam_path=None, macro_path=None, shadow=True):
        """
        Initializes the DataLoader class.

        Args:
            country (str): Country name (e.g., 'US').
            variable_list (list): List of macroeconomic variables to load (e.g., ['GDP', 'IP']).
            baam_path (str): Path to the BAAM betas `.mat` file (default=None, uses default path).
            macro_path (str): Path to the macroeconomic `.mat` file (default=None, uses default path).
            shadow (bool): Whether to use shadow rotation for betas (default=True).
        """
        # Set default paths using the global DATA_PATH
        self.default_baam_path = DATA_PATH + "\\BaseDB.mat"
        self.default_macro_path = DATA_PATH + "\\BaseDBmacro.mat"

        self.country = country
        self.variable_list = variable_list
        self.baam_path = baam_path if baam_path else self.default_baam_path
        self.macro_path = macro_path if macro_path else self.default_macro_path
        self.shadow = shadow
        self.data = None
        self.macro_data = None
        self.betas_data = None

    @staticmethod
    def read_BAAM_betas(country, baam_path, shadow=True):
        """
        Reads BAAM betas from the specified `.mat` file.

        Args:
            country (str): Country name (e.g., 'US').
            baam_path (str): Path to the BAAM betas `.mat` file.
            shadow (bool): Whether to use shadow rotation for betas (default=True).

        Returns:
            pd.DataFrame: DataFrame containing the betas.
        """
        try:
            if country == 'EA':
                country = 'DE'
                
            mat = scipy.io.loadmat(baam_path)
            AllCalcData = mat['AllCalcData']
            selectedCurveName = f"{country}GovernmentNominal"
            selected_curve = AllCalcData[selectedCurveName]
            selected_curve_data = selected_curve[0][0][0, 0][0]
            dates_num = selected_curve_data['Dates'][0][0]

            # Convert MATLAB datenum to Python datetime
            dates_str = [
                datetime.strftime(datetime.fromordinal(int(d)) - timedelta(days=366), '%Y-%m-%d')
                for d in dates_num
            ]

            model = 'NSFixedLZC10IC' if shadow else 'NSFixedLZC10'
            non_rotated_betas = AllCalcData[selectedCurveName][0][0][0, 0][0]['NSSEstim'][0][0][model][0][0]

            # Rotation logic
            lambda_ = 0.7173
            Short_m1 = 0.25
            auxEta = (1 - np.exp(-lambda_ * Short_m1)) / (Short_m1 * lambda_)

            rotation_matrix = np.array([
                [1, auxEta, auxEta - np.exp(-lambda_ * Short_m1)],
                [0, -auxEta, np.exp(-lambda_ * Short_m1) - auxEta],
                [0, 1 - auxEta, 1 + np.exp(-lambda_ * Short_m1) - auxEta]
            ])

            # Rotate betas
            pre_calc_betas = non_rotated_betas['Para'][0][0][-len(dates_str):, 0:3] @ rotation_matrix.T

            cols = ['beta1', 'beta2', 'beta3']
            df_betas = pd.DataFrame(pre_calc_betas, index=pd.to_datetime(dates_str), columns=cols)
            df_betas = df_betas.resample('MS').last()

            return df_betas

        except Exception as e:
            raise RuntimeError(f"Error reading BAAM betas: {e}")

    @staticmethod
    def read_baseDBmacro(country, variable_list, macro_path):
        """
        Reads macroeconomic variables from the specified `.mat` file.

        Args:
            country (str): Country name (e.g., 'US').
            variable_list (list): List of macroeconomic variables to load (e.g., ['GDP', 'IP']).
            macro_path (str): Path to the macroeconomic `.mat` file.

        Returns:
            pd.DataFrame: DataFrame containing the macroeconomic variables.
        """
        try:
            mat = scipy.io.loadmat(macro_path)

            # Extract data
            df_base_db_macro = pd.DataFrame(mat['L_PriceIndexSA'])
            df_base_db_macro.columns = mat['Header1'][1:]

            # Convert MATLAB datenum to Python datetime
            dates = mat['Dates']
            dates_converted = [
                datetime.fromordinal(int(d[0])) + timedelta(days=int(d[0]) % 1) - timedelta(days=366)
                for d in dates
            ]
            df_base_db_macro.index = dates_converted

            # Filter data for the specified country and variables
            df_base_db_macro_cc = df_base_db_macro.filter(regex=f'{country}')
            df_base_db_macro_cc.columns = [
                col.replace(' ', '').replace(f'{country}', f'{country}_') for col in df_base_db_macro_cc.columns
            ]

            cols = variable_list
            df_base_db_macro_cc = df_base_db_macro_cc[[f"{country}_{col}" for col in cols]]
            df_base_db_macro_cc = df_base_db_macro_cc.resample('MS').last()

            return df_base_db_macro_cc

        except Exception as e:
            raise RuntimeError(f"Error reading macroeconomic data: {e}")

    def load_betas(self):
        """
        Loads only the BAAM betas data.
        """
        self.betas_data = self.read_BAAM_betas(self.country, self.baam_path, self.shadow)

    def load_macro_data(self):
        """
        Loads only the macroeconomic data.
        """
        self.macro_data = self.read_baseDBmacro(self.country, self.variable_list, self.macro_path)

    def load_data(self):
        """
        Loads and combines BAAM betas and macroeconomic variables.
        """
        if self.betas_data is None:
            self.load_betas()
        if self.macro_data is None:
            self.load_macro_data()
        self.data = self.betas_data.merge(self.macro_data, left_index=True, right_index=True)
        self.data.sort_index(inplace=True)

    def get_betas(self):
        """
        Returns only the BAAM betas data.

        Returns:
            pd.DataFrame: DataFrame containing the betas.
        """
        if self.betas_data is None:
            self.load_betas()
        return self.betas_data

    def get_macro_data(self):
        """
        Returns only the macroeconomic data.

        Returns:
            pd.DataFrame: DataFrame containing the macroeconomic variables.
        """
        if self.macro_data is None:
            self.load_macro_data()
        return self.macro_data

    def get_data(self):
        """
        Returns the combined data (betas and macroeconomic variables).

        Returns:
            pd.DataFrame: Combined DataFrame of betas and macroeconomic variables.
        """
        if self.data is None:
            self.load_data()
        return self.data
    
class DataLoaderYC:
    """
    A class to load and process yield curve data from .mat files.
    """

    def __init__(self, file_path):
        """
        Initializes the DataLoader with the file path to the .mat file.

        Args:
            file_path (str): Path to the .mat file containing yield curve data.
        """
        self.file_path = file_path
        self.AllCalcData = None
        self.countries = ['US', 'FR', 'DE', 'UK', 'BE', 'CA', 'ES', 'NO', 'NZ', 'CH',
                          'JP', 'DK', 'IT', 'NL', 'AT', 'GR', 'PT', 'CN', 'IE', 'FI',
                          'AU', 'SE', 'MY', 'KR']
        self.sdr_countries = ['US', 'JP', 'CN', 'DE', 'UK']

    def load_data(self):
        """
        Loads the .mat file and extracts the AllCalcData structure.

        Returns:
            tuple: A tuple containing:
                - AllCalcData (dict): The loaded data structure from the .mat file.
                - countries (list): List of all available countries.
                - sdr_countries (list): List of SDR countries.
        """
        try:
            mat = scipy.io.loadmat(self.file_path)
            self.AllCalcData = mat['AllCalcData']
            return self.AllCalcData, self.countries, self.sdr_countries
        except FileNotFoundError:
            raise FileNotFoundError(f"The file {self.file_path} was not found.")
        except KeyError:
            raise KeyError("The .mat file does not contain the expected 'AllCalcData' structure.")
        except Exception as e:
            raise RuntimeError(f"An error occurred while loading the .mat file: {e}")

    def process_data(self, country):
        """
        Processes the yield curve data for a specific country.

        Args:
            country (str): The country to extract data for.

        Returns:
            tuple: A tuple containing:
                - selectedCurveName (str): The name of the selected yield curve.
                - selected_curve_data (dict): The data for the selected yield curve.
                - modelParams (dict): The model parameters for the yield curve.

        Raises:
            ValueError: If the specified country is not in the list of available countries.
            KeyError: If the yield curve for the country is not found in AllCalcData.
        """
        if country not in self.countries:
            raise ValueError(f"Country '{country}' is not available. Please select from: {self.countries}")

        try:
            selectedCurveName = f"{country}GovernmentNominal"
            selected_curve = self.AllCalcData[selectedCurveName]
            selected_curve_data = selected_curve[0][0][0, 0][0]
            curve_parameters = selected_curve_data['CurMethodParameters'][0][0]

            modelParams = self._extract_model_params(curve_parameters)
            return selectedCurveName, selected_curve_data, modelParams
        except KeyError:
            raise KeyError(f"The yield curve data for country '{country}' could not be found in AllCalcData.")
        except Exception as e:
            raise RuntimeError(f"An error occurred while processing data for country '{country}': {e}")

    def _extract_model_params(self, curve_parameters):
        """
        Extracts model parameters from the curve parameters structure.

        Args:
            curve_parameters (dict): The curve parameters structure from the .mat file.

        Returns:
            dict: A dictionary containing the extracted model parameters.
        """
        try:
            modelParams = {
                'deltaC': float(curve_parameters['deltaC'][0][0][0]),
                'gammaC': float(curve_parameters['gammaC'][0][0][0]),
                'deltaS': float(curve_parameters['deltaS'][0][0][0]),
                'gammaS': float(curve_parameters['gammaS'][0][0][0]),
                'fCurvatureConstr': bool(int(curve_parameters['fCurvatureConstr'][0][0][0])),
                'alphabar': float(curve_parameters['alphabar'][0][0][0]),
                'y_minOffset': float(curve_parameters['y_minOffset'][0][0][0])
            }
            return modelParams
        except KeyError as e:
            raise KeyError(f"Missing parameter in curve_parameters: {e}")
        except Exception as e:
            raise RuntimeError(f"An error occurred while extracting model parameters: {e}")

    def get_available_countries(self):
        """
        Returns the list of available countries.

        Returns:
            list: List of available countries.
        """
        return self.countries

    def get_sdr_countries(self):
        """
        Returns the list of SDR countries.

        Returns:
            list: List of SDR countries.
        """
        return self.sdr_countries