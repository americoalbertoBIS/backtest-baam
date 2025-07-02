import os
import sys
import time
import streamlit as st
from tqdm import tqdm
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.optimize import minimize, fminbound, Bounds
from concurrent.futures import ProcessPoolExecutor, as_completed

import os
os.chdir(r'C:\git\backtest-baam\code')
from modeling.nelson_siegel import compute_nsr, compute_ns_rotated_shadow

class YieldCurveModel:
    def __init__(self, selected_curve_data, modelParams):
        self.selected_curve_data = selected_curve_data
        self.modelParams = modelParams
        self._initialize_data()

    def _initialize_data(self):
        self.source_data = self.selected_curve_data['SourceData']
        self.yields = self.source_data[0][0][:, :, 0]
        self.maturities = self.source_data[0][0][:, :, 1]
        self.coupons = self.source_data[0][0][:, :, 2]
        self.dates_num = self.selected_curve_data['Dates'][0][0]
        self.dates_str = [datetime.strftime(datetime.fromordinal(int(d)) - timedelta(days=366), '%Y-%m-%d') for d in self.dates_num]
        self.nObs = None
        self.varNames = None
        self.uniqueTaus = None

        self._prepare_data()

    def _prepare_data(self):
        self._set_min_yields_hist()
        self._filter_first_observations()
        self._process_maturities()

        if np.any(self.couponsObservedAgg == -1):
            self._convert_par_to_zero_yields()

        self.varNames = [f'{tau} years' for tau in self.uniqueTaus]
        self.yieldsObservedTT = pd.DataFrame(self.yieldsObservedAgg, columns=self.varNames, index=pd.to_datetime(self.dates_str))

        self._set_fixed_lambdas()
        self._define_rotation_matrix()

    def _set_min_yields_hist(self):
        min_yields_hist = min(np.nanmin(self.yields), 0)
        self.modelParams['y_min'] = min_yields_hist + self.modelParams['y_minOffset']

    def _filter_first_observations(self):
        firstObservations = np.where(~np.all(np.isnan(self.yields), axis=1))[0][0]
        self.yields = self.yields[firstObservations:, :]
        self.maturities = self.maturities[firstObservations:, :]
        self.coupons = self.coupons[firstObservations:, :]
        self.dates_str = self.dates_str[firstObservations:]
        self.nObs = self.yields.shape[0]

    def _process_maturities(self):
        auxMaturities = np.nanmax(self.maturities, axis=0)
        minMaturity = self.modelParams['minMaturity']
        maxMaturity = self.modelParams['maxMaturity']
        tinyDiff = 1e-5
        selectedColumns = (minMaturity - tinyDiff <= auxMaturities) & (auxMaturities <= maxMaturity + tinyDiff)

        self.taus = auxMaturities[selectedColumns]
        self.ys = self.yields[:, selectedColumns]
        self.cns = self.coupons[:, selectedColumns]

        self.uniqueTaus = np.sort(np.unique(self.taus))
        auxDiff = (self.uniqueTaus[1:] - self.uniqueTaus[:-1]) < tinyDiff
        if np.any(auxDiff):
            self.uniqueTaus = self.uniqueTaus[~np.concatenate(([False], auxDiff))]

        self.numMaturities = len(self.uniqueTaus)
        self.yieldsObservedAgg = np.full((self.nObs, self.numMaturities), np.nan)
        self.couponsObservedAgg = np.full((self.nObs, self.numMaturities), np.nan)

        self._aggregate_yields_and_coupons(tinyDiff)

    def _aggregate_yields_and_coupons(self, tinyDiff):
        for i in range(self.numMaturities):
            tau = self.uniqueTaus[i]
            selectCol = np.abs(self.taus - tau) <= tinyDiff
            self._aggregate_yields(i, selectCol)
            self._aggregate_coupons(i, selectCol)

    def _aggregate_yields(self, i, selectCol):
        selectYields = self.ys[:, selectCol]
        if np.all(np.sum(~np.isnan(selectYields), axis=1) <= 1):
            self.yieldsObservedAgg[:, i] = np.nansum(selectYields, axis=1)
            self.yieldsObservedAgg[np.sum(~np.isnan(selectYields), axis=1) == 0, i] = np.nan
        else:
            comPart = np.sum(~np.isnan(selectYields), axis=1) > 1
            self.yieldsObservedAgg[~comPart, i] = np.nansum(selectYields[~comPart, :], axis=1)
            self.yieldsObservedAgg[comPart, i] = np.nanmean(selectYields[comPart, :], axis=1)
            self.yieldsObservedAgg[np.sum(~np.isnan(selectYields), axis=1) == 0, i] = np.nan

    def _aggregate_coupons(self, i, selectCol):
        selectCoupons = self.cns[:, selectCol]
        if np.all(np.sum(~np.isnan(selectCoupons), axis=1) <= 1):
            self.couponsObservedAgg[:, i] = np.nansum(selectCoupons, axis=1)
            self.couponsObservedAgg[np.sum(~np.isnan(selectCoupons), axis=1) == 0, i] = np.nan
        else:
            comPart = np.sum(~np.isnan(selectCoupons), axis=1) > 1
            self.couponsObservedAgg[~comPart, i] = np.nansum(selectCoupons[~comPart, :], axis=1)
            self.couponsObservedAgg[comPart, i] = np.nanmean(selectCoupons[comPart, :], axis=1)
            self.couponsObservedAgg[np.sum(~np.isnan(selectCoupons), axis=1) == 0, i] = np.nan

    def _convert_par_to_zero_yields(self):
        zero_coupon_yields = self.convert_par_to_zero_yields(self.yieldsObservedAgg, self.couponsObservedAgg, self.uniqueTaus)
        par_yields_mask = self.couponsObservedAgg == -1
        self.yieldsObservedAgg[par_yields_mask] = zero_coupon_yields[par_yields_mask]

    def _set_fixed_lambdas(self):
        tau1 = 2.5
        Lambda1Fixed = self.modelParams['lambda1fixed']

        tau2 = 0.5
        Lambda2Fixed = fminbound(lambda x: -((1 - np.exp(-x * tau2)) / (x * tau2) - np.exp(-x * tau2)), 0, 30)

        self.modelParams['lambda'] = Lambda1Fixed

    def _define_rotation_matrix(self):
        lambda_ = self.modelParams['lambda']
        Short_m1 = 0.25
        auxEta = (1 - np.exp(-lambda_ * Short_m1)) / (Short_m1 * lambda_)

        self.RotationMatrix = np.array([
            [1, auxEta, auxEta - np.exp(-lambda_ * Short_m1)],
            [0, -auxEta, np.exp(-lambda_ * Short_m1) - auxEta],
            [0, 1 - auxEta, 1 + np.exp(-lambda_ * Short_m1) - auxEta]
        ])

        self.invRotationMatrix = np.array([
            [1, 1, 0],
            [0, (auxEta - 1) * np.exp(lambda_ * Short_m1) - 1, 1 - auxEta * np.exp(lambda_ * Short_m1)],
            [0, -(auxEta - 1) * np.exp(lambda_ * Short_m1), auxEta * np.exp(lambda_ * Short_m1)]
        ])

        invRotationMatrixNum = np.linalg.inv(self.RotationMatrix)
        assert np.all(np.abs(self.invRotationMatrix - invRotationMatrixNum) < 1e-10)

    def bootstrap(self, par_yields, maturities):
        zero_coupon_rates = []
        for i, (maturity, y) in enumerate(zip(maturities, par_yields)):
            if maturity == 0.25 or maturity == 0.5:
                z = y * 0.5
            else:
                y = y * 0.5
                sum_coupons = sum([y / (1 + zero_coupon_rates[j - 1] / 100) ** j for j in range(1, i + 1)])
                z = ((((100 + y) / (100 - sum_coupons)) ** (1 / (i + 1))) - 1) * 100
            zero_coupon_rates.append(z)
        return [i / 0.5 for i in zero_coupon_rates]

    def convert_par_to_zero_yields(self, par_yields, coupons, maturities):
        """
        Convert par yields to zero-coupon yields using the bootstrap method.
        """
        zero_coupon_yields = np.full_like(par_yields, np.nan)
        
        for t in range(par_yields.shape[0]):
            if np.any(coupons[t] == -1):
                available_maturities = maturities[~np.isnan(par_yields[t])]
                par_yields_t = par_yields[t, ~np.isnan(par_yields[t])]
                zero_coupon_rates = self.bootstrap(par_yields_t, available_maturities)
                
                for j, maturity in enumerate(available_maturities):
                    zero_coupon_yields[t, maturities == maturity] = zero_coupon_rates[j]
        
        return zero_coupon_yields

    def estimate_for_time(self, t, tTaus=None, tYields=None):
        """
        Estimate factors for a single time step.

        Parameters:
        t (int): Time index for estimation.
        tTaus (np.ndarray, optional): Maturities for the current time step. If None, derives from `self.uniqueTaus`.
        tYields (np.ndarray, optional): Yields for the current time step. If None, derives from `self.yieldsObservedAgg`.

        Returns:
        tuple: Estimated factors and related metrics for the time step, including selectedInds (if applicable).
        """
        if tTaus is None or tYields is None:
            # Default behavior: Derive tTaus and tYields from class attributes
            yt = self.yieldsObservedAgg[t, :]
            selectedInds = ~np.isnan(yt)
            tYields = yt[selectedInds]
            tTaus = self.uniqueTaus[selectedInds]
            
        else:
            # Specialized behavior: Use provided tTaus and tYields, no selectedInds needed
            selectedInds = None

        # Ensure tYields and tTaus are valid
        if len(tYields) < 3:
            return None
        
        tYields = tYields.reshape(-1, 1)
        level0 = tYields[0, 0]
        slope0 = tYields[-1, 0] - tYields[0, 0]
        curve0 = 0.05
        x0 = np.array([level0, slope0, curve0])

        bounds = Bounds([self.modelParams['y_min'], -np.inf, -np.inf], [np.inf, np.inf, np.inf])

        def objective_func(x):
            return compute_nsr(x, tYields, tTaus, self.modelParams['lambda'], self.invRotationMatrix)[0]

        try:
            Beta_est = minimize(objective_func, x0, bounds=bounds, options={'maxiter': 1e8, 'disp': False}, method='trust-constr').x
        except Exception as e:
            print(f'Error in minimization: {e}')
            return None

        RSS, observedYields_est, tEstErrors = compute_nsr(Beta_est, tYields, tTaus, self.modelParams['lambda'], self.invRotationMatrix)
        MSE = RSS / len(tYields)  # Calculate MSE
        tCurv = Beta_est[2]
        Acurv = np.array([[0, 0, 1], [0, 0, -1]])
        smallWindow = 0.001
        bcurv = np.array([tCurv + smallWindow, -(tCurv - smallWindow)])
        shadowBetas0 = Beta_est

        def objective_func_shadow(x):
            return compute_ns_rotated_shadow(x, tYields, tTaus, self.invRotationMatrix, self.modelParams)[0]

        constraints_shadow = [{'type': 'ineq', 'fun': lambda x: Acurv @ x - bcurv}]

        try:
            shadowBeta_est = minimize(objective_func_shadow, shadowBetas0, constraints=constraints_shadow, options={'maxiter': 1e8, 'disp': False}, method='trust-constr').x
        except Exception as e:
            print(f'Error in minimization: {e}')
            return None

        RSS_shadow, observedYields_est_shadow, tEstErrors_shadow, shadowYields_est, tAlpha, tSlope, tCurv = compute_ns_rotated_shadow(shadowBeta_est, tYields, tTaus, self.invRotationMatrix, self.modelParams)
        MSE_shadow = RSS_shadow / len(tYields)  # Calculate MSE for shadow

        return (t, Beta_est, MSE, observedYields_est, tEstErrors, shadowBeta_est, MSE_shadow, observedYields_est_shadow, tEstErrors_shadow, shadowYields_est, tAlpha, tSlope, tCurv, selectedInds)

    def estimate_yield_curve(self):
        estYields = np.full_like(self.yieldsObservedAgg, np.nan)
        estErrors = np.full_like(self.yieldsObservedAgg, np.nan)
        estFactors = np.full((self.nObs, 3), np.nan)
        estMSE = np.full(self.nObs, np.nan)
        estYields_shadow = np.full_like(self.yieldsObservedAgg, np.nan)
        estShadowYields_shadow = np.full_like(self.yieldsObservedAgg, np.nan)
        estErrors_shadow = np.full_like(self.yieldsObservedAgg, np.nan)
        estFactors_shadow = np.full((self.nObs, 3), np.nan)
        estMSE_shadow = np.full(self.nObs, np.nan)
        alpha_shadow = np.full(self.nObs, np.nan)
        slope_shadow = np.full(self.nObs, np.nan)
        curv_shadow = np.full(self.nObs, np.nan)

        start_time = time.time()

        progress_bar = st.progress(0)
        progress_text = st.empty()

        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(self.estimate_for_time, t): t for t in range(self.nObs)}
            for i, future in enumerate(as_completed(futures)):
                result = future.result()
                if result is not None:
                    t, Beta_est, MSE, observedYields_est, tEstErrors, shadowBeta_est, MSE_shadow, observedYields_est_shadow, tEstErrors_shadow, shadowYields_est, tAlpha, tSlope, tCurv, selectedInds = result
                    estMSE[t] = MSE
                    estFactors[t, :] = Beta_est
                    estYields[t, selectedInds] = observedYields_est.flatten()
                    estErrors[t, selectedInds] = tEstErrors.flatten()
                    estMSE_shadow[t] = MSE_shadow
                    estFactors_shadow[t, :] = shadowBeta_est
                    estYields_shadow[t, selectedInds] = observedYields_est_shadow.flatten()
                    estShadowYields_shadow[t, selectedInds] = shadowYields_est.flatten()
                    alpha_shadow[t] = tAlpha
                    slope_shadow[t] = tSlope
                    curv_shadow[t] = tCurv
                    estErrors_shadow[t, selectedInds] = tEstErrors_shadow.flatten()
                progress_bar.progress((i + 1) / self.nObs)
                progress_text.text(f"Progress: {((i + 1) / self.nObs) * 100:.2f}%")

        end_time = time.time()
        st.success(f"Yield curve estimation completed for {self.modelParams['country']} in {end_time - start_time:.2f} seconds")

        estYieldsTT = pd.DataFrame(estYields, index=pd.to_datetime(self.dates_str), columns=self.varNames)
        estYieldsShadowTT = pd.DataFrame(estYields_shadow, index=pd.to_datetime(self.dates_str), columns=self.varNames)
        estShadowYieldsShadowTT = pd.DataFrame(estShadowYields_shadow, index=pd.to_datetime(self.dates_str), columns=self.varNames)

        estFactorsTT = pd.DataFrame(estFactors, index=pd.to_datetime(self.dates_str), columns=['beta1', 'beta2', 'beta3'])
        estFactorsShadowTT = pd.DataFrame(estFactors_shadow, index=pd.to_datetime(self.dates_str), columns=['beta1', 'beta2', 'beta3'])

        estMSETT = pd.DataFrame(estMSE, index=pd.to_datetime(self.dates_str), columns=['MSE'])
        estMSEShadowTT = pd.DataFrame(estMSE_shadow, index=pd.to_datetime(self.dates_str), columns=['MSE_shadow'])

        return self.yieldsObservedTT, estYieldsTT, estYieldsShadowTT, estShadowYieldsShadowTT, estFactorsTT, estFactorsShadowTT, alpha_shadow, slope_shadow, curv_shadow, estMSETT, estMSEShadowTT