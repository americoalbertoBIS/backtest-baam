import numpy as np


def manual_column_stack(numYields, taus, lambda_):
    """
    Manually construct the matrix H.
    """
    H = np.zeros((numYields, 3))
    H[:, 0] = 1
    H[:, 1] = (1 - np.exp(-lambda_ * taus)) / (lambda_ * taus)
    H[:, 2] = (1 - np.exp(-lambda_ * taus)) / (lambda_ * taus) - np.exp(-lambda_ * taus)
    return H


def compute_nsr_ts_noErr(rotatedbetas, taus, lambda_, invRotMat):
    H = manual_column_stack(len(taus), taus, lambda_)
    G = H @ invRotMat
    yields_est = np.dot(G, rotatedbetas.T).T

    return yields_est


def compute_nsr_shadow_ts_noErr(shadowBetas, taus, invRotMat, modelParams):
    lambda_ = modelParams['lambda']

    shadowYields_est = compute_nsr_ts_noErr(shadowBetas, taus, lambda_, invRotMat)

    y_min = modelParams['y_min']
    shSlope_ = shadowBetas[:,1] + np.minimum(0, shadowBetas[:,0] - y_min)
    shCurv_ = shadowBetas[:,2]

    deltaS = modelParams['deltaS']
    gammaS = modelParams['gammaS']
    deltaC = modelParams['deltaC']
    gammaC = modelParams['gammaC']
    alphabar = modelParams['alphabar']

    alpha_ = ((np.tanh(deltaS * shSlope_ + gammaS) + 3) / 2) * ((np.tanh(deltaC * shCurv_ + gammaC) + 3) / 2) + alphabar
    alpha_ = alpha_.reshape(-1,1)
    smallEps = 1e-7

    if np.any(np.abs(-alpha_ * (shadowYields_est - y_min)) < smallEps):
        shadowYields_est[np.abs(-alpha_ * (shadowYields_est - y_min)) < smallEps] = y_min + smallEps
    observedYields_est = y_min + (shadowYields_est - y_min) / (1 - np.exp(-alpha_ * (shadowYields_est - y_min)))

    return observedYields_est


def compute_nsr(rotatedbetas, observedYields, taus, lambda_, invRotMat):
    numYields = observedYields.shape[0]
    H = manual_column_stack(numYields, taus, lambda_)
    G = H @ invRotMat
    yields_est = G @ rotatedbetas[:3]
    estErrors = observedYields.flatten() - yields_est.flatten()
    RSS = np.sum(estErrors**2)
    return RSS, yields_est, estErrors


def compute_ns_rotated_shadow(shadowBetas, observedYields, taus, invRotMat, modelParams):
    lambda_ = modelParams['lambda']
    RSS, shadowYields_est, _ = compute_nsr(shadowBetas, observedYields, taus, lambda_, invRotMat)
    y_min = modelParams['y_min']
    shSlope_ = shadowBetas[1] + min(0, shadowBetas[0] - y_min)
    shCurv_ = shadowBetas[2]
    deltaS = modelParams['deltaS']
    gammaS = modelParams['gammaS']
    deltaC = modelParams['deltaC']
    gammaC = modelParams['gammaC']
    alphabar = modelParams['alphabar']
    alpha_ = ((np.tanh(deltaS * shSlope_ + gammaS) + 3) / 2) * ((np.tanh(deltaC * shCurv_ + gammaC) + 3) / 2) + alphabar
    smallEps = 1e-7
    if np.any(np.abs(-alpha_ * (shadowYields_est - y_min)) < smallEps):
        shadowYields_est[np.abs(-alpha_ * (shadowYields_est - y_min)) < smallEps] = y_min + smallEps
    observedYields_est = y_min + (shadowYields_est - y_min) / (1 - np.exp(-alpha_ * (shadowYields_est - y_min)))
    estErrors = observedYields.flatten() - observedYields_est.flatten()
    RSS = np.sum(estErrors**2)
    return RSS, observedYields_est, estErrors, shadowYields_est, alpha_, shSlope_, shCurv_