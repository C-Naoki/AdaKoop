import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.models.adakoop import AdaKoop
from utils.metrics import mae, mse


def run(data: pd.DataFrame, model: AdaKoop, cfg: DictConfig) -> None:
    X = data.to_numpy()
    n, d = X.shape

    train_rate = cfg.model.train_rate
    test_rate = cfg.model.test_rate
    train_size = int(n * train_rate)
    test_size = int(n * test_rate)
    valid_size = n - train_size - test_size

    lcurr = cfg.model.lcurr
    lstep = cfg.model.lstep
    X_train = X[:train_size]
    X_valid = X[train_size - lcurr : train_size + valid_size]

    grid_res = grid_search(X_train, X_valid, cfg)

    model.init_params(
        d=d,
        lcurr=lcurr,
        nu=cfg.model.nu,
        gamma=grid_res['best_gamma'],
        kernel_type=cfg.model.kernel_type,
        lambda_A=grid_res['best_lambda_A'],
        em_iters=cfg.model.em_iters,
        em_tol=cfg.model.em_tol,
        r_init=cfg.model.r_init,
        burnin=cfg.model.burnin,
        chi2_p=cfg.model.chi2_p,
        cusum_h=cfg.model.cusum_h,
        exceed_rate_th=cfg.model.exceed_rate_th,
        state_reset_P_scale=cfg.model.state_reset_P_scale,
        jitter=cfg.model.jitter,
        compress=cfg.model.compress,
        add_dict=cfg.model.add_dict,
        online_update=cfg.model.online_update,
    )

    model.initialize(X_train)

    X_test = X[train_size + valid_size - lcurr :]
    pred = np.zeros_like(X_test)
    pred[:] = np.nan
    model.set_initial_model(X_test[:lcurr])
    for i in range(test_size - lcurr - lstep):
        tc = i + lcurr
        x_new = X_test[tc]
        # estimate best model
        model.model_selection(x_new)
        # parameter update
        _ = model.update(x_new)
        # forecast future value
        Vf, _, _ = model.forecast(lstep=lstep)
        pred[tc + lstep] = Vf[-1]

    mask = ~np.isnan(pred).any(axis=1)
    print(f'MSE: {mse(X_test[mask], pred[mask]):.4f}, MAE: {mae(X_test[mask], pred[mask]):.4f}')

    return {'pred': pred}


def grid_search(X_train: np.ndarray, X_valid: np.ndarray, cfg: DictConfig) -> None:
    gamma_cand = [1e-2, 3e-3, 1e-3]
    lambda_A_cand = [1e-8, 1e-7, 1e-6, 1e-5]
    err_ls, gamma_ls, lambda_A_ls = [], [], []
    for gamma in gamma_cand:
        for lambda_A in lambda_A_cand:
            model_cand = AdaKoop(verbose=False)
            model_cand.init_params(
                d=X_train.shape[1],
                lcurr=cfg.model.lcurr,
                nu=cfg.model.nu,
                gamma=gamma,
                kernel_type=cfg.model.kernel_type,
                lambda_A=lambda_A,
                em_iters=cfg.model.em_iters,
                em_tol=cfg.model.em_tol,
                r_init=cfg.model.r_init,
                burnin=cfg.model.burnin,
                chi2_p=cfg.model.chi2_p,
                cusum_h=cfg.model.cusum_h,
                exceed_rate_th=cfg.model.exceed_rate_th,
                state_reset_P_scale=cfg.model.state_reset_P_scale,
                jitter=cfg.model.jitter,
                compress=cfg.model.compress,
                add_dict=cfg.model.add_dict,
                online_update=cfg.model.online_update,
            )
            model_cand.initialize(X_train)
            model_cand.set_initial_model(X_valid[: cfg.model.lcurr])

            preds, trues = [], []
            for i in range(len(X_valid) - cfg.model.lcurr - cfg.model.lstep):
                tc = i + cfg.model.lcurr
                x_new = X_valid[tc]
                model_cand.model_selection(x_new)
                model_cand.update(x_new)
                Vf, _, _ = model_cand.forecast(lstep=cfg.model.lstep)
                true = X_valid[tc + 1 : tc + 1 + cfg.model.lstep]
                preds.append(Vf)
                trues.append(true)
            preds = np.array(preds)
            trues = np.array(trues)
            mse = np.mean((preds - trues) ** 2)
            err_ls.append(mse)
            gamma_ls.append(gamma)
            lambda_A_ls.append(lambda_A)
            print(f'gamma: {gamma}, lambda_A: {lambda_A}, MSE: {mse:.4f}')
    best_idx = np.argmin(err_ls)
    return {
        'best_gamma': gamma_ls[best_idx],
        'best_lambda_A': lambda_A_ls[best_idx],
        'best_mse': err_ls[best_idx],
    }
