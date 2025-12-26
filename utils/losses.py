import torch
import torch.nn.functional as F

def gaussian_nll_loss_with_alpha(target, predicted_mean, predicted_var_diag, alpha=1.0):
    """ Vypočítá Gaussian Negative Log Likelihood. """
    predicted_var_diag = predicted_var_diag + 1e-12
    
    log_det_cov = torch.sum(torch.log(predicted_var_diag), dim=-1)
    
    diff = target - predicted_mean
    mahalanobis_sq = torch.sum((diff**2) / predicted_var_diag, dim=-1)
    
    element_wise_nll = 0.5 * (alpha * log_det_cov + mahalanobis_sq)
    
    return torch.mean(element_wise_nll)


def gaussian_nll(target, predicted_mean, predicted_var, min_var=1e-3):
    """Gaussian Negative Log Likelihood"""
    predicted_var = torch.clamp(predicted_var, min=min_var)
    mahal = torch.square(target - predicted_mean) / torch.abs(predicted_var)
    log2pi = torch.log(torch.tensor(2 * torch.pi, device=target.device))
    element_wise_nll = 0.5 * (torch.log(torch.abs(predicted_var)) + log2pi + mahal)
    sample_wise_error = torch.sum(element_wise_nll, dim=-1)
    return torch.mean(sample_wise_error)

import torch
import math

def gaussian_nll_safe(target, preds, var, min_var=1e-6, max_error_cap=100.0):
    # 1. Pouze technické epsilon proti dělení nulou (necháme gradienty protékat)
    var = var + min_var 
    
    # 2. Výpočet kvadratické chyby
    error_sq = (preds - target) ** 2
    
    # 3. Mahalanobis distance (chyba vážená nejistotou)
    mahalanobis = error_sq / var
    
    # === ZDE JE TA MAGIE ===
    # Místo clampování variance clampujeme AŽ VÝSLEDEK dělení.
    # Tím říkáme: "Trest za chybu může být maximálně 100."
    # To zabrání explozi Loss (3000+), ale zachová směr gradientu "zvyš varianci".
    mahalanobis_clamped = torch.clamp(mahalanobis, max=max_error_cap)
    
    # 4. NLL = log(var) + mahalanobis
    nll = 0.5 * (torch.log(var) + mahalanobis_clamped)
    
    return nll.mean()

def hybrid_loss(target, predicted_mean, predicted_var, lambda_mse, regularization_loss=0):
    nll_term = gaussian_nll(target, predicted_mean, predicted_var)

    mse_term = F.mse_loss(predicted_mean, target)

    combined_loss = (1 - lambda_mse) * nll_term + lambda_mse * mse_term

    total_loss = combined_loss + regularization_loss
    
    return total_loss, nll_term, mse_term


def empirical_loss_sum(target, predicted_mean, predicted_var, beta):

    l1_loss_mean = F.mse_loss(predicted_mean, target)

    with torch.no_grad():
        empirical_variance = (target - predicted_mean)**2

    l2_loss_sum = torch.sum(torch.abs(predicted_var - empirical_variance))

    total_data_loss = (1 - beta) * l1_loss_mean + beta * l2_loss_sum

    return total_data_loss, l1_loss_mean, l2_loss_sum


def empirical_loss_mean(target, predicted_mean, predicted_var, beta):

    l1_loss_mean = F.mse_loss(predicted_mean, target)

    with torch.no_grad():
        empirical_variance = (target - predicted_mean)**2

    l2_loss_mean = torch.mean(torch.abs(predicted_var - empirical_variance))

    total_data_loss = (1 - beta) * l1_loss_mean + beta * l2_loss_mean

    return total_data_loss, l1_loss_mean, l2_loss_mean    