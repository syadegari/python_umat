import numpy as np

import torch
import torch.optim as optim
from .umat import *
from .constants import consts
from copy import deepcopy


def forward_pass(dgamma, ds, s0, beta0, F_p0, F1, slip_sys, elas_stif, dt):
    H = get_H_matrix(get_ks(ds, s0))
    ws = get_ws(H)
    beta1 = get_beta(dgamma, ws, beta0)
    F_p1 = plastic_def_grad(dgamma, slip_sys, F_p0)
    F_e1 = F1 @ torch.linalg.inv(F_p1)
    #
    g_d = get_gd(beta1, ws)
    g_m = get_gm(F_e1, slip_sys, elas_stif)
    r_I = get_r_I(ds, H, dgamma)
    r_II = get_r_II(g_m + g_d + consts.g_th, s0 + ds, g_m, dgamma, dt)
    return r_I, r_II, F_p1, beta1


def driver(F_final, n_steps, orientation):

    f = lambda t: F_init + t * (F_final - F_init)

    times = np.linspace(0, 1, num=n_steps + 1)
    F_init = torch.eye(3)

    slip_sys, elas_stif = material_properties_bcc(orientation)

    Fp_hist, gamma_hist, s_hist, beta_hist = [], [], [], []
    Fp_init, gamma_init, s_init, beta_init = sdvini()
    Fp_hist.append(Fp_init)
    gamma_hist.append(gamma_init)
    s_hist.append(s_init)
    beta_hist.append(beta_init)

    for t0, t1 in zip(times[:-1], times[1:]):
        F0, F1 = f(t0), f(t1)
        #
        dgamma = torch.zeros(24, requires_grad=True)
        ds = torch.zeros(24, requires_grad=True)
        # optimizer = optim.SGD([dgamma, ds], lr=1e-7)
        #
        # import pdb; pdb.set_trace()
        F_p0 = Fp_hist[-1].clone().detach()
        gamma0 = gamma_hist[-1].clone().detach()
        s0 = s_hist[-1].clone().detach()
        beta0 = deepcopy(beta_hist[-1])

        iter_num = 0
        converged = False
        penalty_factor = 100
        eta = 5e-6

        while True:
            r_I, r_II, F_p1, beta1 = forward_pass(dgamma,
                                    ds,
                                    s0,
                                    beta0,
                                    F_p0,
                                    F1,
                                    slip_sys,
                                    elas_stif,
                                    t1 - t0)
            loss, r_I_norm, r_II_norm, rI_rII_ratio = func_without_penalty(r_I, r_II)
            # loss, r_I_norm, r_II_norm, rI_rII_ratio = func_penalty(r_I, r_II, dgamma, s0 + ds, penalty_factor)
            #
            if loss < 1e-6:
                Fp_hist.append(F_p1.clone().detach())
                gamma_hist.append(gamma0 + dgamma.clone().detach())
                s_hist.append(s0 + ds.clone().detach())
                beta_hist.append(beta0 + beta1.clone().detach())
                converged = True
                dgamma, ds = torch.zeros(24, requires_grad=True), torch.zeros(24, requires_grad=True)
            else:
                # optimizer.zero_grad()
                # loss.backward()
                # optimizer.step()
                dgamma.grad, ds.grad = None, None
                # import pdb; pdb.set_trace()
                loss.backward()
                dgamma = torch.tensor(torch.clamp(-eta * dgamma.grad, min=0) + dgamma.detach(), requires_grad=True)
                ds = torch.tensor(-eta * ds.grad + ds.detach(), requires_grad=True)

            iter_num += 1
            if iter_num % 1 == 0:
                penalty_factor = penalty_factor * 1.2
                print(penalty_factor)
            print(f't={t0:.3f}, iter={iter_num}, loss={loss.item():e}')
            print(f'r_I={r_I_norm:e}, r_II={r_II_norm:e}, |rI|/|rII|={rI_rII_ratio.item():.3f}')
            print(dgamma)

            if converged:
                break
