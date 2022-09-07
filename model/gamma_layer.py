# from https://github.com/usnistgov/COSMOSAC
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from data.utils import get_params

q0 = 79.53  # [A^2]
r0 = 66.69  # [A^3]
z_coordination = 10
c_hb = 85580.0  # kcal A^4 / mol/e^2
R = 8.3144598 / 4184  # 0.001987 # but really: 8.3144598/4184
sigma_hb = 0.0084
EPS = 3.667  # (LIN AND SANDLER USE A CONSTANT FPOL WHICH YIELDS EPS=3.68)
AEFFPRIME = 7.5
EO = 2.395e-4
FPOL = (EPS - 1.0) / (EPS + 0.5)
ALPHA = (0.3 * AEFFPRIME ** (1.5)) / (EO)
alpha_prime = FPOL * ALPHA

sigma_tabulated = torch.arange(-0.03, 0.031, 0.001)
sigma_m = sigma_tabulated.repeat((len(sigma_tabulated), 1))
sigma_n = sigma_m.T
sigma_acc = torch.tril(sigma_n) + torch.triu(sigma_m, 1)
sigma_don = torch.tril(sigma_m) + torch.triu(sigma_n, 1)

DELTAW = (alpha_prime / 2) * (sigma_m + sigma_n) ** 2 + c_hb * torch.max(torch.zeros_like(sigma_acc), sigma_acc - sigma_hb) * torch.min(torch.zeros_like(sigma_don),
                                                                                                                sigma_don + sigma_hb)


solvents_volumes = {"O": 25.8, "C[N+](=O)[O-]": 71.4, "CS(C)=O": 98.4, "CCCCCC": 145.3} #A^2
solvents_sigma_profiles = {"O": torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                0.0, 0.1469999873352021, 0.984999980335459, 2.249999968858116,
                                2.691999989052083, 2.3549999719283154, 2.507999969132144,
                                2.8479999911902247, 2.0980000298426478, 0.9249999664645762,
                                0.5020000340648496, 0.8820000088265967, 1.1859999716729177,
                                0.7839999889980006, 0.4659999748530897, 0.4290000270837368,
                                0.2849999598674643, 0.525000015350996, 0.768999964326434,
                                0.4230000172151101, 0.2839999864944877, 0.6840000224234515,
                                1.0969999666472616, 1.102999976515888, 0.8910000236295366,
                                0.7529999662818907, 0.8910000236295366, 1.021000039547219,
                                0.8519999594834633, 1.0989999982085985, 1.4970000025762251,
                                1.586000007601881, 1.6360000332968476, 1.8560000276131623,
                                2.3080000359830457, 1.9759999705395448, 0.7960000087352539,
                                0.0959999882672592, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                0.0, 0.0]),
                           "C[N+](=O)[O-]": torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0359999999476963,
                                0.396999999580612, 2.3999999965130927, 5.845000010134141,
                                7.526000005540657, 6.36500000308246, 4.805000003250221,
                                5.1399999956803, 5.846999993236473, 4.158000002458783,
                                1.7959999996992322, 0.9769999997873068, 0.9220000006542384,
                                0.9960000008615346, 1.3130000005583768, 1.4760000001641531,
                                1.3769999992061557, 1.8419999993175904, 3.2699999915763143,
                                5.046999994398775, 6.290999998677707, 6.60200000499426,
                                5.986000004629997, 4.43500000221374, 2.774000005728803,
                                1.8290000000185644, 1.042000000479893, 0.2749999998627996,
                                0.0010000000006458, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                           "CS(C)=O": torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0720000529689353,
                                0.39099988463715, 2.200000099444, 5.997000133726578,
                                9.541999982617885, 11.432999947740006, 10.75999994864433,
                                8.313000070501051, 7.710999875630264, 9.30199999206104,
                                8.869999953256842, 4.891000126092069, 2.314999967039839,
                                2.6379999566560017, 2.479999995423845, 1.7120000814438312,
                                1.4470000104929053, 1.0729998903539428, 0.5510000643479869,
                                0.3269998685546978, 0.4179999045005008, 0.6780001267782575,
                                0.7150001229973589, 1.1860000974899274, 1.896000092804705,
                                2.016000088083126, 2.855000085297441, 4.413000084447651,
                                4.036000055104788, 1.7400000710418158, 0.2429998997607441, 0.0,
                                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),

                           "CCCCCC": torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                0.0, 0.220999861877722, 3.471000129780297, 12.924999875518871,
                                24.2130002734691, 27.44800019205504, 23.05599986321451,
                                19.587000220403155, 18.50699984173372, 14.602999977591269,
                                6.324000070402117, 0.785999807669221, 0.0, 0.0, 0.0, 0.0, 0.0,
                                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

}


# COSMO surface area derived from the sum of sigma profile
solvents_areas = {'O': 43.484999853872516, 'C[N+](=O)[O-]': 90.77000001225453,
                  'CS(C)=O': 112.22100052993952, 'CCCCCC': 151.14100011371502}



def get_Gamma(T, psigma):
    """
    Get the value of Γ (capital gamma) for the given sigma profile
    """
    Gamma = torch.ones_like(psigma)
    AA = torch.exp(-DELTAW / (R * T)) * psigma  # constant and can be pre-calculated outside of the loop
    for i in range(1):
        Gammanew = 1 / torch.sum(AA * Gamma, dim=1)
        difference = torch.abs((Gamma - Gammanew) / Gamma)
        Gamma = (Gammanew + Gamma) / 2
        if torch.max(difference) < 1e-8:
            break
        else:
            pass
    return Gamma


def get_lngamma_resid(T, psigma_mix, prof, area, lnGamma_mix=None):
    """
    The residual contribution to ln(γ_i)
    """
    # For the mixture
    if lnGamma_mix is None:
        lnGamma_mix = torch.log(get_Gamma(T, psigma_mix))
    # For this component
    psigma = prof / area
    A_i = area
    lnGammai = torch.log(get_Gamma(T, psigma))
    lngammai = A_i/AEFFPRIME*torch.sum(psigma*(lnGamma_mix - lnGammai))
    return lngammai


def get_lngamma_comb(T, x, i, volumes, areas):
    """
    The combinatorial part of ln(γ_i)
    """
    A = areas
    q = A / q0
    r = volumes / r0
    theta_i = x[i] * q[i] / torch.dot(x, q)
    phi_i = x[i] * r[i] / torch.dot(x, r)
    l = z_coordination / 2 * (r - q) - (r - 1)
    return (torch.log(phi_i / x[i]) + z_coordination / 2 * q[i] * torch.log(theta_i / phi_i)
            + l[i] - phi_i / x[i] * torch.dot(x, l))


def get_lngamma(T, x, i, psigma_mix, profs, volumes, areas, lnGamma_mix=None):
    """
    Sum of the contributions to ln(γ_i)
    """
    return (get_lngamma_resid(T, psigma_mix, profs[i], areas[i], lnGamma_mix=lnGamma_mix)
            + get_lngamma_comb(T, x, i, volumes, areas))

class GammaLayer(nn.Module):
    def __init__(self):
        super(GammaLayer, self).__init__()


def activ_coef_generates(smiles, sigma_profile, path):
    if isinstance(sigma_profile, torch.Tensor):
        sigma_profile = sigma_profile.cpu()
    _mask = sigma_profile >= 0
    sigma_profile *= _mask
    volume, sigma_norm = get_params(smiles, path)
    sigma_profile *= sigma_norm
    area = torch.sum(sigma_profile)
    results = []

    for solvent in solvents_areas.keys():
        profs = [sigma_profile, solvents_sigma_profiles[solvent]]
        areas = torch.tensor([area, solvents_areas[solvent]], dtype=torch.float)
        volumes = torch.tensor([volume, solvents_volumes[solvent]], dtype=torch.float)
        x = torch.tensor([0.235, 1 - 0.235], dtype=torch.float)
        T = 623.15
        i = 0

        psigma_mix = sum([x[i] * profs[i] for i in range(2)]) / sum([x[i] * areas[i] for i in range(2)])
        lnGamma_mix = torch.log(get_Gamma(T, psigma_mix))
        lngamma = get_lngamma(T, x, i, psigma_mix, profs, volumes, areas, lnGamma_mix=lnGamma_mix)
        results.append(lngamma)
    return results
