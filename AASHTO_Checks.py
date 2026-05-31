import numpy as np


def check_truss_lrfd(Pu, Ag, An, E, KL, r, Fy=345e6, Fu=427e6, Rp=1.0):
    """AASHTO LRFD axial member check matching Check_Truss_LRFD.m."""
    phi_ty = 0.95
    phi_cb = 0.95
    phi_uf = 0.80
    u_shear_lag = 1.0

    r = max(float(r), 1e-12)
    KL = max(float(KL), 1e-12)
    KLr = KL / r

    if Pu >= 0.0:
        Pny = Fy * Ag
        pr_yield = phi_ty * Pny

        Pnu = Fu * An * Rp * u_shear_lag
        pr_fracture = phi_uf * Pnu

        if pr_yield <= pr_fracture:
            Pn = Pny
            phi = phi_ty
            phiPn = pr_yield
            mode = "Tension-Yield"
        else:
            Pn = Pnu
            phi = phi_uf
            phiPn = pr_fracture
            mode = "Tension-Fracture"
    else:
        Po = Fy * Ag
        Pe = (np.pi ** 2 * E * Ag) / (KLr ** 2)
        ratio = Po / Pe

        if ratio <= 2.25:
            Pn = (0.658 ** ratio) * Po
            mode = "Compression-Inelastic"
        else:
            Pn = 0.877 * Pe
            mode = "Compression-Elastic"

        phi = phi_cb
        phiPn = phi * Pn

    DCR = abs(Pu) / abs(phiPn) if phiPn != 0.0 else np.inf
    return bool(DCR <= 1.0), mode, Pn, phi, phiPn, DCR


def local_buckling_pass(E, Fy, bt=10.7, ht=24.5):
    lambda_r = 1.28 * np.sqrt(E / Fy)
    return bt <= lambda_r and ht <= lambda_r, lambda_r
