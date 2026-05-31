import numpy as np


def arema_member_check(
    P: float,
    A_g: float,
    A_net: float,
    r_min: float,
    L: float,
    F_y: float,
    F_u: float,
    K: float = 1.0,
    E: float = 29000.0,
    member_type: str = "main",
) -> dict:
    """
    AREMA Chapter 15 ASD axial member strength check.
    Automatically selects tension or compression check based on sign of P.

    Parameters
    ----------
    P           : Axial force (kips); positive = tension, negative = compression
    A_g         : Gross cross-sectional area (in²)
    A_net       : Net cross-sectional area (in²); used for tension check
    r_min       : Minimum radius of gyration (in)
    L           : Unbraced length (in)
    F_y         : Steel yield strength (ksi)
    F_u         : Steel ultimate tensile strength (ksi)
    K           : Effective length factor (default 1.0)
    E           : Modulus of elasticity (ksi, default 29000)
    member_type : 'main' (KL/r ≤ 120) or 'secondary' (KL/r ≤ 140)

    Returns
    -------
    dict with all computed values and pass/fail for each check
    """

    results = {}
    results["force_kips"] = P
    results["member_type"] = member_type.capitalize()

    # ------------------------------------------------------------------ #
    # Determine loading type
    # ------------------------------------------------------------------ #
    if P >= 0:
        results["loading"] = "Tension"
        results = _tension_check(P, A_g, A_net, r_min, L, F_y, F_u,
                                  K, member_type, results)
    else:
        results["loading"] = "Compression"
        results = _compression_check(abs(P), A_g, r_min, L, F_y,
                                      K, E, member_type, results)

    return results


# ------------------------------------------------------------------ #
# Tension check — AREMA Ch. 15
# ------------------------------------------------------------------ #
def _tension_check(
    P, A_g, A_net, r_min, L, F_y, F_u, K, member_type, results
) -> dict:
    """
    AREMA tension checks:
      (1) Yielding on gross section  : ft ≤ 0.55 * Fy
      (2) Fracture on net section    : ft ≤ 0.55 * Fu
      (3) Slenderness limit          : L/r ≤ 200
    """

    # Slenderness (L/r, K not applied for tension per AREMA)
    L_r = L / r_min
    results["L_r"] = round(L_r, 2)
    results["L_r_limit"] = 200
    results["slenderness_ok"] = L_r <= 200

    # Allowable stresses
    Ft_gross = 0.55 * F_y       # gross section yielding
    Ft_net   = 0.55 * F_u       # net section fracture

    results["Ft_gross_ksi"] = round(Ft_gross, 3)
    results["Ft_net_ksi"]   = round(Ft_net,   3)

    # Stress demands
    ft_gross = P / A_g
    ft_net   = P / A_net

    results["ft_gross_ksi"] = round(ft_gross, 3)
    results["ft_net_ksi"]   = round(ft_net,   3)

    # DCR
    results["DCR_gross"] = round(ft_gross / Ft_gross, 3)
    results["DCR_net"]   = round(ft_net   / Ft_net,   3)

    results["gross_ok"] = ft_gross <= Ft_gross
    results["net_ok"]   = ft_net   <= Ft_net

    results["PASS"] = all([
        results["slenderness_ok"],
        results["gross_ok"],
        results["net_ok"],
    ])

    return results


# ------------------------------------------------------------------ #
# Compression check — AREMA Ch. 15
# ------------------------------------------------------------------ #
def _compression_check(
    P, A_g, r_min, L, F_y, K, E, member_type, results
) -> dict:
    """
    AREMA compression checks:
      (1) Slenderness limit   : KL/r ≤ 120 (main) or 140 (secondary)
      (2) Johnson parabola    : inelastic buckling when KL/r ≤ Cc
      (3) Euler formula       : elastic buckling when KL/r > Cc
    """

    # Slenderness ratio
    KL_r = (K * L) / r_min
    results["KL_r"] = round(KL_r, 2)

    limit_map = {"main": 120, "secondary": 140}
    KL_r_limit = limit_map.get(member_type.lower(), 120)
    results["KL_r_limit"] = KL_r_limit
    results["slenderness_ok"] = KL_r <= KL_r_limit

    # Critical slenderness ratio
    Cc = np.sqrt((2 * np.pi**2 * E) / F_y)
    results["Cc"] = round(Cc, 2)

    # Allowable compressive stress
    if KL_r <= Cc:
        FS = (5/3) + (3 * KL_r) / (8 * Cc) - (KL_r**3) / (8 * Cc**3)
        Fa = (1 - (KL_r**2) / (2 * Cc**2)) * F_y / FS
        results["buckling_mode"] = "Inelastic (Johnson)"
    else:
        FS = 23 / 12
        Fa = (12 * np.pi**2 * E) / (23 * KL_r**2)
        results["buckling_mode"] = "Elastic (Euler)"

    results["FS"]     = round(FS, 3)
    results["Fa_ksi"] = round(Fa, 3)

    # Axial stress demand
    fa = P / A_g
    results["fa_ksi"]   = round(fa,  3)
    results["axial_DCR"] = round(fa / Fa, 3)
    results["axial_ok"] = fa <= Fa

    results["PASS"] = all([
        results["slenderness_ok"],
        results["axial_ok"],
    ])

    return results


# ------------------------------------------------------------------ #
# Pretty printer
# ------------------------------------------------------------------ #
def print_arema_report(res: dict) -> None:
    width = 54
    tag   = "✓  PASS" if res["PASS"] else "✗  FAIL"
    print("=" * width)
    print("  AREMA Ch.15 – Axial Member Strength Check")
    print("=" * width)
    print(f"  Member type   : {res['member_type']}")
    print(f"  Loading       : {res['loading']}  "
          f"(P = {res['force_kips']:+.1f} kips)")
    print("-" * width)

    if res["loading"] == "Tension":
        print(f"  Slenderness  L/r          : {res['L_r']:>8.2f}"
              f"  (limit {res['L_r_limit']})")
        print(f"  Slenderness OK?           : "
              f"{'✓ PASS' if res['slenderness_ok'] else '✗ FAIL'}")
        print("-" * width)
        print(f"  Gross area check")
        print(f"    Demand  ft,gross (ksi)  : {res['ft_gross_ksi']:>8.3f}")
        print(f"    Allow.  Ft,gross (ksi)  : {res['Ft_gross_ksi']:>8.3f}"
              f"  (0.55 Fy)")
        print(f"    DCR                     : {res['DCR_gross']:>8.3f}"
              f"  (limit 1.0)")
        print(f"    Gross section OK?       : "
              f"{'✓ PASS' if res['gross_ok'] else '✗ FAIL'}")
        print("-" * width)
        print(f"  Net area check")
        print(f"    Demand  ft,net   (ksi)  : {res['ft_net_ksi']:>8.3f}")
        print(f"    Allow.  Ft,net   (ksi)  : {res['Ft_net_ksi']:>8.3f}"
              f"  (0.55 Fu)")
        print(f"    DCR                     : {res['DCR_net']:>8.3f}"
              f"  (limit 1.0)")
        print(f"    Net section OK?         : "
              f"{'✓ PASS' if res['net_ok'] else '✗ FAIL'}")

    else:  # Compression
        print(f"  Slenderness  KL/r         : {res['KL_r']:>8.2f}"
              f"  (limit {res['KL_r_limit']})")
        print(f"  Slenderness OK?           : "
              f"{'✓ PASS' if res['slenderness_ok'] else '✗ FAIL'}")
        print(f"  Cc                        : {res['Cc']:>8.2f}")
        print(f"  Buckling mode             : {res['buckling_mode']}")
        print(f"  Factor of Safety (FS)     : {res['FS']:>8.3f}")
        print("-" * width)
        print(f"  Demand  fa  (ksi)         : {res['fa_ksi']:>8.3f}")
        print(f"  Allow.  Fa  (ksi)         : {res['Fa_ksi']:>8.3f}")
        print(f"  Axial DCR                 : {res['axial_DCR']:>8.3f}"
              f"  (limit 1.0)")
        print(f"  Axial OK?                 : "
              f"{'✓ PASS' if res['axial_ok'] else '✗ FAIL'}")

    print("=" * width)
    print(f"  OVERALL: {tag}")
    print("=" * width)


# ------------------------------------------------------------------ #
# Example usage
# ------------------------------------------------------------------ #
if __name__ == "__main__":

    common = dict(
        A_g=28.2, A_net=24.5, r_min=2.50,
        L=180.0,  F_y=50.0,   F_u=65.0,
        K=1.0,    member_type="main",
    )

    # Tension member  (positive P)
    res_t = arema_member_check(P=+350.0, **common)
    print_arema_report(res_t)

    print()

    # Compression member  (negative P)
    res_c = arema_member_check(P=-400.0, **common)
    print_arema_report(res_c)