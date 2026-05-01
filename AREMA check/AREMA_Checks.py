import numpy as np


PSI_TO_PA = 6894.757293168
AREMA_K_PIN_END = 7.0 / 8.0
AREMA_K_BOLTED_OR_WELDED = 3.0 / 4.0
AREMA_MAIN_COMPRESSION_SLENDERNESS_LIMIT = 100.0
AREMA_TENSION_SLENDERNESS_LIMIT = 200.0


def arema_compression_allowable_stress(E, Fy, effective_length, r):
    """AREMA MRE Chapter 15 Table 15-1-11 compression allowable stress."""
    r = max(float(r), 1e-12)
    effective_length = max(float(effective_length), 1e-12)
    effective_slenderness = effective_length / r
    sqrt_E_Fy = np.sqrt(float(E) / float(Fy))

    lower = 0.629 * sqrt_E_Fy
    upper = 5.034 * sqrt_E_Fy

    if effective_slenderness <= lower:
        return 0.55 * Fy, "Compression-Low-Slenderness", effective_slenderness
    if effective_slenderness < upper:
        coeff = ((17500.0 * Fy / E) ** 1.5) * PSI_TO_PA
        return 0.60 * Fy - coeff * effective_slenderness, "Compression-Intermediate", effective_slenderness

    return 0.514 * (np.pi ** 2) * E / (effective_slenderness ** 2), "Compression-Elastic", effective_slenderness


def check_truss_arema(
    Pu,
    Ag,
    An,
    E,
    L,
    r,
    Fy=345e6,
    Fu=427e6,
    Ae=None,
    allowable_stress_factor=1.0,
    effective_length_factor=AREMA_K_BOLTED_OR_WELDED,
    slenderness_limit=AREMA_MAIN_COMPRESSION_SLENDERNESS_LIMIT,
    tension_slenderness_limit=AREMA_TENSION_SLENDERNESS_LIMIT,
):
    """Allowable-stress axial member check for AREMA railway bridge mode.

    Positive Pu is tension, negative Pu is compression. Units are SI.
    Compression allowable stress uses kL/r from Table 15-1-11. The main
    compression member slenderness limit uses bare L/r per Article 1.5.1.
    Tension fracture uses effective net area Ae. If Ae is not supplied, Ae=An
    preserves the source LRFD model assumption U=1.0.
    """
    Ag = max(float(Ag), 1e-12)
    An = max(float(An), 1e-12)
    Ae = An if Ae is None else max(float(Ae), 1e-12)
    Pu = float(Pu)
    L = max(float(L), 1e-12)
    r = max(float(r), 1e-12)

    if Pu >= 0.0:
        gross_allowable = 0.55 * Fy * Ag
        net_allowable = 0.47 * Fu * Ae
        allowable = allowable_stress_factor * min(gross_allowable, net_allowable)
        mode = "Tension-Gross" if gross_allowable <= net_allowable else "Tension-Net"
        stress_dcr = abs(Pu) / max(abs(allowable), 1e-12)
        member_slenderness = L / r
        slenderness_dcr = member_slenderness / max(float(tension_slenderness_limit), 1e-12)
        dcr = max(stress_dcr, slenderness_dcr)
        if slenderness_dcr > 1.0:
            mode = f"{mode}-L/r-Slenderness"
        return bool(dcr <= 1.0), mode, allowable, dcr

    effective_length = effective_length_factor * L
    allowable_stress, mode, effective_slenderness = arema_compression_allowable_stress(E, Fy, effective_length, r)
    allowable = allowable_stress_factor * allowable_stress * Ag
    stress_dcr = abs(Pu) / max(abs(allowable), 1e-12)
    member_slenderness = L / r
    slenderness_dcr = member_slenderness / max(float(slenderness_limit), 1e-12)
    dcr = max(stress_dcr, slenderness_dcr)
    if slenderness_dcr > 1.0:
        mode = f"{mode}-L/r-Slenderness"
    return bool(dcr <= 1.0), mode, allowable, dcr


def arema_outstanding_element_pass(E, Fy, bt=10.7, ht=24.5):
    """Simple AREMA Chapter 15 outstanding-element screen for axial main members.

    The existing app passes width/thickness ratios. This mirrors that style and
    reports the AREMA main-member ratio limit without stopping the analysis.
    """
    limit = 0.43 * np.sqrt(float(E) / float(Fy))
    return bool(bt <= limit and ht <= limit), limit
