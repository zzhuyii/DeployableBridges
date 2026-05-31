import numpy as np


def arema_member_check(
    Pu,    Ag,    An,    E,    KL,    r,
    Fy=345e6, Fu=427e6, Rp=1.0
):

    
    # Tension forces
    if Pu >= 0:
        L_r = KL / r
        if L_r > 200:
            print(' Member is too slender')
            
        # Equation from Table 15-1-11    
        # Allowable stresses
        Ft_gross = 0.55 * Fy       # gross section yielding
        Ft_net   = 0.55 * Fu       # net section fracture
  
        # Stress demands
        ft_gross = Pu / Ag
        ft_net   = Pu / An  
    
        # DCR
        dcr_g = ft_gross / Ft_gross
        dcr_n = ft_net   / Ft_net 
    
        dcr=max(dcr_g,dcr_n)
    
        return bool(dcr <= 1.0), dcr
            

    # Compression forces
    else:
        KL_r = (KL) / r

        # slenderness limit from Table 15-1-11
        limit1=0.629/np.sqrt(Fy/E)
        limit2=5.034/np.sqrt(Fy/E)
        
        Epsi=E*0.000145037738
        Fypsi=Fy*0.000145037738
        
        # Equation from Table 15-1-11
        if KL_r < limit1:
            Fallow=0.55*Fy
        elif KL_r < limit2:
            Fallow=0.60*Fy-(17500.0*Fypsi/Epsi)**(1.5)*KL_r/0.000145037738
        else:
            Fallow=(0.514 * 3.14**2)/(KL_r ** 2)*E

        dcr = - Pu / (Fallow * Ag)
        
         
    return bool(dcr <= 1.0), dcr




