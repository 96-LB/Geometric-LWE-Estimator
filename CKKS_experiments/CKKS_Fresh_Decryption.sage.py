

# This file was *autogenerated* from the file CKKS_Fresh_Decryption.sage
from sage.all_cmdline import *   # import sage library

_sage_const_128 = Integer(128); _sage_const_1024 = Integer(1024); _sage_const_25 = Integer(25); _sage_const_192 = Integer(192); _sage_const_17 = Integer(17); _sage_const_256 = Integer(256); _sage_const_13 = Integer(13); _sage_const_2048 = Integer(2048); _sage_const_51 = Integer(51); _sage_const_35 = Integer(35); _sage_const_27 = Integer(27); _sage_const_4096 = Integer(4096); _sage_const_101 = Integer(101); _sage_const_70 = Integer(70); _sage_const_54 = Integer(54); _sage_const_8192 = Integer(8192); _sage_const_202 = Integer(202); _sage_const_141 = Integer(141); _sage_const_109 = Integer(109); _sage_const_16384 = Integer(16384); _sage_const_411 = Integer(411); _sage_const_284 = Integer(284); _sage_const_220 = Integer(220); _sage_const_32768 = Integer(32768); _sage_const_827 = Integer(827); _sage_const_571 = Integer(571); _sage_const_443 = Integer(443); _sage_const_1 = Integer(1); _sage_const_2 = Integer(2); _sage_const_4 = Integer(4); _sage_const_8 = Integer(8); _sage_const_7 = Integer(7); _sage_const_3 = Integer(3); _sage_const_3p2 = RealNumber('3.2'); _sage_const_0 = Integer(0); _sage_const_1000 = Integer(1000); _sage_const_30 = Integer(30); _sage_const_12 = Integer(12); _sage_const_0p0001 = RealNumber('0.0001')
reset()

load("../framework/LWE.sage")

params = [
    {"target": _sage_const_128 ,
     "n": _sage_const_1024 ,
     "m": _sage_const_1024 ,
     "logq": _sage_const_25 
     },
    {"target": _sage_const_192 ,
     "n": _sage_const_1024 ,
     "m": _sage_const_1024 ,
     "logq": _sage_const_17 
     },
    {"target": _sage_const_256 ,
     "n": _sage_const_1024 ,
     "m": _sage_const_1024 ,
     "logq": _sage_const_13 
     },
    {"target": _sage_const_128 ,
     "n": _sage_const_2048 ,
     "m": _sage_const_2048 ,
     "logq": _sage_const_51 
     },
    {"target": _sage_const_192 ,
     "n": _sage_const_2048 ,
     "m": _sage_const_2048 ,
     "logq": _sage_const_35 
     },
    {"target": _sage_const_256 ,
     "n": _sage_const_2048 ,
     "m": _sage_const_2048 ,
     "logq": _sage_const_27 
     },
    {"target": _sage_const_128 ,
     "n": _sage_const_4096 ,
     "m": _sage_const_4096 ,
     "logq": _sage_const_101 
     },
    {"target": _sage_const_192 ,
     "n": _sage_const_4096 ,
     "m": _sage_const_4096 ,
     "logq": _sage_const_70 
     },
    {"target": _sage_const_256 ,
     "n": _sage_const_4096 ,
     "m": _sage_const_4096 ,
     "logq": _sage_const_54 
     },
    {"target": _sage_const_128 ,
     "n": _sage_const_8192 ,
     "m": _sage_const_8192 ,
     "logq": _sage_const_202 
     },
    {"target": _sage_const_192 ,
     "n": _sage_const_8192 ,
     "m": _sage_const_8192 ,
     "logq": _sage_const_141 
     },
    {"target": _sage_const_256 ,
     "n": _sage_const_8192 ,
     "m": _sage_const_8192 ,
     "logq": _sage_const_109 
     },
    {"target": _sage_const_128 ,
     "n": _sage_const_16384 ,
     "m": _sage_const_16384 ,
     "logq": _sage_const_411 
     },
    {"target": _sage_const_192 ,
     "n": _sage_const_16384 ,
     "m": _sage_const_16384 ,
     "logq": _sage_const_284 
     },
    {"target": _sage_const_256 ,
     "n": _sage_const_16384 ,
     "m": _sage_const_16384 ,
     "logq": _sage_const_220 
     },
    {"target": _sage_const_128 ,
     "n": _sage_const_32768 ,
     "m": _sage_const_32768 ,
     "logq": _sage_const_827 
     },
    {"target": _sage_const_192 ,
     "n": _sage_const_32768 ,
     "m": _sage_const_32768 ,
     "logq": _sage_const_571 
     },
    {"target": _sage_const_256 ,
     "n": _sage_const_32768 ,
     "m": _sage_const_32768 ,
     "logq": _sage_const_443 
     }
]

# Hc function in CKKS error estimation section
def Hc(alpha, N):
        return sqrt(- (log(_sage_const_1  - (_sage_const_1  - alpha)**(_sage_const_2 /N))/log(_sage_const_2 )))

# Calculate the denominator of the expectation of the volume resulting from integrating t hints.
def det_denom(s_s, s_e, s_eps, s_hs, s_he, n, t):
        outer_coeff = _sage_const_4 *ln(s_s) + _sage_const_4 *ln(s_e) + (_sage_const_4 *t - _sage_const_8 )*ln(s_eps)

        inner_coeff_1 = (_sage_const_7 /_sage_const_4 )*t*(t-_sage_const_1 )*(n**_sage_const_4 )*(s_hs**_sage_const_4 )*(s_he**_sage_const_4 ) + t*(n**_sage_const_2 )*(s_eps**_sage_const_4 )*((s_hs**_sage_const_4 ) / (s_e**_sage_const_4 ) + (s_he**_sage_const_4 )/(s_s**_sage_const_4 ))
        inner_coeff_2 = (t*(t-_sage_const_1 )*(n**_sage_const_2 )*(s_hs**_sage_const_2 )*(s_he**_sage_const_2 ) + t*n*(s_eps**_sage_const_2 )*((s_hs**_sage_const_2  / s_e**_sage_const_2 ) + (s_he**_sage_const_2  / s_s**_sage_const_2 )) + s_eps**_sage_const_4  / (s_s**_sage_const_2  * s_e**_sage_const_2 ))

        return (n/_sage_const_2 )*(outer_coeff + ln(inner_coeff_1 + inner_coeff_2**_sage_const_2 ))

for param in params:
    print(f'Parameter set: n = {param["n"]}, target {param["target"]}-bit security')
    print(f'log(q) = {param["logq"]}')
    print("==========================================")

    # Calculate Volume of starting lattice and ellipsoid, use m*log(q) for bvol
    Bvol = param["m"] * (param["logq"]*ln(_sage_const_2 ))
    Svol_orig = RR(param["n"]*log(_sage_const_2 /_sage_const_3 ) + param["m"]*log(_sage_const_3p2 *_sage_const_3p2 ))

    dvol_orig = Bvol - Svol_orig / _sage_const_2 

    # Calculate BKZ Beta, delta for starting lattice
    beta_orig, delta_orig = compute_beta_delta(
            (param["m"]+param["n"]+_sage_const_1 ), dvol_orig, probabilistic=False, tours=_sage_const_1 , verbose=_sage_const_0 ,
            ignore_lift_proba=False, number_targets=_sage_const_1 , lift_union_bound=False)

    print(f"BKZ Beta Estimate (Initial): {beta_orig: .2f} bikz ~ {beta_orig*0.265: .2f} bits")

    # Calculate estimate after t micciancio style decryption hints
    adv_queries = _sage_const_1000  # adv_queries = t
    stat_security = _sage_const_30 
    std_fresh = sqrt((_sage_const_4 /_sage_const_3 )*param["n"] + _sage_const_1 )*_sage_const_3p2     
    bits_fresh = (_sage_const_1 /_sage_const_2 )*log(param["n"]*(std_fresh**_sage_const_2  + (_sage_const_1 /_sage_const_12 )))/log(_sage_const_2 ) + log(Hc(_sage_const_0p0001 , param["n"]))/log(_sage_const_2 )

    # Calculate ciphertext noise estimate
    sigma_eps = sqrt(_sage_const_12 *adv_queries)*_sage_const_2 **(stat_security / _sage_const_2 )*std_fresh
    sigma_eps_bits = sqrt(_sage_const_12 *adv_queries)*_sage_const_2 **(stat_security / _sage_const_2 )*sqrt(bits_fresh)

    num = _sage_const_2 *param["n"]*adv_queries*log(sigma_eps) + _sage_const_2 *param["n"]*(log(sqrt(_sage_const_2 /_sage_const_3 )) + log(_sage_const_3p2 ))
    denom = det_denom(sqrt(_sage_const_2 /_sage_const_3 ), _sage_const_3p2 , sigma_eps, _sage_const_3p2 , sqrt(_sage_const_2 /_sage_const_3 ), param["n"], adv_queries)

    num_bits = _sage_const_2 *param["n"]*adv_queries*log(sigma_eps_bits) + _sage_const_2 *param["n"]*(log(sqrt(_sage_const_2 /_sage_const_3 )) + log(_sage_const_3p2 ))
    denom_bits = det_denom(sqrt(_sage_const_2 /_sage_const_3 ), _sage_const_3p2 , sigma_eps_bits, _sage_const_3p2 , sqrt(_sage_const_2 /_sage_const_3 ), param["n"], adv_queries)

    num_fresh = _sage_const_2 *param["n"]*adv_queries*log(std_fresh) + _sage_const_2 *param["n"]*(log(sqrt(_sage_const_2 /_sage_const_3 )) + log(_sage_const_3p2 ))
    denom_fresh = det_denom(sqrt(_sage_const_2 /_sage_const_3 ), _sage_const_3p2 , std_fresh, _sage_const_3p2 , sqrt(_sage_const_2 /_sage_const_3 ), param["n"], adv_queries)

    # Calculate (expected) volume for ellipsoid after t hints
    Svol_t_hints = RR(num - denom)
    Svol_t_hints_bits = RR(num_bits - denom_bits)
    Svol_t_hints_fresh = RR(num_fresh - denom_fresh)


    dvol_t_hints = Bvol - Svol_t_hints / _sage_const_2 
    dvol_t_hints_bits = Bvol - Svol_t_hints_bits / _sage_const_2 
    dvol_t_hints_fresh = Bvol - Svol_t_hints_fresh / _sage_const_2 

    # Calculate BKZ Beta, delta for lattice after t hints
    beta_t_hints, delta_t_hints = compute_beta_delta(
            (param["m"]+param["n"]+_sage_const_1 ), dvol_t_hints, probabilistic=False, tours=_sage_const_1 , verbose=_sage_const_0 ,
            ignore_lift_proba=False, number_targets=_sage_const_1 , lift_union_bound=False)

    beta_t_hints_bits, delta_t_hints_bits = compute_beta_delta(
            (param["m"]+param["n"]+_sage_const_1 ), dvol_t_hints_bits, probabilistic=False, tours=_sage_const_1 , verbose=_sage_const_0 ,
            ignore_lift_proba=False, number_targets=_sage_const_1 , lift_union_bound=False)

    beta_t_hints_fresh, delta_t_hints_fresh = compute_beta_delta(
            (param["m"]+param["n"]+_sage_const_1 ), dvol_t_hints_fresh, probabilistic=False, tours=_sage_const_1 , verbose=_sage_const_0 ,
            ignore_lift_proba=False, number_targets=_sage_const_1 , lift_union_bound=False)

    print(f"BKZ Beta Estimate ({adv_queries} Hint): {beta_t_hints: .2f} bikz ~ {beta_t_hints*0.265: .2f} bits")
    print(f"BKZ Beta Estimate ({adv_queries} Hint, sigma_eps measured in bits): {beta_t_hints_bits: .2f} bikz ~ {beta_t_hints_bits*0.265 :.2f} bits")
    print(f"BKZ Beta Estimate ({adv_queries} Hint, sigma_eps = rho_fresh): {beta_t_hints_fresh: .3f} bikz ~ {beta_t_hints_fresh*0.265: .2f} bits")
    print("")

