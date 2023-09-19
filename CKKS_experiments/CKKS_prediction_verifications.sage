from multiprocessing import Pool
from numpy.random import seed as np_seed

load("../framework/LWE.sage")
load("../framework/LWR.sage")


def det_full(s_s, s_e, s_eps, s_hs, s_he, n, t):
    outer_coeff = 4*ln(s_s) + 4*ln(s_e) + (4*t - 8)*ln(s_eps)

    inner_coeff_1 = (7/4)*t*(t-1)*(n**4)*(s_hs**4)*(s_he**4) + t*(n**2)*(s_eps**4)*((s_hs**4) / (s_e**4) + (s_he**4)/(s_s**4))
    inner_coeff_2 = (t*(t-1)*(n**2)*(s_hs**2)*(s_he**2) + t*n*(s_eps**2)*((s_hs**2 / s_e**2) + (s_he**2 / s_s**2)) + s_eps**4 / (s_s**2 * s_e**2))

    return (n/2)*(outer_coeff + ln(inner_coeff_1 + inner_coeff_2**2)) - 2 * n * (t * ln(s_eps) + ln(s_e) + ln(s_s))


def negacyclic(col):
    row = [-1 * i for i in col[:0:-1]]
    return matrix.toeplitz(col, row)


def one_experiment(args):
    id = args['id']
    n = args['n']
    Sigma = args['Sigma']
    Sigma_eps = args['Sigma_eps']
    D_e1 = args['D_e1']
    D_v = args['D_v']
    t = args['t']
    
    set_random_seed(id)
    np_seed(seed=id)
    
    blocks = []
    for i in range(t):
        E1 = negacyclic([draw_from_distribution(D_e1) for j in range(n)])
        V = negacyclic([draw_from_distribution(D_v) for j in range(n)])
        blocks.append([E1, V])
    H = block_matrix(blocks)
    
    Sigma_prime = Sigma - Sigma * H.T * (H * Sigma * H.T + Sigma_eps)^-1 * H * Sigma
    return det(Sigma_prime)


def run_experiments(experiments, threads, *, n, sigma_s, sigma_e, sigma_eps, sigma_h_s, sigma_h_e, t):
    
    # static parameters for the experiments
    D_e1 = build_Gaussian_law(sigma_h_e, 20)
    D_v = build_Gaussian_law(sigma_h_s, 20)
    Sigma = diagonal_matrix((sigma_s,) * n + (sigma_e,) * n)
    Sigma_eps = (sigma_eps**2) * identity_matrix(t * n)
    theoretical_det = det_full(sigma_s, sigma_e, sigma_eps, sigma_h_s, sigma_h_e, n, t)
    
    print(theoretical_det)
    print(ln(det(Sigma)))
    print()
    
    # run many experiments with threading
    with Pool(threads) as p:
        args = [{'id': i, 'n': n, 'Sigma': Sigma, 'Sigma_eps': Sigma_eps, 'D_e1': D_e1, 'D_v': D_v, 't': t} for i in range(experiments)]
        results = []
        for result in p.imap_unordered(one_experiment, args):
            print(ln(result))
            results.append(result)
    
    print()
    print(f'Theoretical: {theoretical_det}')
    print(f'Experimental: {ln(sum(results) / experiments)}')
