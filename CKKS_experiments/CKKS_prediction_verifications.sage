from functools import wraps
from multiprocessing import Pool

from numpy.random import seed as np_seed

from typing import Callable, TypeVar

load("../framework/LWE.sage")
load("../framework/LWR.sage")

### HELPERS

def det_full(s_s, s_e, s_eps, s_hs, s_he, n, t):
    outer_coeff = 4*ln(s_s) + 4*ln(s_e) + (4*t - 8)*ln(s_eps)

    inner_coeff_1 = (7/4)*t*(t-1)*(n**4)*(s_hs**4)*(s_he**4) + t*(n**2)*(s_eps**4)*((s_hs**4) / (s_e**4) + (s_he**4)/(s_s**4))
    inner_coeff_2 = (t*(t-1)*(n**2)*(s_hs**2)*(s_he**2) + t*n*(s_eps**2)*((s_hs**2 / s_e**2) + (s_he**2 / s_s**2)) + s_eps**4 / (s_s**2 * s_e**2))

    return (n/2)*(outer_coeff + ln(inner_coeff_1 + inner_coeff_2**2)) - (2*n)*(t*ln(s_eps) + ln(s_s) + ln(s_e))


def negacyclic(col):
    row = [-1 * i for i in col[:0:-1]]
    return matrix.toeplitz(col, row)


def sample_polynomial_matrix(D, n):
    return negacyclic([draw_from_distribution(D) for _ in range(n)])


def build_H(D_e1, D_v, t, n):
    return block_matrix([[sample_polynomial_matrix(D_e1, n), sample_polynomial_matrix(D_v, n)] for _ in range(t)])

### EXPERIMENTS

T = TypeVar('T')
def experiment(f: Callable[..., T]) -> Callable[[dict], tuple[int, T]]:
    @wraps(f)
    def wrapper(kwargs):
        id = kwargs.pop('id')
        set_random_seed(id)
        np_seed(seed=id)
        return id, f(**kwargs)
    return wrapper


@experiment
def experiment_slow(n, Sigma, Sigma_eps, D_e1, D_v, t):
    H = build_H(D_e1, D_v, t, n)
    Sigma_prime = Sigma - Sigma * H.T * (H * Sigma * H.T + Sigma_eps)^-1 * H * Sigma
    return 1/det(Sigma_prime)


@experiment
def experiment_fast(n, rtSigma, sigma_s, sigma_e, sigma_eps, D_e1, D_v, t):
    H = build_H(D_e1, D_v, t, n)
    G = identity_matrix(2 * n) + 1/sigma_eps^2 * rtSigma * H.T * H * rtSigma
    return det(G) / sigma_s^(2 * n) / sigma_e^(2 * n)


def run_experiments(experiments, threads, *, n, sigma_s, sigma_e, sigma_eps, sigma_h_e, t, slow=False):
    sigma_h_s = sqrt(2/3)
    D_e1 = build_Gaussian_law(sigma_h_e, 20)
    D_v = {-1: 1/3, 0: 1/3, 1: 1/3} # build_Gaussian_law(sigma_h_s, 20)
    
    original_det = -2 * n * ln(sigma_s * sigma_e)
    theoretical_det = det_full(sigma_s, sigma_e, sigma_eps, sigma_h_s, sigma_h_e, n, t)
    
    print(f'Original: {original_det}')
    print(f'Theoretical: {theoretical_det}')
    print()
    
    # run many experiments with threading
    with Pool(threads) as p:
        
        slow_results = {}
        if slow:
            print()
            print('Slow!')
            args = [{
                'id': i,
                'n': n,
                'Sigma': diagonal_matrix((sigma_s**2,) * n + (sigma_e**2,) * n),
                'Sigma_eps': (sigma_eps**2) * identity_matrix(t * n),
                'D_e1': D_e1,
                'D_v': D_v,
                't': t
            } for i in range(experiments)]
            for id, result in p.imap_unordered(experiment_slow, args):
                print(ln(result))
                slow_results[id] = result
        
        print()
        print('Fast!')
        fast_results = {}
        args = [{
            'id': i,
            'n': n,
            'rtSigma': diagonal_matrix((sigma_s,) * n + (sigma_e,) * n),
            'sigma_s': sigma_s,
            'sigma_e': sigma_e,
            'sigma_eps': sigma_eps,
            'D_e1': D_e1,
            'D_v': D_v,
            't': t
        } for i in range(experiments)]
        for id, result in p.imap_unordered(experiment_fast, args):
            print(ln(result))
            fast_results[id] = result
    
    print()
    print(f'Original: {original_det}')
    print(f'Theoretical: {theoretical_det}')
    if slow: print(f'Slow: {ln(sum(slow_results.values())/experiments)}')
    print(f'Fast: {ln(sum(fast_results.values())/experiments)}')
    if slow: print(f'Difference: {ln(sum(abs(slow_results[id] - fast_results[id]) for id in slow_results)/experiments)}')
