import sys
from functools import wraps
from multiprocessing import Pool

from numpy.random import seed as np_seed

from typing import Callable, TypeVar

load("../framework/LWE.sage")
load("../framework/LWR.sage")

### HELPERS

R = RealField(53)

def det_full(s_s, s_e, s_eps, s_hs, s_he, n, t):
    outer_coeff = 4*ln(s_s) + 4*ln(s_e) + (4*t - 8)*ln(s_eps)

    inner_coeff_1 = (7/4)*t*(t-1)*(n**4)*(s_hs**4)*(s_he**4) + t*(n**2)*(s_eps**4)*((s_hs**4) / (s_e**4) + (s_he**4)/(s_s**4))
    inner_coeff_2 = (t*(t-1)*(n**2)*(s_hs**2)*(s_he**2) + t*n*(s_eps**2)*((s_hs**2 / s_e**2) + (s_he**2 / s_s**2)) + s_eps**4 / (s_s**2 * s_e**2))

    return (n/2)*(outer_coeff + ln(inner_coeff_1 + inner_coeff_2**2)) - (2*n)*(t*ln(s_eps) + ln(s_s) + ln(s_e))


def negacyclic(col):
    row = [-1 * i for i in col[:0:-1]]
    return matrix.toeplitz(col, row, ring=R)


def sample_polynomial_matrix(D, n):
    return negacyclic([draw_from_distribution(D) for _ in range(n)])


def build_H(D_e1, D_v, t, n):
    return block_matrix([[sample_polynomial_matrix(D_e1, n), sample_polynomial_matrix(D_v, n)] for _ in range(t)])

### EXPERIMENTS

def log(*, id, n, t, og, th, det, s_s, s_e, s_eps, s_h_s, s_h_e):
    print(f'{id},{n},{t},{og},{th},{det},{s_s},{s_e},{s_eps},{s_h_s},{s_h_e}')


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
    return 1/Sigma_prime.det()


@experiment
def experiment_fast(n, rtSigma, sigma_s, sigma_e, sigma_eps, D_e1, D_v, t):
    H = build_H(D_e1, D_v, t, n)
    G = identity_matrix(R, 2 * n) + 1/R(sigma_eps)^2 * rtSigma * H.T * H * rtSigma
    return G.det() / sigma_s^(2 * n) / sigma_e^(2 * n)


def run_experiments(experiments, threads, *, n, sigma_s, sigma_e, sigma_eps, sigma_h_e, t):
    sigma_h_s = sqrt(2/3)
    D_e1 = build_Gaussian_law(sigma_h_e, 20)
    D_v = {-1: 1/3, 0: 1/3, 1: 1/3} # build_Gaussian_law(sigma_h_s, 20)
    
    original_det = -2 * n * ln(sigma_s * sigma_e)
    theoretical_det = det_full(sigma_s, sigma_e, sigma_eps, sigma_h_s, sigma_h_e, n, t)
    
    global R # sage struggles with large determinants unless the precision is high
    R = RealField(max(53, theoretical_det * 2))
    rtSigma = diagonal_matrix(R, (sigma_s,) * n + (sigma_e,) * n)
    
    # run many experiments with threading
    with Pool(threads) as p:
        results = {}
        args = [{
            'id': i,
            'n': n,
            'rtSigma': rtSigma,
            'sigma_s': sigma_s,
            'sigma_e': sigma_e,
            'sigma_eps': sigma_eps,
            'D_e1': D_e1,
            'D_v': D_v,
            't': t
        } for i in range(experiments)]
        for id, result in p.imap_unordered(experiment_fast, args):
            results[id] = result
            log(
                id=id,
                n=n,
                t=t,
                og=original_det,
                th=theoretical_det,
                det=ln(result),
                s_s=sigma_s,
                s_e=sigma_e,
                s_eps=sigma_eps,
                s_h_s=sigma_h_s,
                s_h_e=sigma_h_e
            )
    
    avg = ln(sum(results.values())/experiments)
    log(
        id=f'AVG-{experiments}',
        n=n,
        t=t,
        og=original_det,
        th=theoretical_det,
        det=avg,
        s_s=sigma_s,
        s_e=sigma_e,
        s_eps=sigma_eps,
        s_h_s=sigma_h_s,
        s_h_e=sigma_h_e
    )



if __name__ == '__main__':
    experiments = int(sys.argv[1])
    threads = int(sys.argv[2])
    if len(sys.argv) == 4:
        nt = int(sys.argv[3])
        n = 2**(nt // 10)
        t = 2**(nt % 10)
    else:
        n = int(sys.argv[3])
        t = int(sys.argv[4])
    
    run_experiments(experiments, threads, n=n, sigma_e=3.2, sigma_eps=3.2, sigma_s=3.2, sigma_h_e=3.2, t=t)
