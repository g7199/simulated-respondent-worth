"""Best non-negative weights for a set of sources (Proposition 1 with w >= 0).

min_{w>=0} 1/2 w'Gw - V 1'w is a convex quadratic programme. With the Cholesky factor G = R'R and R'b = V 1 it is the
non-negative least-squares problem min_{w>=0} ||Rw - b||^2, which scipy's nnls solves exactly. At the solution
w'Gw = V 1'w, so the set is worth 1'w respondents (n + 1'w together with n respondents)."""
import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import nnls


def best_weights(G, V):
    G = np.asarray(G, float); G = 0.5 * (G + G.T); V = float(V); m = len(G)
    R = cholesky(G, lower=False)                       # raises LinAlgError if G is not positive definite
    b = solve_triangular(R.T, V * np.ones(m), lower=True)
    w, _ = nnls(R, b, maxiter=50 * m)
    g = G @ w - V                                      # KKT: w >= 0, g >= 0 and w * g = 0 at the global optimum
    assert g.min() >= -1e-8 * abs(V) and np.max(np.abs(w * g)) <= 1e-8 * abs(V), "non-negative weights fail the KKT conditions"
    return w
