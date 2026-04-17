import numpy as np

class Elements_Nodes:
    def __init__(self):
        self.coordinates_mat = None  # (N x 3)
        self.mass_vec = None         # (N,)
        self.current_U_mat = None    # (N x 3)
        self.current_ext_force_mat = None  # (N x 3)

    def Find_Mass_Mat(self):
        # Returns the mass matrix as a diagonal block matrix (3N x 3N)
        N = len(self.mass_vec)
        M = np.zeros((3 * N, 3 * N))
        for i in range(N):
            m = self.mass_vec[i]
            idx = slice(3 * i, 3 * i + 3)
            M[idx, idx] = np.eye(3) * m
        return M
