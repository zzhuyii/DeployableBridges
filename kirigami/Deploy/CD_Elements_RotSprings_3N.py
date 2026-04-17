import numpy as np


class CD_Elements_RotSprings_3N:
    def __init__(self):
        self.node_ijk_mat = None
        self.rot_spr_K_vec = None
        self.theta_stress_free_vec = None
        self.theta_current_vec = None
        self.delta = 1e-4

    @staticmethod
    def Potential(theta, theta0, K):
        d = theta - theta0
        return 0.5 * K * d * d

    @staticmethod
    def Solve_Theta(X):
        Xi = X[0, :]
        Xj = X[1, :]
        Xk = X[2, :]

        v1 = Xi - Xj
        v2 = Xk - Xj
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-14 or n2 < 1e-14:
            return 0.0

        c = float(np.dot(v1 / n1, v2 / n2))
        c = max(-1.0, min(1.0, c))
        return float(np.arccos(c))

    def Solve_Local_Force(self, X, theta0, K):
        Flocal = np.zeros(9, dtype=float)
        for i in range(9):
            node_idx = i // 3
            dim = i % 3
            X_for = X.copy()
            X_back = X.copy()
            X_for[node_idx, dim] += self.delta
            X_back[node_idx, dim] -= self.delta
            Flocal[i] = (
                self.Potential(self.Solve_Theta(X_for), theta0, K)
                - self.Potential(self.Solve_Theta(X_back), theta0, K)
            ) / (2.0 * self.delta)
        return Flocal

    def Solve_Local_Stiff(self, X, theta0, K):
        Klocal = np.zeros((9, 9), dtype=float)
        for i in range(9):
            node_idx = i // 3
            dim = i % 3
            X_for = X.copy()
            X_back = X.copy()
            X_for[node_idx, dim] += self.delta
            X_back[node_idx, dim] -= self.delta
            Klocal[i, :] = (
                self.Solve_Local_Force(X_for, theta0, K)
                - self.Solve_Local_Force(X_back, theta0, K)
            ) / (2.0 * self.delta)
        return Klocal

    def Initialize(self, node):
        if self.node_ijk_mat is None or np.asarray(self.node_ijk_mat).size == 0:
            self.theta_current_vec = np.zeros((0,), dtype=float)
            self.theta_stress_free_vec = np.zeros((0,), dtype=float)
            if self.rot_spr_K_vec is None:
                self.rot_spr_K_vec = np.zeros((0,), dtype=float)
            return

        rot_spr_ijk = np.asarray(self.node_ijk_mat, dtype=int)
        if self.rot_spr_K_vec is None:
            raise ValueError("rot_spr_K_vec is not set.")

        self.theta_current_vec = np.zeros(rot_spr_ijk.shape[0], dtype=float)
        self.theta_stress_free_vec = np.zeros(rot_spr_ijk.shape[0], dtype=float)
        for i, (n1, n2, n3) in enumerate(rot_spr_ijk):
            X = np.vstack([
                node.coordinates_mat[n1 - 1, :],
                node.coordinates_mat[n2 - 1, :],
                node.coordinates_mat[n3 - 1, :],
            ])
            theta = self.Solve_Theta(X)
            self.theta_current_vec[i] = theta
            self.theta_stress_free_vec[i] = theta

    def Solve_Global_Force(self, node, U):
        rot_spr_ijk = np.asarray(self.node_ijk_mat, dtype=int)
        theta0 = np.asarray(self.theta_stress_free_vec, dtype=float).reshape(-1)
        rot_spr_k = np.asarray(self.rot_spr_K_vec, dtype=float).reshape(-1)

        Trs = np.zeros(3 * node.coordinates_mat.shape[0], dtype=float)
        for i, (n1, n2, n3) in enumerate(rot_spr_ijk):
            X = np.vstack([
                node.coordinates_mat[n1 - 1, :] + U[n1 - 1, :],
                node.coordinates_mat[n2 - 1, :] + U[n2 - 1, :],
                node.coordinates_mat[n3 - 1, :] + U[n3 - 1, :],
            ])
            Flocal = self.Solve_Local_Force(X, theta0[i], rot_spr_k[i])
            for a, n in enumerate((n1, n2, n3)):
                Trs[3 * (n - 1):3 * (n - 1) + 3] += Flocal[3 * a:3 * a + 3]
        return Trs

    def Solve_Global_Stiff(self, node, U):
        rot_spr_ijk = np.asarray(self.node_ijk_mat, dtype=int)
        theta0 = np.asarray(self.theta_stress_free_vec, dtype=float).reshape(-1)
        rot_spr_k = np.asarray(self.rot_spr_K_vec, dtype=float).reshape(-1)

        Krs = np.zeros((3 * node.coordinates_mat.shape[0], 3 * node.coordinates_mat.shape[0]), dtype=float)
        for i, (n1, n2, n3) in enumerate(rot_spr_ijk):
            X = np.vstack([
                node.coordinates_mat[n1 - 1, :] + U[n1 - 1, :],
                node.coordinates_mat[n2 - 1, :] + U[n2 - 1, :],
                node.coordinates_mat[n3 - 1, :] + U[n3 - 1, :],
            ])
            Klocal = self.Solve_Local_Stiff(X, theta0[i], rot_spr_k[i])
            for a, na in enumerate((n1, n2, n3)):
                for b, nb in enumerate((n1, n2, n3)):
                    Krs[3 * (na - 1):3 * (na - 1) + 3,
                        3 * (nb - 1):3 * (nb - 1) + 3] += Klocal[3 * a:3 * a + 3,
                                                                  3 * b:3 * b + 3]
        return Krs

    def Solve_FK(self, node, U):
        return self.Solve_Global_Force(node, U), self.Solve_Global_Stiff(node, U)
