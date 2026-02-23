import numpy as np


class CD_Elements_RotSprings_3N:
    """
    3-Node rotational spring (central difference) - MATLAB port

    Expected attributes to be assigned by main script:
      - node_ijk_mat: (Ns,3) int, 1-based node indices
      - rot_spr_K_vec: (Ns,) float
    """

    def __init__(self):
        # (Ns,3) 1-based
        self.node_ijk_mat = None

        # (Ns,)
        self.rot_spr_K_vec = None

        # (Ns,)
        self.theta_stress_free_vec = None
        self.theta_current_vec = None

        # central difference step
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

        v1 = v1 / n1
        v2 = v2 / n2

        c = float(np.dot(v1, v2))
        c = max(-1.0, min(1.0, c))
        return float(np.arccos(c))

    def Solve_Local_Force(self, X, theta0, K):
        delta = self.delta
        Flocal = np.zeros(9, dtype=float)

        for i in range(9):
            node_idx = i // 3
            dim = i % 3

            X_for = X.copy()
            X_back = X.copy()
            X_for[node_idx, dim] += delta
            X_back[node_idx, dim] -= delta

            theta_for = self.Solve_Theta(X_for)
            theta_back = self.Solve_Theta(X_back)

            Flocal[i] = (self.Potential(theta_for, theta0, K) -
                         self.Potential(theta_back, theta0, K)) / (2.0 * delta)

        return Flocal

    def Solve_Local_Stiff(self, X, theta0, K):
        delta = self.delta
        Klocal = np.zeros((9, 9), dtype=float)

        for i in range(9):
            node_idx = i // 3
            dim = i % 3

            X_for = X.copy()
            X_back = X.copy()
            X_for[node_idx, dim] += delta
            X_back[node_idx, dim] -= delta

            F_for = self.Solve_Local_Force(X_for, theta0, K)
            F_back = self.Solve_Local_Force(X_back, theta0, K)

            Klocal[i, :] = (F_for - F_back) / (2.0 * delta)

        return Klocal

    def Initialize(self, node):
        if self.node_ijk_mat is None:
            # allow empty initialization before springs are assigned
            self.theta_current_vec = np.zeros((0,), dtype=float)
            self.theta_stress_free_vec = np.zeros((0,), dtype=float)
            if self.rot_spr_K_vec is None:
                self.rot_spr_K_vec = np.zeros((0,), dtype=float)
            return

        rotSprIJK = np.asarray(self.node_ijk_mat, dtype=int)
        if rotSprIJK.size == 0:
            self.theta_current_vec = np.zeros((0,), dtype=float)
            self.theta_stress_free_vec = np.zeros((0,), dtype=float)
            if self.rot_spr_K_vec is None:
                self.rot_spr_K_vec = np.zeros((0,), dtype=float)
            return

        if self.rot_spr_K_vec is None:
            raise ValueError("rotSpr3N.rot_spr_K_vec is not set.")
        numSpr = rotSprIJK.shape[0]

        self.theta_current_vec = np.zeros(numSpr, dtype=float)
        self.theta_stress_free_vec = np.zeros(numSpr, dtype=float)

        for i in range(numSpr):
            n1, n2, n3 = rotSprIJK[i, :]
            x1 = node.coordinates_mat[n1 - 1, :]
            x2 = node.coordinates_mat[n2 - 1, :]
            x3 = node.coordinates_mat[n3 - 1, :]
            X = np.vstack([x1, x2, x3])
            th = self.Solve_Theta(X)
            self.theta_current_vec[i] = th
            self.theta_stress_free_vec[i] = th

    def Solve_Global_Force(self, node, U):
        if self.node_ijk_mat is None:
            raise ValueError("rotSpr3N.node_ijk_mat is not set.")
        if self.theta_stress_free_vec is None:
            raise ValueError("rotSpr3N.theta_stress_free_vec is not initialized (call Initialize).")

        nodalCoordinates = node.coordinates_mat
        rotSprIJK = np.asarray(self.node_ijk_mat, dtype=int)
        theta0 = np.asarray(self.theta_stress_free_vec, dtype=float).reshape(-1)
        rotSprK = np.asarray(self.rot_spr_K_vec, dtype=float).reshape(-1)

        nodeNum = nodalCoordinates.shape[0]
        Trs = np.zeros(3 * nodeNum, dtype=float)

        for i in range(rotSprIJK.shape[0]):
            n1, n2, n3 = rotSprIJK[i, :]

            X1 = nodalCoordinates[n1 - 1, :] + U[n1 - 1, :]
            X2 = nodalCoordinates[n2 - 1, :] + U[n2 - 1, :]
            X3 = nodalCoordinates[n3 - 1, :] + U[n3 - 1, :]
            X = np.vstack([X1, X2, X3])

            Flocal = self.Solve_Local_Force(X, theta0[i], rotSprK[i])

            Trs[3 * (n1 - 1): 3 * (n1 - 1) + 3] += Flocal[0:3]
            Trs[3 * (n2 - 1): 3 * (n2 - 1) + 3] += Flocal[3:6]
            Trs[3 * (n3 - 1): 3 * (n3 - 1) + 3] += Flocal[6:9]

        return Trs

    def Solve_Global_Stiff(self, node, U):
        if self.node_ijk_mat is None:
            raise ValueError("rotSpr3N.node_ijk_mat is not set.")
        if self.theta_stress_free_vec is None:
            raise ValueError("rotSpr3N.theta_stress_free_vec is not initialized (call Initialize).")

        nodalCoordinates = node.coordinates_mat
        rotSprIJK = np.asarray(self.node_ijk_mat, dtype=int)
        theta0 = np.asarray(self.theta_stress_free_vec, dtype=float).reshape(-1)
        rotSprK = np.asarray(self.rot_spr_K_vec, dtype=float).reshape(-1)

        nodeNum = nodalCoordinates.shape[0]
        Krs = np.zeros((3 * nodeNum, 3 * nodeNum), dtype=float)

        for i in range(rotSprIJK.shape[0]):
            n1, n2, n3 = rotSprIJK[i, :]

            X1 = nodalCoordinates[n1 - 1, :] + U[n1 - 1, :]
            X2 = nodalCoordinates[n2 - 1, :] + U[n2 - 1, :]
            X3 = nodalCoordinates[n3 - 1, :] + U[n3 - 1, :]
            X = np.vstack([X1, X2, X3])

            Klocal = self.Solve_Local_Stiff(X, theta0[i], rotSprK[i])

            nodeIndex = [n1, n2, n3]
            for a in range(3):
                for b in range(3):
                    Ia = nodeIndex[a] - 1
                    Ib = nodeIndex[b] - 1
                    Krs[3 * Ia: 3 * Ia + 3, 3 * Ib: 3 * Ib + 3] += Klocal[3 * a: 3 * a + 3,
                                                                           3 * b: 3 * b + 3]

        return Krs

    def Solve_Global_Theta(self, node, U):
        if self.node_ijk_mat is None:
            raise ValueError("rotSpr3N.node_ijk_mat is not set.")
        rotSprIJK = np.asarray(self.node_ijk_mat, dtype=int)

        if self.theta_current_vec is None:
            self.theta_current_vec = np.zeros(rotSprIJK.shape[0], dtype=float)

        nodalCoordinates = node.coordinates_mat
        for i in range(rotSprIJK.shape[0]):
            n1, n2, n3 = rotSprIJK[i, :]

            X1 = nodalCoordinates[n1 - 1, :] + U[n1 - 1, :]
            X2 = nodalCoordinates[n2 - 1, :] + U[n2 - 1, :]
            X3 = nodalCoordinates[n3 - 1, :] + U[n3 - 1, :]
            X = np.vstack([X1, X2, X3])

            self.theta_current_vec[i] = self.Solve_Theta(X)

    def Solve_FK(self, node, U):
        Trs = self.Solve_Global_Force(node, U)
        Krs = self.Solve_Global_Stiff(node, U)
        return Trs, Krs
