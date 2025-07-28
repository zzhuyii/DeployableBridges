import numpy as np

class CD_Elements_RotSprings_3N:
    def __init__(self):
        """
        node_ijk_mat: Ns x 3 numpy array
        rot_spr_K_vec: Ns x 1 numpy array
        """
        self.node_ijk_mat = None
        self.rot_spr_K_vec = None
        self.theta_stress_free_vec = None
        self.theta_current_vec = None
        self.delta = 1e-8  # step for central difference

    def Initialize(self, node):
        rotSprIJK = self.node_ijk_mat
        numSpr = rotSprIJK.shape[0]
        self.theta_current_vec = np.zeros(numSpr)
        self.theta_stress_free_vec = np.zeros(numSpr)

        for i in range(numSpr):
            
            n1 = rotSprIJK[i,0]-1
            n2 = rotSprIJK[i,1]-1
            n3 = rotSprIJK[i,2]-1
            
            x1 = node.coordinates_mat[n1]
            x2 = node.coordinates_mat[n2]
            x3 = node.coordinates_mat[n3]
            X = np.vstack([x1, x2, x3])
            self.theta_current_vec[i] = self.Solve_Theta(X)
            self.theta_stress_free_vec[i] = self.theta_current_vec[i]

    @staticmethod
    def Potential(theta, theta0, K):
        return 0.5 * K * (theta - theta0) ** 2

    @staticmethod
    def Solve_Theta(X):
        Xi, Xj, Xk = X
        v1 = Xi - Xj
        v2 = Xk - Xj
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)
        theta = np.arccos(np.clip(np.dot(v1, v2.T), -1.0, 1.0))
        return np.real(theta)  # in case of minor floating error

    def Solve_Local_Force(self, X, theta0, K):
        Flocal = np.zeros(9)
        delta = self.delta

        for i in range(9):
            index1 = i % 3
            index2 = i // 3
            tempXfor = X.copy()
            tempXback = X.copy()
            tempXfor[index2, index1] += delta
            tempXback[index2, index1] -= delta

            thetaFor = self.Solve_Theta(tempXfor)
            thetaBack = self.Solve_Theta(tempXback)
            Flocal[i] = 0.5 / delta * (self.Potential(thetaFor, theta0, K) - self.Potential(thetaBack, theta0, K))
        return Flocal

    def Solve_Global_Force(self, node, U):
        nodalCoordinates = node.coordinates_mat
        rotSprIJK = self.node_ijk_mat
        theta0 = self.theta_stress_free_vec
        rotSprK = self.rot_spr_K_vec
        rotSprNum = rotSprK.shape[0]
        nodeNum = nodalCoordinates.shape[0]
        Trs = np.zeros(3 * nodeNum)

        for i in range(rotSprNum):
            
            node1 = rotSprIJK[i,0]-1
            node2 = rotSprIJK[i,1]-1
            node3 = rotSprIJK[i,2]-1
            
            X1 = nodalCoordinates[node1] + U[node1]
            X2 = nodalCoordinates[node2] + U[node2]
            X3 = nodalCoordinates[node3] + U[node3]
            
            X = np.vstack([X1, X2, X3])
            
            Flocal = self.Solve_Local_Force(X, theta0[i], rotSprK[i])
            for j, n in enumerate([node1, node2, node3]):
                Trs[3*n:3*n+3] += Flocal[3*j:3*j+3]
        return Trs

    def Solve_Local_Stiff(self, X, theta0, K):
        Klocal = np.zeros((9, 9))
        delta = self.delta

        for i in range(9):
            index1 = i % 3
            index2 = i // 3
            tempXfor = X.copy()
            tempXback = X.copy()
            tempXfor[index2, index1] += delta
            tempXback[index2, index1] -= delta
            Klocal[i, :] = 0.5 / delta * (
                self.Solve_Local_Force(tempXfor, theta0, K) - self.Solve_Local_Force(tempXback, theta0, K)
            )
        return Klocal

    def Solve_Global_Stiff(self, node, U):
        nodalCoordinates = node.coordinates_mat
        rotSprIJK = self.node_ijk_mat
        theta0 = self.theta_stress_free_vec
        rotSprK = self.rot_spr_K_vec
        rotSprNum = rotSprK.shape[0]
        nodeNum = nodalCoordinates.shape[0]
        Krs = np.zeros((3 * nodeNum, 3 * nodeNum))

        for i in range(rotSprNum):
            
            node1 = rotSprIJK[i,0]-1
            node2 = rotSprIJK[i,1]-1
            node3 = rotSprIJK[i,2]-1
            
            X1 = nodalCoordinates[node1] + U[node1]
            X2 = nodalCoordinates[node2] + U[node2]
            X3 = nodalCoordinates[node3] + U[node3]
            X = np.vstack([X1, X2, X3])
            Klocal = self.Solve_Local_Stiff(X, theta0[i], rotSprK[i])
            nodeIndex = [node1, node2, node3]
            for j in range(3):
                for k in range(3):
                    Krs[3*nodeIndex[j]:3*nodeIndex[j]+3, 3*nodeIndex[k]:3*nodeIndex[k]+3] += Klocal[3*j:3*j+3, 3*k:3*k+3]
        return Krs

    def Solve_FK(self, node, U):
        Trs = self.Solve_Global_Force(node, U)
        Krs = self.Solve_Global_Stiff(node, U)
        return Trs, Krs