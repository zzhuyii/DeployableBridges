import numpy as np

class CD_Elements_RotSprings_4N:
    def __init__(self):
        # node_ijkl_mat: (Ns, 4) array of node indices
        # rot_spr_K_vec: (Ns,) array of rotational stiffness values
        self.node_ijkl_mat = None
        self.rot_spr_K_vec = None
        self.theta_stress_free_vec = None
        self.theta_current_vec = None
        self.energy_current_vec = None
        self.delta = 1e-8

    def Initialize(self, node):
        rotSprIJKL = self.node_ijkl_mat
        numSpr = rotSprIJKL.shape[0]
        
        self.theta_current_vec = np.zeros(numSpr)
        self.theta_stress_free_vec = np.zeros(numSpr)
        self.energy_current_vec = np.zeros(numSpr)

        for i in range(numSpr):
            
            n1 = rotSprIJKL[i,0]-1
            n2 = rotSprIJKL[i,1]-1
            n3 = rotSprIJKL[i,2]-1
            n4 = rotSprIJKL[i,3]-1
            
            x1 = node.coordinates_mat[n1]
            x2 = node.coordinates_mat[n2]
            x3 = node.coordinates_mat[n3]
            x4 = node.coordinates_mat[n4]

            v1 = x1 - x2
            v2 = x3 - x2

            norm1 = np.cross(v1, v2)
            
            # Direction check, reordering as in MATLAB
            if norm1[2] != 0:
                if norm1[2] < 0:
                    self.node_ijkl_mat[i, 1], self.node_ijkl_mat[i, 2] = n3+1, n2+1
                    
            elif norm1[1] != 0:
                if norm1[1] < 0:
                    self.node_ijkl_mat[i, 1], self.node_ijkl_mat[i, 2] = n3+1, n2+1
                    
            else:
                if norm1[0] < 0:
                    self.node_ijkl_mat[i, 1], self.node_ijkl_mat[i, 2] = n3+1, n2+1

            X = np.vstack([x1, x2, x3, x4])
            self.theta_current_vec[i] = self.Solve_Theta(X)
            self.theta_stress_free_vec[i] = self.theta_current_vec[i]

    @staticmethod
    def Potential(theta, theta0, K):
        # Linear elastic potential energy
        return 0.5 * K * (theta - theta0) ** 2

    def Solve_Theta(self, X):
        
        Xi, Xj, Xk, Xl = X
        
        rij = Xi - Xj
        rkj = Xk - Xj
        rkl = Xk - Xl

        m = np.cross(rij, rkj)
        n = np.cross(rkj, rkl)

        dot_m_rkl = np.dot(m, rkl.T)
        yita = 1 if dot_m_rkl == 0 else np.sign(dot_m_rkl)
        
        cos_theta = np.dot(m, n.T) / (np.linalg.norm(m) * np.linalg.norm(n))
        # Prevent domain errors due to floating point
        
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.mod(yita * np.real(np.arccos(cos_theta)), 2 * np.pi)
        return theta

    def Solve_Local_Force(self, X, theta0, K):
        # 4 nodes × 3 coords = 12 dofs
        Flocal = np.zeros(12)
        delta = self.delta
        
        for i in range(12):
            index1 = i % 3
            index2 = i // 3
            tempXfor = np.array(X, copy=True)
            tempXback = np.array(X, copy=True)
            tempXfor[index2, index1] += delta
            tempXback[index2, index1] -= delta
            
            thetaFor = self.Solve_Theta(tempXfor)
            thetaBack = self.Solve_Theta(tempXback)
            
            Flocal[i] = 0.5 / delta * (self.Potential(thetaFor, theta0, K) -
                                      self.Potential(thetaBack, theta0, K))
        return Flocal

    def Solve_Local_Stiff(self, X, theta0, K):
        # 12x12 Hessian via central difference
        Klocal = np.zeros((12, 12))
        delta = self.delta
        for i in range(12):
            index1 = i % 3
            index2 = i // 3
            tempXfor = np.array(X, copy=True)
            tempXback = np.array(X, copy=True)
            tempXfor[index2, index1] += delta
            tempXback[index2, index1] -= delta
            F_for = self.Solve_Local_Force(tempXfor, theta0, K)
            F_back = self.Solve_Local_Force(tempXback, theta0, K)
            Klocal[i, :] = 0.5 / delta * (F_for - F_back)
        return Klocal

    def Solve_Global_Force(self, node, U):
        nodalCoordinates = node.coordinates_mat
        rotSprIJKL = self.node_ijkl_mat
        theta0 = self.theta_stress_free_vec
        rotSprK = self.rot_spr_K_vec
        nodeNum = nodalCoordinates.shape[0]
        Trs = np.zeros((3 * nodeNum,))
        
        for i in range(len(rotSprK)):
            
            node1 = rotSprIJKL[i,0]-1
            node2 = rotSprIJKL[i,1]-1
            node3 = rotSprIJKL[i,2]-1
            node4 = rotSprIJKL[i,3]-1
            
            X1 = nodalCoordinates[node1] + U[node1]
            X2 = nodalCoordinates[node2] + U[node2]
            X3 = nodalCoordinates[node3] + U[node3]
            X4 = nodalCoordinates[node4] + U[node4]
            
            X = np.vstack([X1, X2, X3, X4])
            
            Flocal = self.Solve_Local_Force(X, theta0[i], rotSprK[i])
            idx = [node1, node2, node3, node4]
            for j, base in enumerate(idx):
                Trs[3*base:3*base+3] += Flocal[j*3:j*3+3]
        return Trs

    def Solve_Global_Stiff(self, node, U):
        nodalCoordinates = node.coordinates_mat
        rotSprIJKL = self.node_ijkl_mat
        theta0 = self.theta_stress_free_vec
        rotSprK = self.rot_spr_K_vec
        nodeNum = nodalCoordinates.shape[0]
        Krs = np.zeros((3 * nodeNum, 3 * nodeNum))
        
        for i in range(len(rotSprK)):
            
            node1 = rotSprIJKL[i,0]-1
            node2 = rotSprIJKL[i,1]-1
            node3 = rotSprIJKL[i,2]-1
            node4 = rotSprIJKL[i,3]-1
            
            X1 = nodalCoordinates[node1] + U[node1]
            X2 = nodalCoordinates[node2] + U[node2]
            X3 = nodalCoordinates[node3] + U[node3]
            X4 = nodalCoordinates[node4] + U[node4]
            X = np.vstack([X1, X2, X3, X4])
            
            Klocal = self.Solve_Local_Stiff(X, theta0[i], rotSprK[i])
            nodeIndex = [node1, node2, node3, node4]
            for j in range(4):
                for k in range(4):
                    Krs[3*nodeIndex[j]:3*nodeIndex[j]+3,
                        3*nodeIndex[k]:3*nodeIndex[k]+3] += \
                        Klocal[3*j:3*j+3, 3*k:3*k+3]
        return Krs

    def Solve_FK(self, node, U):
        
        
        Trs = self.Solve_Global_Force(node, U)
        Krs = self.Solve_Global_Stiff(node, U)
        
        rotSprIJKL = self.node_ijkl_mat
        nodalCoordinates = node.coordinates_mat
        
        for i in range(len(self.rot_spr_K_vec)):
            
            node1 = rotSprIJKL[i,0]-1
            node2 = rotSprIJKL[i,1]-1
            node3 = rotSprIJKL[i,2]-1
            node4 = rotSprIJKL[i,3]-1
            
            X1 = nodalCoordinates[node1] + U[node1]
            X2 = nodalCoordinates[node2] + U[node2]
            X3 = nodalCoordinates[node3] + U[node3]
            X4 = nodalCoordinates[node4] + U[node4]
            
            X = np.vstack([X1, X2, X3, X4])
            
            self.theta_current_vec[i] = self.Solve_Theta(X)
            self.energy_current_vec[i] = self.Potential(self.theta_current_vec[i],
                                                       self.theta_stress_free_vec[i],
                                                       self.rot_spr_K_vec[i])
        return Trs, Krs