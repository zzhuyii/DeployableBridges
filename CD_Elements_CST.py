import numpy as np

class CD_Elements_CST:
    def __init__(self):
        self.node_ijk_mat = None
        self.t_vec = None
        self.E_vec = None
        self.v_vec = None
        self.energy_current_vec = None
        self.delta = 1e-8

    def Initialize(self, node):
        numCST = self.node_ijk_mat.shape[0]
        for i in range(numCST):
            
            nodeIndex = self.node_ijk_mat[i]-1
            
            n1 = nodeIndex[0]
            n2 = nodeIndex[1]
            n3 = nodeIndex[2]
            
            x1 = node.coordinates_mat[n1]
            x2 = node.coordinates_mat[n2]
            x3 = node.coordinates_mat[n3]
            
            vtemp1 = (x2 - x1) / np.linalg.norm(x2 - x1)
            vtemp2 = (x3 - x1) / np.linalg.norm(x3 - x1)
 
            alpha1 = np.arccos(np.dot(vtemp1, vtemp2.T))
            
            vtemp1 = (x1 - x2) / np.linalg.norm(x1 - x2)
            vtemp2 = (x3 - x2) / np.linalg.norm(x3 - x2)
            alpha2 = np.arccos(np.dot(vtemp1, vtemp2.T))
            
            vtemp1 = (x1 - x3) / np.linalg.norm(x1 - x3)
            vtemp2 = (x2 - x3) / np.linalg.norm(x2 - x3)
            alpha3 = np.arccos(np.dot(vtemp1, vtemp2.T))
            
            alphaVec = np.array([alpha1, alpha2, alpha3])
            alphaVec = np.squeeze(alphaVec)

            index = np.argsort(alphaVec)            
            index = np.squeeze(index)
            
            self.node_ijk_mat[i, 0] = nodeIndex[index[2]]+1
            self.node_ijk_mat[i, 1] = nodeIndex[index[1]]+1
            self.node_ijk_mat[i, 2] = nodeIndex[index[0]]+1
            
            
    def Potential(self, strainMat, E, v, A, t):
        strain = np.array([strainMat[0, 0], strainMat[1, 1], strainMat[1, 0] + strainMat[0, 1]])
        
        E=E.item()
        v=v.item()
        A=A.item()
        t=t.item()
        
        C = E / (1 - v ** 2) * np.array([
            [1, v, 0],
            [v, 1, 0],
            [0, 0, (1 - v) / 2]
        ])
        PE = 0.5 * (strain @ C @ strain) * A * t
        return PE

    def Solve_Strain(self, x, Xoriginal):
        
        x=np.squeeze(x)
        Xoriginal=np.squeeze(Xoriginal)
        
        N1, N2, N3 = Xoriginal
        
        L1 = np.linalg.norm(N2 - N3)
        L2 = np.linalg.norm(N1 - N3)
        L3 = np.linalg.norm(N1 - N2)
        
        n1, n2, n3 = x

        l1 = np.linalg.norm(n2 - n3)
        l2 = np.linalg.norm(n1 - n3)
        l3 = np.linalg.norm(n1 - n2)
        
        epsilon1 = (l1 - L1) / L1
        epsilon2 = (l2 - L2) / L2
        epsilon3 = (l3 - L3) / L3
        
        v1 = (n2 - n1) / np.linalg.norm(n2 - n1)
        v2 = (n3 - n1) / np.linalg.norm(n3 - n1)
        cosa1 = np.dot(v1, v2.T)
        sina1 = np.sqrt(1 - cosa1 ** 2)
        
        v1 = (n1 - n2) / np.linalg.norm(n1 - n2)
        v2 = (n3 - n2) / np.linalg.norm(n3 - n2)
        cosa2 = np.dot(v1, v2.T)
        sina2 = np.sqrt(1 - cosa2 ** 2)
        
        v1 = (n1 - n3) / np.linalg.norm(n1 - n3)
        v2 = (n2 - n3) / np.linalg.norm(n2 - n3)
        cosa3 = np.dot(v1, v2.T)
        sina3 = np.sqrt(1 - cosa3 ** 2)
        
        Mat = np.array([
            [sina3 ** 2, -2 * cosa3 * sina3],
            [sina2 ** 2, 2 * cosa2 * sina2]
        ])
        
        Rhs = np.array([
            epsilon2 - cosa3 ** 2 * epsilon1,
            epsilon3 - cosa2 ** 2 * epsilon1
        ])
        
        Mat=np.squeeze(Mat)
        Rhs=np.squeeze(Rhs)
        
        vec = np.linalg.solve(Mat, Rhs)

        strainMat = np.zeros((2, 2))
        strainMat[0, 0] = epsilon1
        strainMat[1, 1] = vec[0]
        strainMat[0, 1] = vec[1]
        strainMat[1, 0] = vec[1]
        
        return strainMat

    def Solve_Local_Force(self, x, X, E, v, t):
        
        Flocal = np.zeros(9)
        delta = self.delta
        
        for i in range(9):
            
            index1 = i % 3 
            index2 = i // 3 
            
            tempXfor = x.copy()  
            tempXfor = np.squeeze(tempXfor)
            tempXfor[index2, index1] += delta
            
            tempXback = x.copy()   
            tempXback = np.squeeze(tempXback)
            tempXback[index2, index1] -= delta
            
            strainFor = self.Solve_Strain(tempXfor, X)
            strainBack = self.Solve_Strain(tempXback, X)
            N1, N2, N3 = X
            v1 = N2 - N1
            v2 = N3 - N1
            area = np.linalg.norm(np.cross(v1, v2)) / 2
            Flocal[i] = 0.5 / delta * (
                self.Potential(strainFor, E, v, area, t) - self.Potential(strainBack, E, v, area, t)
            )
        return Flocal

    def Solve_Local_Stiff(self, x, X, E, v, t):
        Klocal = np.zeros((9, 9))
        delta = self.delta
        for i in range(9):
            index1 = i % 3
            index2 = i // 3
            
            x = np.squeeze(x)
            
            tempXfor = x.copy()
            tempXfor[index2, index1] += delta
            
            tempXback = x.copy()
            tempXback[index2, index1] -= delta
            
            Klocal[i, :] = 0.5 / delta * (
                self.Solve_Local_Force(tempXfor, X, E, v, t) - self.Solve_Local_Force(tempXback, X, E, v, t)
            )
        return Klocal

    def Solve_Global_Force(self, node, U):
        nodalCoordinates = node.coordinates_mat
        cstIJK = self.node_ijk_mat.astype(int)
        E_vec = self.E_vec
        t_vec = self.t_vec
        v_vec = self.v_vec
        nodeNum = nodalCoordinates.shape[0]
        Tcst = np.zeros(3 * nodeNum)
        cstNum = E_vec.shape[0]
        
        for i in range(cstNum):
            
            node1 = cstIJK[i,0]-1
            node2 = cstIJK[i,1]-1
            node3 = cstIJK[i,2]-1
            
            X1 = nodalCoordinates[node1]
            X2 = nodalCoordinates[node2]
            X3 = nodalCoordinates[node3]
            
            X = np.array([X1, X2, X3])
            
            x1 = nodalCoordinates[node1] + U[node1]
            x2 = nodalCoordinates[node2] + U[node2]
            x3 = nodalCoordinates[node3] + U[node3]
            
            x = np.array([x1, x2, x3])
            
            Flocal = self.Solve_Local_Force(x, X, E_vec[i], v_vec[i], t_vec[i])

            for local_idx, global_idx in enumerate([node1, node2, node3]):
                Tcst[3 * global_idx:3 * global_idx + 3] += Flocal[3 * local_idx:3 * local_idx + 3]
        
        return Tcst

    def Solve_Global_Stiff(self, node, U):
        nodalCoordinates = node.coordinates_mat
        cstIJK = self.node_ijk_mat.astype(int)
        E_vec = self.E_vec
        t_vec = self.t_vec
        v_vec = self.v_vec
        nodeNum = nodalCoordinates.shape[0]
        Kcst = np.zeros((3 * nodeNum, 3 * nodeNum))
        cstNum = E_vec.shape[0]
        
        for i in range(cstNum):
            
            node1 = cstIJK[i,0]-1
            node2 = cstIJK[i,1]-1
            node3 = cstIJK[i,2]-1
            
            X1 = nodalCoordinates[node1]
            X2 = nodalCoordinates[node2]
            X3 = nodalCoordinates[node3]
            
            X = np.array([X1, X2, X3])
            
            x1 = nodalCoordinates[node1] + U[node1]
            x2 = nodalCoordinates[node2] + U[node2]
            x3 = nodalCoordinates[node3] + U[node3]
            
            x = np.array([x1, x2, x3])
            
            Klocal = self.Solve_Local_Stiff(x, X, E_vec[i], v_vec[i], t_vec[i])
            nodeIndex = [node1, node2, node3]
            for j in range(3):
                for k in range(3):
                    Kcst[
                        3 * nodeIndex[j]:3 * nodeIndex[j] + 3,
                        3 * nodeIndex[k]:3 * nodeIndex[k] + 3
                    ] += Klocal[3 * j:3 * j + 3, 3 * k:3 * k + 3]
        return Kcst

    def Solve_FK(self, node, U):
        Tcst = self.Solve_Global_Force(node, U)
        Kcst = self.Solve_Global_Stiff(node, U)
        return Tcst, Kcst