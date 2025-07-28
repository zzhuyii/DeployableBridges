import numpy as np

class CD_Elements_Bars:
    def __init__(self):
        self.node_ij_mat = None  # (Nb x 2)
        self.A_vec = None        # (Nb,)
        self.E_vec = None        # (Nb,)
        self.L0_vec = None       # (Nb,)
        self.delta = 1e-8

    def Initialize(self, node):
        coords = node.coordinates_mat
        self.L0_vec = np.zeros_like(self.A_vec)

        for i in range(len(self.A_vec)):
            
            node1 = coords[self.node_ij_mat[i, 0]-1]
            node2 = coords[self.node_ij_mat[i, 1]-1]
 
            self.L0_vec[i] = np.linalg.norm(node1 - node2)

    def Potential(self, X1, X2, L0, E, A):
        return 0.5 * E * A / L0 * (np.linalg.norm(X1 - X2) - L0) ** 2

    def Solve_Local_Force(self, X1, X2, L0, E, A):
        delta = self.delta
        Flocal = np.zeros(6)
        for i in range(3):
            dX = np.zeros(3)
            dX[i] = delta
            Flocal[i] = 0.5 / delta * (self.Potential(X1 + dX, X2, L0, E, A) - self.Potential(X1 - dX, X2, L0, E, A))
            Flocal[3 + i] = 0.5 / delta * (self.Potential(X1, X2 + dX, L0, E, A) - self.Potential(X1, X2 - dX, L0, E, A))
        return Flocal

    def Solve_Local_Stiff(self, X1, X2, L0, E, A):
        delta = self.delta
        Klocal = np.zeros((6, 6))
        for i in range(3):
            dX = np.zeros(3)
            dX[i] = delta
            f_plus = self.Solve_Local_Force(X1 + dX, X2, L0, E, A)
            f_minus = self.Solve_Local_Force(X1 - dX, X2, L0, E, A)
            Klocal[i, :] = 0.5 / delta * (f_plus - f_minus)

            f_plus = self.Solve_Local_Force(X1, X2 + dX, L0, E, A)
            f_minus = self.Solve_Local_Force(X1, X2 - dX, L0, E, A)
            Klocal[3 + i, :] = 0.5 / delta * (f_plus - f_minus)
        return Klocal

    def Solve_Global_Force(self, node, U):
        coords = node.coordinates_mat
        Nb = self.node_ij_mat.shape[0]
        Nd = coords.shape[0]
        Tbar = np.zeros(3 * Nd)

        for i in range(Nb):
            node1 = self.node_ij_mat[i,0]-1
            node2 = self.node_ij_mat[i,1]-1
            X1 = coords[node1] + U[node1]
            X2 = coords[node2] + U[node2]
            Flocal = self.Solve_Local_Force(X1, X2, self.L0_vec[i], self.E_vec[i], self.A_vec[i])
            Tbar[3*node1:3*node1+3] += Flocal[:3]
            Tbar[3*node2:3*node2+3] += Flocal[3:]
        return Tbar

    def Solve_Global_Stiff(self, node, U):
        coords = node.coordinates_mat
        Nb = self.node_ij_mat.shape[0]
        Nd = coords.shape[0]
        Kbar = np.zeros((3*Nd, 3*Nd))

        for i in range(Nb):
            node1 = self.node_ij_mat[i,0]-1
            node2 = self.node_ij_mat[i,1]-1
            X1 = coords[node1] + U[node1]
            X2 = coords[node2] + U[node2]
            Klocal = self.Solve_Local_Stiff(X1, X2, self.L0_vec[i], self.E_vec[i], self.A_vec[i])

            idx1 = slice(3*node1, 3*node1+3)
            idx2 = slice(3*node2, 3*node2+3)
            Kbar[idx1, idx1] += Klocal[:3, :3]
            Kbar[idx1, idx2] += Klocal[:3, 3:]
            Kbar[idx2, idx1] += Klocal[3:, :3]
            Kbar[idx2, idx2] += Klocal[3:, 3:]
        return Kbar

    def Solve_FK(self, node, U):
        Tbar = self.Solve_Global_Force(node, U)
        Kbar = self.Solve_Global_Stiff(node, U)
        return Tbar, Kbar

    def Solve_Strain(self, node, U):
        coords = node.coordinates_mat
        Nb = self.node_ij_mat.shape[0]
        strain_vec = np.zeros(Nb)
        for i in range(Nb):
            node1 = self.node_ij_mat[i,0]-1
            node2 = self.node_ij_mat[i,1]-1
            X1 = coords[node1] + U[node1]
            X2 = coords[node2] + U[node2]
            strain_vec[i] = (np.linalg.norm(X1 - X2) - self.L0_vec[i]) / self.L0_vec[i]
        return strain_vec
