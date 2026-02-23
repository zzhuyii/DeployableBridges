# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 16:03:20 2025

@author: zzhuy
"""

import numpy as np

class Vec_Elements_CST:
    """
    CST (Constant Strain Triangle) elements class
    This element is derived using analytical equations
    This code is vectorized for speed 
    This is a constant strain linear elastic formulation
    The CST is constructed using 3 nodes
    """
    
    def __init__(self):
        # Connection information of CST elements (Ncst x 3)
        self.node_ijk_mat = None
        
        # Thickness of each element (Ncst x 1)
        self.t_vec = None
        
        # Young's modulus of each element (Ncst x 1)
        self.E_vec = None
        
        # Poisson's Ratio of each element (Ncst x 1)
        self.v_vec = None
        
        # Triangle Area of each element (Ncst x 1)
        self.A_vec = None
        
        # Original Length of each side (Ncst x 3)
        self.L_mat = None
        
        # Current Strain Energy (Ncst x 1)
        self.energy_current_vec = None
    
    def Initialize(self, node):
        """
        Initialize CST elements
        This includes solving for the A_vec, L_mat
        
        Args:
            node: Node object with coordinates_mat attribute
        """
        numCST = self.node_ijk_mat.shape[0]
        self.A_vec = np.zeros(numCST)
        
        # First pass: reorganize nodes to ensure largest angle is alpha1
        for i in range(numCST):
            # Identify the nodal coordinates
            n1 = self.node_ijk_mat[i, 0]-1
            n2 = self.node_ijk_mat[i, 1]-1
            n3 = self.node_ijk_mat[i, 2]-1
            
            node_index = [n1, n2, n3]
            
            # The three nodal coordinates of nodes
            X1 = node.coordinates_mat[n1, :]
            X2 = node.coordinates_mat[n2, :]
            X3 = node.coordinates_mat[n3, :]
            
            # Find each sector angle
            vtemp1 = (X2 - X1) / np.linalg.norm(X2 - X1)
            vtemp2 = (X3 - X1) / np.linalg.norm(X3 - X1)
            beta1 = np.arccos(np.clip(np.dot(vtemp1, vtemp2), -1, 1))
            
            vtemp1 = (X1 - X2) / np.linalg.norm(X1 - X2)
            vtemp2 = (X3 - X2) / np.linalg.norm(X3 - X2)
            beta2 = np.arccos(np.clip(np.dot(vtemp1, vtemp2), -1, 1))
            
            vtemp1 = (X1 - X3) / np.linalg.norm(X1 - X3)
            vtemp2 = (X2 - X3) / np.linalg.norm(X2 - X3)
            beta3 = np.arccos(np.clip(np.dot(vtemp1, vtemp2), -1, 1))
            
            # Rank the sector angles by size
            beta_vec = [beta1, beta2, beta3]
            index = np.argsort(beta_vec)
            
            # Reorganize the node sequence (largest angle first)
            self.node_ijk_mat[i, 0] = node_index[index[2]]+1
            self.node_ijk_mat[i, 1] = node_index[index[1]]+1
            self.node_ijk_mat[i, 2] = node_index[index[0]]+1
        
        # Initialize L_mat
        self.L_mat = np.zeros((numCST, 3))
        
        # Second pass: calculate lengths and areas
        for i in range(numCST):
            # Identify the nodal coordinates
            n1 = self.node_ijk_mat[i, 0]-1
            n2 = self.node_ijk_mat[i, 1]-1
            n3 = self.node_ijk_mat[i, 2]-1
            
            # The three nodal coordinates of nodes
            X1 = node.coordinates_mat[n1, :]
            X2 = node.coordinates_mat[n2, :]
            X3 = node.coordinates_mat[n3, :]
            
            L1 = np.linalg.norm(X2 - X3)
            L2 = np.linalg.norm(X1 - X3)
            L3 = np.linalg.norm(X1 - X2)
            
            # Store the lengths
            self.L_mat[i, 0] = L1
            self.L_mat[i, 1] = L2
            self.L_mat[i, 2] = L3
            
            # Find the area of the triangle
            v1 = X3 - X1
            v2 = X2 - X1
            area = np.linalg.norm(np.cross(v1, v2)) / 2
            
            # Store the area info
            self.A_vec[i] = area
    
    def solve_bar_strain(self, U, X0):
        """
        Solve the bar strain of the elements
        
        Args:
            U: Displacement vector
            X0: Initial coordinates
            
        Returns:
            bar_strain_mat: Bar strain matrix
            l_mat: Current length matrix
            x_reshape: Reshaped coordinates
            trans_mat: Transformation matrix
        """
        x = U + X0  # Deformed nodes
        ijk_mat = self.node_ijk_mat  # Connectivity matrix
        
        # Number of CST elements
        Ncst = ijk_mat.shape[0]
        
        x_reshape = np.zeros((Ncst, 9))
        x_reshape[:, 0:3] = x[ijk_mat[:, 0]-1, :]
        x_reshape[:, 3:6] = x[ijk_mat[:, 1]-1, :]
        x_reshape[:, 6:9] = x[ijk_mat[:, 2]-1, :]
        
        L1_vec = self.L_mat[:, 0]
        L2_vec = self.L_mat[:, 1]
        L3_vec = self.L_mat[:, 2]
        
        l1_vec = x_reshape[:, 3:6] - x_reshape[:, 6:9]
        l2_vec = x_reshape[:, 0:3] - x_reshape[:, 6:9]
        l3_vec = x_reshape[:, 3:6] - x_reshape[:, 0:3]
        
        l1_vec = np.linalg.norm(l1_vec, axis=1)
        l2_vec = np.linalg.norm(l2_vec, axis=1)
        l3_vec = np.linalg.norm(l3_vec, axis=1)
        
        # Engineering strain of each side
        epsilon1_vec = (l1_vec - L1_vec) / L1_vec
        epsilon2_vec = (l2_vec - L2_vec) / L2_vec
        epsilon3_vec = (l3_vec - L3_vec) / L3_vec
        
        # Output results
        bar_strain_mat = np.column_stack([epsilon1_vec, epsilon2_vec, epsilon3_vec])
        l_mat = np.column_stack([l1_vec, l2_vec, l3_vec])
        
        # Update alpha angles and transformation matrix
        cos1_vec = np.sum((x_reshape[:, 3:6] - x_reshape[:, 0:3]) * 
                         (x_reshape[:, 6:9] - x_reshape[:, 0:3]), axis=1) / (l2_vec * l3_vec)
        
        cos2_vec = np.sum((x_reshape[:, 0:3] - x_reshape[:, 3:6]) * 
                         (x_reshape[:, 6:9] - x_reshape[:, 3:6]), axis=1) / (l1_vec * l3_vec)
        
        cos3_vec = np.sum((x_reshape[:, 0:3] - x_reshape[:, 6:9]) * 
                         (x_reshape[:, 3:6] - x_reshape[:, 6:9]), axis=1) / (l1_vec * l2_vec)
        
        beta1_vec = np.arccos(np.clip(cos1_vec, -1, 1))
        beta2_vec = np.arccos(np.clip(cos2_vec, -1, 1))
        beta3_vec = np.arccos(np.clip(cos3_vec, -1, 1))
        
        # Solve for transformation matrix
        alpha2_vec = beta2_vec
        alpha3_vec = np.pi - beta3_vec
        
        tan2_vec = np.tan(alpha2_vec)
        tan3_vec = np.tan(alpha3_vec)
        
        cot2_vec = 1.0 / tan2_vec
        cot3_vec = 1.0 / tan3_vec
        
        sin2_vec = np.sin(alpha2_vec)
        sin3_vec = np.sin(alpha3_vec)
        
        cos2_vec = np.cos(alpha2_vec)
        cos3_vec = np.cos(alpha3_vec)
        
        # Transformation factors
        B1_vec = (tan3_vec - tan2_vec) / (cot2_vec - cot3_vec)
        B2_vec = (-1.0 / (cot2_vec - cot3_vec)) / sin3_vec / cos3_vec
        B3_vec = (1.0 / (cot2_vec - cot3_vec)) / sin2_vec / cos2_vec
        
        C1_vec = (tan3_vec**2 - tan2_vec**2) / 2.0 / (tan2_vec - tan3_vec)
        C2_vec = -1.0 / 2.0 / cos3_vec**2 / (tan2_vec - tan3_vec)
        C3_vec = 1.0 / 2.0 / cos2_vec**2 / (tan2_vec - tan3_vec)
        
        # Factor of 2 for shear
        C1_vec *= 2
        C2_vec *= 2
        C3_vec *= 2
        
        trans_mat = np.zeros((Ncst, 3, 3))
        trans_mat[:, 0, :] = np.tile([1, 0, 0], (Ncst, 1))
        trans_mat[:, 1, 0] = B1_vec
        trans_mat[:, 1, 1] = B2_vec
        trans_mat[:, 1, 2] = B3_vec
        trans_mat[:, 2, 0] = C1_vec
        trans_mat[:, 2, 1] = C2_vec
        trans_mat[:, 2, 2] = C3_vec
        
        return bar_strain_mat, l_mat, x_reshape, trans_mat
    
    def solve_cst_strain(self, bar_strain_mat, trans_mat):
        """
        Compute CST strain from bar strain
        
        Args:
            bar_strain_mat: Bar strain matrix
            trans_mat: Transformation matrix
            
        Returns:
            cst_strain_mat: CST strain matrix
        """
        epsilon1_vec = bar_strain_mat[:, 0]
        epsilon2_vec = bar_strain_mat[:, 1]
        epsilon3_vec = bar_strain_mat[:, 2]
        
        B1_vec = trans_mat[:, 1, 0]
        B2_vec = trans_mat[:, 1, 1]
        B3_vec = trans_mat[:, 1, 2]
        
        C1_vec = trans_mat[:, 2, 0]
        C2_vec = trans_mat[:, 2, 1]
        C3_vec = trans_mat[:, 2, 2]
        
        # Convert bar strain to CST element strain
        epsilon_p_vec = (B1_vec * epsilon1_vec + 
                        B2_vec * epsilon2_vec + 
                        B3_vec * epsilon3_vec)
        
        gamma_vec = (C1_vec * epsilon1_vec + 
                    C2_vec * epsilon2_vec + 
                    C3_vec * epsilon3_vec)
        
        cst_strain_mat = np.column_stack([epsilon1_vec, epsilon_p_vec, gamma_vec])
        
        return cst_strain_mat
    
    def solve_derivatives(self, x_reshape, l_mat):
        """
        Compute Jacobian and Hessian of epsilon matrix
        
        Args:
            x_reshape: Reshaped coordinates
            l_mat: Current length matrix
            
        Returns:
            dedx: Jacobian matrix (Ncst, 3, 9)
            d2edx2: Hessian matrix (Ncst, 3, 9, 9)
        """
        ijk_mat = self.node_ijk_mat
        Ncst = ijk_mat.shape[0]
        
        L1_vec = self.L_mat[:, 0]
        L2_vec = self.L_mat[:, 1]
        L3_vec = self.L_mat[:, 2]
        
        L_mat = self.L_mat
        
        l1_vec = l_mat[:, 0]
        l2_vec = l_mat[:, 1]
        l3_vec = l_mat[:, 2]
        
        x1_mat = x_reshape[:, 0:3]
        x2_mat = x_reshape[:, 3:6]
        x3_mat = x_reshape[:, 6:9]
        
        # Derivatives for epsilon 1
        direction1 = np.hstack([np.zeros((Ncst, 3)), 
                               x2_mat - x3_mat, 
                               x3_mat - x2_mat])
        deps1dx = direction1 / (l1_vec[:, np.newaxis] * L1_vec[:, np.newaxis])
        
        # Derivatives for epsilon 2
        direction2 = np.hstack([x1_mat - x3_mat, 
                               np.zeros((Ncst, 3)), 
                               x3_mat - x1_mat])
        deps2dx = direction2 / (l2_vec[:, np.newaxis] * L2_vec[:, np.newaxis])
        
        # Derivatives for epsilon 3
        direction3 = np.hstack([x1_mat - x2_mat, 
                               x2_mat - x1_mat, 
                               np.zeros((Ncst, 3))])
        deps3dx = direction3 / (l3_vec[:, np.newaxis] * L3_vec[:, np.newaxis])
        
        # Organize Jacobian
        dedx = np.zeros((Ncst, 3, 9))
        dedx[:, 0, :] = deps1dx
        dedx[:, 1, :] = deps2dx
        dedx[:, 2, :] = deps3dx
        
        # Solve for Hessian matrix
        d2edx2 = np.zeros((Ncst, 3, 9, 9))
        
        # First part of Hessian for strain 1
        K1e1 = np.zeros((Ncst, 9, 9))
        K1e1[:, 3:6, 3:6] = np.tile(np.eye(3), (Ncst, 1, 1))
        K1e1[:, 6:9, 6:9] = np.tile(np.eye(3), (Ncst, 1, 1))
        K1e1[:, 3:6, 6:9] = -np.tile(np.eye(3), (Ncst, 1, 1))
        K1e1[:, 6:9, 3:6] = -np.tile(np.eye(3), (Ncst, 1, 1))
        
        # Apply scaling factors
        for idx in [3, 4, 5, 6, 7, 8]:
            K1e1[:, idx, 3:9] = (K1e1[:, idx, 3:9] / 
                                (L_mat[:, 0][:, np.newaxis] * l_mat[:, 0][:, np.newaxis]))
        
        # Similar operations for K1e2 and K1e3...
        # (Implementing the full vectorized operations as in original MATLAB)
        
        # For brevity, I'll implement a simplified version
        # The full implementation would follow the same pattern as K1e1
        
        K1e2 = np.zeros((Ncst, 9, 9))
        K1e2[:, 0:3, 0:3] = np.tile(np.eye(3), (Ncst, 1, 1))
        K1e2[:, 6:9, 6:9] = np.tile(np.eye(3), (Ncst, 1, 1))
        K1e2[:, 0:3, 6:9] = -np.tile(np.eye(3), (Ncst, 1, 1))
        K1e2[:, 6:9, 0:3] = -np.tile(np.eye(3), (Ncst, 1, 1))
        
        for idx in [0, 1, 2, 6, 7, 8]:
            indices = [0, 1, 2, 6, 7, 8]
            K1e2[:, idx, indices] = (K1e2[:, idx, indices] / 
                                    (L_mat[:, 1][:, np.newaxis] * l_mat[:, 1][:, np.newaxis]))
        
        K1e3 = np.zeros((Ncst, 9, 9))
        K1e3[:, 3:6, 3:6] = np.tile(np.eye(3), (Ncst, 1, 1))
        K1e3[:, 0:3, 0:3] = np.tile(np.eye(3), (Ncst, 1, 1))
        K1e3[:, 3:6, 0:3] = -np.tile(np.eye(3), (Ncst, 1, 1))
        K1e3[:, 0:3, 3:6] = -np.tile(np.eye(3), (Ncst, 1, 1))
        
        for idx in [0, 1, 2, 3, 4, 5]:
            K1e3[:, idx, 0:6] = (K1e3[:, idx, 0:6] / 
                                (L_mat[:, 2][:, np.newaxis] * l_mat[:, 2][:, np.newaxis]))
        
        # Assemble first part
        d2edx2[:, 0, :, :] = K1e1
        d2edx2[:, 1, :, :] = K1e2
        d2edx2[:, 2, :, :] = K1e3
        
        # Second part of Hessian (geometric nonlinearity terms)
        # This involves outer products of direction vectors
        for i in range(9):
            for j in range(9):
                # For strain 1
                d2edx2[:, 0, i, j] += (direction1[:, i] * direction1[:, j] / 
                                      (L_mat[:, 0] * l_mat[:, 0]**3))
                # For strain 2  
                d2edx2[:, 1, i, j] += (direction2[:, i] * direction2[:, j] / 
                                      (L_mat[:, 1] * l_mat[:, 1]**3))
                # For strain 3
                d2edx2[:, 2, i, j] += (direction3[:, i] * direction3[:, j] / 
                                      (L_mat[:, 2] * l_mat[:, 2]**3))
        
        return dedx, d2edx2
    
    def solve_global_force(self, U, dedx, cst_strain_mat, trans_mat):
        """
        Calculate global internal force
        
        Args:
            U: Displacement vector
            dedx: Jacobian matrix
            cst_strain_mat: CST strain matrix
            trans_mat: Transformation matrix
            
        Returns:
            Tcst: Global force vector
        """
        cst_ijf = self.node_ijk_mat
        Ncst = cst_ijf.shape[0]
        
        # Material properties
        E_vec = self.E_vec
        v_vec = self.v_vec
        
        E_mat = np.zeros((Ncst, 3, 3))
        E_mat[:, 0, 0] = E_vec / (1 - v_vec**2)
        E_mat[:, 1, 1] = E_vec / (1 - v_vec**2)
        E_mat[:, 0, 1] = v_vec * E_vec / (1 - v_vec**2)
        E_mat[:, 1, 0] = v_vec * E_vec / (1 - v_vec**2)
        E_mat[:, 2, 2] = E_vec / (1 + v_vec) / 2
        
        # Calculate stress
        sigma_mat = np.zeros((Ncst, 3))
        for i in range(3):
            sigma_mat[:, i] = np.sum(E_mat[:, i, :] * cst_strain_mat, axis=1)
        
        # Transform derivatives
        depdx = np.zeros_like(dedx)
        for i in range(9):
            for j in range(3):
                depdx[:, j, i] = np.sum(trans_mat[:, j, :] * dedx[:, :, i], axis=1)
        
        # Calculate local forces
        localF_mat = np.zeros((Ncst, 9))
        for i in range(9):
            localF_mat[:, i] = (np.sum(sigma_mat * depdx[:, :, i], axis=1) * 
                               self.A_vec * self.t_vec)
        
        # Assemble global force vector
        nodeNum = U.shape[0]
        Tcst = np.zeros(3 * nodeNum)
        
        ijk_mat = self.node_ijk_mat.flatten()
        localF_flat = localF_mat.flatten()
      
        
        for i in range(Ncst * 3):
            node_num = ijk_mat[i]-1

            Tcst[(3*node_num):(3*node_num+3)] += localF_flat[(3*i):(3*i+3)]
        
        return Tcst
    
    def solve_global_stiff(self, U, dedx, d2edx2, cst_strain_mat, trans_mat):
        """
        Calculate global stiffness matrix
        
        Args:
            U: Displacement vector
            dedx: Jacobian matrix
            d2edx2: Hessian matrix
            cst_strain_mat: CST strain matrix
            trans_mat: Transformation matrix
            
        Returns:
            Kcst: Global stiffness matrix
        """
        cst_ijk = self.node_ijk_mat
        Ncst = cst_ijk.shape[0]
        
        # Material properties
        E_vec = self.E_vec
        t_vec = self.t_vec
        v_vec = self.v_vec
        
        E_mat = np.zeros((Ncst, 3, 3))
        E_mat[:, 0, 0] = E_vec / (1 - v_vec**2)
        E_mat[:, 1, 1] = E_vec / (1 - v_vec**2)
        E_mat[:, 0, 1] = v_vec * E_vec / (1 - v_vec**2)
        E_mat[:, 1, 0] = v_vec * E_vec / (1 - v_vec**2)
        E_mat[:, 2, 2] = E_vec / (1 + v_vec) / 2
        
        nodeNum = U.shape[0]
        
        # Initialize local stiffness matrices
        K1cst_local = np.zeros((Ncst, 9, 9))
        K2cst_local = np.zeros((Ncst, 9, 9))
        
        # Transform second derivatives
        d2edx2_cst = np.zeros_like(d2edx2)
        for i in range(9):
            for j in range(9):
                for k in range(3):
                    d2edx2_cst[:, k, i, j] = np.sum(trans_mat[:, k, :] * d2edx2[:, :, i, j], axis=1)
        
        # Calculate stress
        sigma_mat = np.zeros((Ncst, 3))
        for i in range(3):
            sigma_mat[:, i] = np.sum(E_mat[:, i, :] * cst_strain_mat, axis=1)
        
        # Geometric stiffness (K1)
        dude = sigma_mat * (self.A_vec[:, np.newaxis] * self.t_vec[:, np.newaxis])
        
        for i in range(9):
            for j in range(9):
                K1cst_local[:, i, j] = np.sum(dude * d2edx2_cst[:, :, i, j], axis=1)
        
        # Transform first derivatives
        dedx_cst = np.zeros_like(dedx)
        for i in range(9):
            for j in range(3):
                dedx_cst[:, j, i] = np.sum(trans_mat[:, j, :] * dedx[:, :, i], axis=1)
        
        # Material stiffness (K2)
        d2ude2 = np.zeros((Ncst, 3, 3))
        for i in range(3):
            d2ude2[:, i, :] = E_mat[:, i, :] * (self.A_vec[:, np.newaxis] * self.t_vec[:, np.newaxis])
        
        # Calculate K2
        Rhs = np.zeros((Ncst, 3, 9))
        for i in range(9):
            for j in range(3):
                Rhs[:, j, i] = np.sum(d2ude2[:, j, :] * dedx_cst[:, :, i], axis=1)
        
        for i in range(9):
            for j in range(9):
                K2cst_local[:, i, j] = np.sum(dedx_cst[:, :, i] * Rhs[:, :, j], axis=1)
        
        # Assemble global stiffness matrix
        Kcst = np.zeros((nodeNum * 3, nodeNum * 3))
        
        for i in range(Ncst):
            # Get node indices
            n1, n2, n3 = cst_ijk[i, :]-1
            
            # Local to global mapping
            dof_map = np.array([3*n1, 3*n1+1, 3*n1+2, 
                               3*n2, 3*n2+1, 3*n2+2,
                               3*n3, 3*n3+1, 3*n3+2])
            
            # Add local stiffness to global
            K_local = K1cst_local[i] + K2cst_local[i]
            
            for ii in range(9):
                for jj in range(9):
                    Kcst[dof_map[ii], dof_map[jj]] += K_local[ii, jj]
        
        return Kcst
    
    def Solve_FK(self, node, U):
        """
        Main function to compute global forces and stiffness
        
        Args:
            node: Node object
            U: Displacement vector
            
        Returns:
            Tcst: Global force vector
            Kcst: Global stiffness matrix
        """
        bar_strain_mat, l_mat, x_reshape, trans_mat = self.solve_bar_strain(U, node.coordinates_mat)
        
        cst_strain_mat = self.solve_cst_strain(bar_strain_mat, trans_mat)
        
        dedx, d2edx2 = self.solve_derivatives(x_reshape, l_mat)
        
        Tcst = self.solve_global_force(U, dedx, cst_strain_mat, trans_mat)
        
        Kcst = self.solve_global_stiff(U, dedx, d2edx2, cst_strain_mat, trans_mat)
        
        return Tcst, Kcst