# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 21:46:04 2025

@author: zzhuy
"""

import numpy as np
from typing import Tuple, Any


class Vec_Elements_Bars:
    """
    Bar elements calculated based on analytical equations.
    This code is vectorized for speed.
    This formulation gives a linear elastic response.
    The bar geometry is defined with two nodes.
    """
    
    def __init__(self):
        """Initialize the bar element properties."""
        # Connection information of the bar, stored as a matrix (Nb x 2)
        self.node_ij_mat = None
        
        # Area of the bar, stored as a vector (Nb x 1)
        self.A_vec = None
        
        # Young's Modulus of the bar, stored as a vector (Nb x 1)
        self.E_vec = None
        
        # Length of the bar, stored as a vector (Nb x 1)
        self.L0_vec = None
        
        # Current Engineering Strain of the bar, stored as a vector (Nb x 1)
        self.strain_current_vec = None
        
        # Current Strain Energy of the bar, stored as a vector (Nb x 1)
        self.energy_current_vec = None
    
    def Initialize(self, node: Any) -> None:
        """
        Initialize the original length of bars.
        
        Args:
            node: Node object containing coordinates_mat attribute
        """
        self.L0_vec = np.zeros_like(self.A_vec,dtype=np.float64)
        
        # Calculate the undeformed length of each bar element
        for i in range(len(self.A_vec)):
            node1 = node.coordinates_mat[self.node_ij_mat[i, 0]-1, :]
            node2 = node.coordinates_mat[self.node_ij_mat[i, 1]-1, :]
            self.L0_vec[i] = np.linalg.norm(node1 - node2)
    
    def solve_strain(self, node: Any, U: np.ndarray) -> np.ndarray:
        """
        Calculate the strain of bars.
        
        Args:
            node: Node object containing coordinates_mat attribute
            U: Displacement matrix
            
        Returns:
            Ex: Strain vector
        """
        nodal_coordinates = node.coordinates_mat
        bar_connect = self.node_ij_mat
        bar_length = self.L0_vec
        
        node_index1 = bar_connect[:, 0]-1
        node_index2 = bar_connect[:, 1]-1
        
        node1 = nodal_coordinates[node_index1, :]
        node2 = nodal_coordinates[node_index2, :]
        
        # B1n calculation
        bar_length_squared = bar_length * bar_length
        diff = node2 - node1
        B1n = np.hstack([
            -(diff) / bar_length_squared[:, np.newaxis],
            (diff) / bar_length_squared[:, np.newaxis]
        ])
        
        # Identity matrix pattern
        iden = np.eye(3)
        iden_mat = np.block([
            [iden, -iden],
            [-iden, iden]
        ])
        
        # Prepare U_temp for all bars at once
        n_bars = len(node_index1)
        U_temp = np.zeros((6, n_bars))
        U_temp[0:3, :] = U[node_index1, :].T
        U_temp[3:6, :] = U[node_index2, :].T
        
        # B2U calculation
        B2U_temp = np.zeros((n_bars, 6))
        for i in range(n_bars):
            B2U_temp[i, :] = (iden_mat @ U_temp[:, i]) / bar_length_squared[i]
        
        # Calculate strain
        Ex = np.sum(B1n * U_temp.T, axis=1) + 0.5 * np.sum(B2U_temp * U_temp.T, axis=1)
        
        return Ex
    
    def solve_stress(self, Ex: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the stress and stiffness of bars.
        This function defines the constitutive model for bars.
        
        Args:
            Ex: Strain vector
            
        Returns:
            Sx: Stress vector
            Cx: Material stiffness vector
        """
        Sx = self.E_vec * Ex
        Cx = self.E_vec
        
        return Sx, Cx
    
    def solve_global_force(self, node: Any, U: np.ndarray, Sx: np.ndarray) -> np.ndarray:
        """
        Calculate the global force vector.
        
        Args:
            node: Node object containing coordinates_mat attribute
            U: Displacement matrix
            Sx: Stress vector
            
        Returns:
            Tbar: Global force vector
        """
        nodal_coordinates = node.coordinates_mat
        bar_connect = self.node_ij_mat
        bar_length = self.L0_vec
        bar_area = self.A_vec
        
        N = nodal_coordinates.shape[0]
        Tbar = np.zeros(3 * N)
        
        node_index1 = bar_connect[:, 0]-1
        node_index2 = bar_connect[:, 1]-1
        node1 = nodal_coordinates[node_index1, :]
        node2 = nodal_coordinates[node_index2, :]
        
        # B1n calculation
        bar_length_squared = bar_length * bar_length
        diff = node2 - node1
        B1n = np.hstack([
            -(diff) / bar_length_squared[:, np.newaxis],
            (diff) / bar_length_squared[:, np.newaxis]
        ])
        
        # Identity matrix pattern
        iden = np.eye(3)
        iden_mat = np.block([
            [iden, -iden],
            [-iden, iden]
        ])
        
        # Prepare U_temp
        n_bars = len(node_index1)
        U_temp = np.zeros((6, n_bars))
        U_temp[0:3, :] = U[node_index1, :].T
        U_temp[3:6, :] = U[node_index2, :].T
        
        # B2U calculation
        B2U_temp = np.zeros((n_bars, 6))
        for i in range(n_bars):
            B2U_temp[i, :] = (iden_mat @ U_temp[:, i]) / bar_length_squared[i]
        
        # Calculate forces
        T_temp = Sx * bar_area * bar_length
        T_temp = T_temp[:, np.newaxis] * (B1n + B2U_temp)
        
        # Assembly indices
        index1 = 3 * (node_index1+1) - 2  # Convert to 0-based indexing later
        index2 = 3 * (node_index2+1) - 2
        
        # Assemble into global force vector
        for i in range(n_bars):
            # Convert MATLAB 1-based to Python 0-based indexing
            idx1 = index1[i] - 1
            idx2 = index2[i] - 1
            
            Tbar[idx1:idx1+3] += T_temp[i, 0:3]
            Tbar[idx2:idx2+3] += T_temp[i, 3:6]
        
        return Tbar
    
    def solve_global_stiff(self, node: Any, U: np.ndarray, Sx: np.ndarray, C: np.ndarray) -> np.ndarray:
        """
        Calculate the global stiffness matrix.
        
        Args:
            node: Node object containing coordinates_mat attribute
            U: Displacement matrix
            Sx: Stress vector
            C: Material stiffness vector
            
        Returns:
            Kbar: Global stiffness matrix
        """
        nodal_coordinates = node.coordinates_mat
        bar_connect = self.node_ij_mat
        bar_length = self.L0_vec
        bar_area = self.A_vec
        
        N_node = nodal_coordinates.shape[0]
        N_bar = C.shape[0]
        
        Kbar = np.zeros((3 * N_node, 3 * N_node))
        
        node_index1_temp = bar_connect[:, 0]-1
        node_index2_temp = bar_connect[:, 1]-1
        node1_temp = nodal_coordinates[node_index1_temp, :]
        node2_temp = nodal_coordinates[node_index2_temp, :]
        
        bar_length_square = bar_length * bar_length
        
        # B1 calculation
        diff = node2_temp - node1_temp
        B1_temp = np.hstack([
            -(diff) / bar_length_square[:, np.newaxis],
            (diff) / bar_length_square[:, np.newaxis]
        ])
        
        
        # B2 pattern
        iden = np.eye(3)
        B2_pattern = np.block([
            [iden, -iden],
            [-iden, iden]
        ])
        
        # Create B2_temp matrix (N_bar x 36)
        B2_temp = np.zeros((N_bar, 36))
        for i in range(N_bar):
            factor = 1.0 / bar_length_square[i]
            B2_temp[i, :] = factor * B2_pattern.flatten()
        
        # Reshape B2_temp to (N_bar x 6 x 6)
        B2_temp_reshaped = B2_temp.reshape(N_bar, 6, 6)
        
        # U_temp preparation
        U_temp = np.hstack([U[node_index1_temp, :], U[node_index2_temp, :]])
        
        # Calculate B2_U
        B2_U = np.zeros((N_bar, 6))
        for i in range(N_bar):
            for j in range(6):
                B2_U[i, j] = np.dot(B2_temp_reshaped[i, j, :], U_temp[i, :])
        
        # B1_B2_U calculation
        B1_B2_U = B1_temp + B2_U
        
        
        # Calculate K_temp
        K_temp = np.zeros((N_bar, 6, 6))
        for i in range(N_bar):
            # First part: C * A * L * (B1_B2_U)^T * (B1_B2_U)
            K_temp_part1 = np.outer(B1_B2_U[i, :], B1_B2_U[i, :])
            
            # Full K_temp calculation
            K_temp[i, :, :] = (C[i] * bar_area[i] * bar_length[i] * K_temp_part1 + 
                               Sx[i] * bar_area[i] * bar_length[i] * B2_temp_reshaped[i, :, :])
        
        # Assembly indices (convert to 0-based)
        index1 = 3 * (node_index1_temp+1) - 3
        index2 = 3 * (node_index2_temp+1) - 3
        
        # Assemble into global stiffness matrix
        for i in range(N_bar):
            # Get the 6x6 local stiffness matrix for bar i
            K_local = K_temp[i, :, :]
            
            # Get global indices for this bar
            idx = np.concatenate([
                np.arange(index1[i], index1[i] + 3),
                np.arange(index2[i], index2[i] + 3)
            ])
            
            # Add to global stiffness matrix
            for j in range(6):
                for k in range(6):
                    Kbar[idx[j], idx[k]] += K_local[j, k]
        
        return Kbar
    
    def Solve_FK(self, node: Any, U: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Main function to compute the global forces and stiffness of the bar elements.
        
        Args:
            node: Node object containing coordinates_mat attribute
            U: Displacement matrix
            
        Returns:
            Tbar: Global force vector
            Kbar: Global stiffness matrix
        """
        Ex = self.solve_strain(node, U)
        Sx, C = self.solve_stress(Ex)
        Tbar = self.solve_global_force(node, U, Sx)
        Kbar = self.solve_global_stiff(node, U, Sx, C)
        
        self.strain_current_vec = Ex
        self.energy_current_vec = 0.5 * self.E_vec * self.A_vec * self.L0_vec * Ex * Ex
        
        return Tbar, Kbar