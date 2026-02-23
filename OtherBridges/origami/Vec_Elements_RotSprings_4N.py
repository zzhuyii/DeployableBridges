# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 13:13:46 2025

@author: zzhuy
"""

import numpy as np
from typing import Tuple, Optional
import pandas as pd

class Vec_Elements_RotSprings_4N:
    """
    4-Node Rotational Spring Elements
    
    This rotational spring element is derived using analytical equations.
    The rotational spring element is vectorized.
    The formulation is for a linear-elastic rotational spring.
    The rotational spring geometry is defined with 4 nodes.
    """
    
    def __init__(self):
        # Node number of the four nodes used to define the rotational 
        # spring elements (Ns*4)
        self.node_ijkl_mat = None
        
        # Rotational stiffness of each element (Ns*1)
        self.rot_spr_K_vec = None
        
        # Stress-free angle of the spring
        self.theta_stress_free_vec = None
        
        # Current Theta
        self.theta_current_vec = None
        
        # Current Strain Energy
        self.energy_current_vec = None
        
        # Threshold for local penetration prevention
        # The theta1 and theta2 are terms for initiating the penetration 
        # prevention. When folding angle is smaller than theta1 or bigger 
        # than theta2, an additional force is added. Please check the 
        # Liu and Paulino RSPA paper for details.
        self.theta1 = 0.1 * np.pi
        self.theta2 = 2 * np.pi - 0.1 * np.pi
    
    def Initialize(self, node):
        """
        This function initializes the rotational spring elements.
        The initialization process includes calculating the current folding angle
        and the current stress-free angle of the spring elements. We set the
        current state to be the stress-free state.
        
        Parameters:
        -----------
        node : object
            Node object containing coordinates_mat attribute
        """
        rot_spr_ijkl = self.node_ijkl_mat
        num_spr = rot_spr_ijkl.shape[0]
        
        # Check the ijkl assignment to make sure they are the same direction
        for i in range(num_spr):
            n1 = rot_spr_ijkl[i, 0]-1
            n2 = rot_spr_ijkl[i, 1]-1
            n3 = rot_spr_ijkl[i, 2]-1
            n4 = rot_spr_ijkl[i, 3]-1
            
            x1 = node.coordinates_mat[n1, :]
            x2 = node.coordinates_mat[n2, :]
            x3 = node.coordinates_mat[n3, :]
            x4 = node.coordinates_mat[n4, :]
            
            v1 = x1 - x2
            v2 = x3 - x2
            
            norm1 = np.cross(v1, v2)
            
            if norm1[2] != 0:
                if norm1[2] < 0:
                    self.node_ijkl_mat[i, 1] = n3+1
                    self.node_ijkl_mat[i, 2] = n2+1
            elif norm1[1] != 0:
                if norm1[1] < 0:
                    self.node_ijkl_mat[i, 1] = n3+1
                    self.node_ijkl_mat[i, 2] = n2+1
            else:
                if norm1[0] < 0:
                    self.node_ijkl_mat[i, 1] = n3+1
                    self.node_ijkl_mat[i, 2] = n2+1
        
        node_size = node.coordinates_mat.shape[0]
        self.theta_current_vec = self.solve_theta(node, np.zeros((node_size, 3), dtype=np.float64))
        self.theta_stress_free_vec = self.theta_current_vec.copy()
    
    def solve_theta(self, node, U) -> np.ndarray:
        """
        Calculate the rotational angle of rotational springs.
        
        Parameters:
        -----------
        node : object
            Node object containing coordinates_mat
        U : np.ndarray
            Displacement matrix
            
        Returns:
        --------
        theta : np.ndarray
            Rotational angles
        """
        nodal_coordinate = node.coordinates_mat
        spr_ijkl = self.node_ijkl_mat
        
        spr_i = spr_ijkl[:, 0]-1
        spr_j = spr_ijkl[:, 1]-1
        spr_k = spr_ijkl[:, 2]-1
        spr_l = spr_ijkl[:, 3]-1
        
        node_i = nodal_coordinate[spr_i, :] + U[spr_i, :]
        node_j = nodal_coordinate[spr_j, :] + U[spr_j, :]
        node_k = nodal_coordinate[spr_k, :] + U[spr_k, :]
        node_l = nodal_coordinate[spr_l, :] + U[spr_l, :]
        
        rij = node_i - node_j
        rkj = node_k - node_j
        rkl = node_k - node_l
        
        m = np.cross(rij, rkj, axis=1)
        n = np.cross(rkj, rkl, axis=1)
        
        d_m_kl = np.sum(m * rkl, axis=1)
        zero_index = np.where(d_m_kl == 0)[0]
        
        yita = np.sign(d_m_kl)
        yita[zero_index] = 1
        
        m_norm = np.sqrt(np.sum(m * m, axis=1))
        n_norm = np.sqrt(np.sum(n * n, axis=1))
        
        cos_theta = np.sum(m * n, axis=1) / (m_norm * n_norm)
        # Clamp values to avoid numerical issues with acos
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        
        theta = np.mod(yita * np.real(np.arccos(cos_theta)), 2 * np.pi)
        
        return theta
    
    def solve_moment(self, theta) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the local moment and stiffness of the rotational spring
        elements. This function gives the constitutive relationship of 
        rotational spring elements.
        
        Parameters:
        -----------
        theta : np.ndarray
            Current angles
            
        Returns:
        --------
        Mspr : np.ndarray
            Moments
        Cspr : np.ndarray
            Stiffnesses
        """
        theta1 = self.theta1
        theta2 = self.theta2
        
        spr_rot_k = self.rot_spr_K_vec
        spr_stress_free = self.theta_stress_free_vec
        
        Mspr = np.zeros_like(theta,dtype=np.float64)
        Cspr = np.zeros_like(theta,dtype=np.float64)
        
        n_rot_spr = theta.shape[0]
        
        # This following part is not vectorized yet
        # The logical judgment for facade penetration is not easily 
        # vectorized due to the multiple "if" statements. However, 
        # because this part is not expensive computationally, we can
        # safely live with the for loop.
        
        for i in range(n_rot_spr):
            if theta[i] < theta1:
                ratio = (1 / np.cos(np.pi * (theta[i] - theta1) / (2 * theta1))) ** 2
                Cspr[i] = spr_rot_k[i] * ratio
                Mspr[i] = (spr_rot_k[i] * (theta[i] - spr_stress_free[i]) +
                          (2 * theta1 * spr_rot_k[i] / np.pi) * 
                          np.tan(np.pi * (theta[i] - theta1) / (2 * theta1)) -
                          spr_rot_k[i] * theta[i] + spr_rot_k[i] * theta1)
            
            elif theta[i] > theta2:
                ratio = (1 / np.cos(np.pi * (theta[i] - theta2) / (4 * np.pi - 2 * theta2))) ** 2
                Cspr[i] = spr_rot_k[i] * ratio
                Mspr[i] = (spr_rot_k[i] * (theta[i] - spr_stress_free[i]) +
                          (2 * (2 * np.pi - theta2) * spr_rot_k[i] / np.pi) *
                          np.tan(np.pi * (theta[i] - theta2) / (4 * np.pi - 2 * theta2)) -
                          spr_rot_k[i] * theta[i] + spr_rot_k[i] * theta2)
            else:
                Cspr[i] = spr_rot_k[i]
                Mspr[i] = spr_rot_k[i] * (theta[i] - spr_stress_free[i])
        
        return Mspr, Cspr
    
    def solve_global_force(self, node, U, M) -> np.ndarray:
        """
        Calculate the global internal force of the spring element.
        
        Parameters:
        -----------
        node : object
            Node object
        U : np.ndarray
            Displacements
        M : np.ndarray
            Moments
            
        Returns:
        --------
        Tspr : np.ndarray
            Global forces
        """
        node_num = U.shape[0]
        Tspr = np.zeros(3 * node_num)
        
        nodal_coordinates = node.coordinates_mat
        spr_ijkl = self.node_ijkl_mat
        
        spr_i = spr_ijkl[:, 0]-1
        spr_j = spr_ijkl[:, 1]-1
        spr_k = spr_ijkl[:, 2]-1
        spr_l = spr_ijkl[:, 3]-1
    
        node_i = nodal_coordinates[spr_i, :] + U[spr_i, :]
        node_j = nodal_coordinates[spr_j, :] + U[spr_j, :]
        node_k = nodal_coordinates[spr_k, :] + U[spr_k, :]
        node_l = nodal_coordinates[spr_l, :] + U[spr_l, :]
        
        rij = node_i - node_j
        rkj = node_k - node_j
        rkl = node_k - node_l
        
        m = np.cross(rij, rkj, axis=1)
        n = np.cross(rkj, rkl, axis=1)
        
        m_square = np.sum(m * m, axis=1)
        n_square = np.sum(n * n, axis=1)
        
        rkj_square = np.sum(rkj * rkj, axis=1)
        rkj_norm = np.sqrt(rkj_square)
        
        parti = (rkj_norm[:, np.newaxis] / m_square[:, np.newaxis]) * m
        partl = -(rkj_norm[:, np.newaxis] / n_square[:, np.newaxis]) * n
        partj = ((np.sum(rij * rkj, axis=1) / rkj_norm / rkj_norm - 1)[:, np.newaxis] * parti -
                 (np.sum(rkl * rkj, axis=1) / rkj_norm / rkj_norm)[:, np.newaxis] * partl)
        partk = ((np.sum(rkl * rkj, axis=1) / rkj_norm / rkj_norm - 1)[:, np.newaxis] * partl -
                 (np.sum(rij * rkj, axis=1) / rkj_norm / rkj_norm)[:, np.newaxis] * parti)

       
        gradient = np.hstack([parti, partj, partk, partl])
        
        M_temp = M
        if len(M_temp) > len(spr_i):
            M_temp = M_temp[:len(spr_i)]
        
        local_t = M_temp[:, np.newaxis] * gradient
        
        index1 = 3 * (spr_i+1) - 3
        index2 = 3 * (spr_j+1) - 3
        index3 = 3 * (spr_k+1) - 3
        index4 = 3 * (spr_l+1) - 3
        
        index = np.concatenate([
            index1, index1 + 1, index1 + 2,
            index2, index2 + 1, index2 + 2,
            index3, index3 + 1, index3 + 2,
            index4, index4 + 1, index4 + 2
        ])
        
        index = index.flatten()
        local_t = local_t.T.flatten()

        
        for i in range(len(index)):
            Tspr[index[i]] = Tspr[index[i]] + local_t[i]
        
        return Tspr
    
    def solve_global_stiff(obj, node, U, Mspr, Cspr):
        """
        Compute the global stiffness matrix.
        
        Parameters:
        -----------
        obj : object
            Object containing node_ijkl_mat attribute
        node : object
            Object containing coordinates_mat attribute
        U : numpy.ndarray
            Displacement matrix (NodeNum x 3)
        Mspr : numpy.ndarray
            Spring mass array
        Cspr : numpy.ndarray
            Spring stiffness array
        
        Returns:
        --------
        Kspr : numpy.ndarray
            Global stiffness matrix (3*NodeNum x 3*NodeNum)
        """
        NodeNum = U.shape[0]
        Kspr = np.zeros((3*NodeNum, 3*NodeNum),dtype=np.float64)
        
        sprIJKL = obj.node_ijkl_mat
        nodalCoordinates = node.coordinates_mat
        
        spr_i = sprIJKL[:, 0]
        spr_j = sprIJKL[:, 1]
        spr_k = sprIJKL[:, 2]
        spr_l = sprIJKL[:, 3]
        
        # Find non-zero indices
        nonzero = np.nonzero(spr_i)[0]
        spr_Num = len(nonzero)
        
        # Extract non-zero elements
        spr_i = spr_i[nonzero]
        spr_j = spr_j[nonzero]
        spr_k = spr_k[nonzero]
        spr_l = spr_l[nonzero]
        
        # Convert to 0-based indexing for Python
        spr_i = spr_i.astype(int) - 1
        spr_j = spr_j.astype(int) - 1
        spr_k = spr_k.astype(int) - 1
        spr_l = spr_l.astype(int) - 1
        
        # Calculate node positions with displacements
        nodei = nodalCoordinates[spr_i, :] + U[spr_i, :]
        nodej = nodalCoordinates[spr_j, :] + U[spr_j, :]
        nodek = nodalCoordinates[spr_k, :] + U[spr_k, :]
        nodel = nodalCoordinates[spr_l, :] + U[spr_l, :]
        
        # Calculate vectors
        rij = nodei - nodej
        rkj = nodek - nodej
        rkl = nodek - nodel
        
        # Cross products
        m = np.cross(rij, rkj)
        n = np.cross(rkj, rkl)
        
        # Squared magnitudes
        m_square = np.sum(m * m, axis=1)
        n_square = np.sum(n * n, axis=1)
        
        rkj_square = np.sum(rkj * rkj, axis=1)
        rkj_norm = np.sqrt(rkj_square)
        
        # Calculate gradients
        parti = (rkj_norm[:, np.newaxis] / m_square[:, np.newaxis]) * m
        partl = -(rkj_norm[:, np.newaxis] / n_square[:, np.newaxis]) * n
        partj = ((np.sum(rij * rkj, axis=1) / rkj_norm / rkj_norm - 1)[:, np.newaxis] * parti -
                 (np.sum(rkl * rkj, axis=1) / rkj_norm / rkj_norm)[:, np.newaxis] * partl)
        partk = ((np.sum(rkl * rkj, axis=1) / rkj_norm / rkj_norm - 1)[:, np.newaxis] * partl -
                 (np.sum(rij * rkj, axis=1) / rkj_norm / rkj_norm)[:, np.newaxis] * parti)
        
        gradient = np.hstack([parti, partj, partk, partl])

        
        # Calculate A and B coefficients
        A = np.sum(rij * rkj, axis=1) / rkj_norm / rkj_norm
        B = np.sum(rkl * rkj, axis=1) / rkj_norm / rkj_norm
        
        # Calculate partial derivatives
        partAj = 1 / rkj_norm[:, np.newaxis] / rkj_norm[:, np.newaxis] * ((2 * A[:, np.newaxis] - 1) * rkj - rij)
        partBj = 1 / rkj_norm[:, np.newaxis] / rkj_norm[:, np.newaxis] * (2 * B[:, np.newaxis] * rkj - rkl)
        partAk = 1 / rkj_norm[:, np.newaxis] / rkj_norm[:, np.newaxis] * (-2 * A[:, np.newaxis] * rkj + rij)
        partBk = 1 / rkj_norm[:, np.newaxis] / rkj_norm[:, np.newaxis] * ((1 - 2 * B[:, np.newaxis]) * rkj + rkl)
        
        m1, m2, m3 = m[:, 0], m[:, 1], m[:, 2]
        n1, n2, n3 = n[:, 0], n[:, 1], n[:, 2]
        
        # Calculate second derivatives (Hessian components)
        cross_rkj_m = np.cross(rkj, m)
        part2ii_temp_1 = m1[:, np.newaxis] * np.cross(rkj, m) + cross_rkj_m[:, 0:1] * m
        part2ii_temp_2 = m2[:, np.newaxis] * np.cross(rkj, m) + cross_rkj_m[:, 1:2] * m
        part2ii_temp_3 = m3[:, np.newaxis] * np.cross(rkj, m) + cross_rkj_m[:, 2:3] * m
        
        part2ii_temp = np.hstack([part2ii_temp_1, part2ii_temp_2, part2ii_temp_3])
        part2ii_temp = -(rkj_norm[:, np.newaxis] / (m_square * m_square)[:, np.newaxis]) * part2ii_temp
        
        cross_rkj_n = np.cross(rkj, n)
        part2ll_temp_1 = n1[:, np.newaxis] * np.cross(rkj, n) + cross_rkj_n[:, 0:1] * n
        part2ll_temp_2 = n2[:, np.newaxis] * np.cross(rkj, n) + cross_rkj_n[:, 1:2] * n
        part2ll_temp_3 = n3[:, np.newaxis] * np.cross(rkj, n) + cross_rkj_n[:, 2:3] * n
        
        part2ll_temp = np.hstack([part2ll_temp_1, part2ll_temp_2, part2ll_temp_3])
        part2ll_temp = (rkj_norm[:, np.newaxis] / (n_square * n_square)[:, np.newaxis]) * part2ll_temp
        
        # Continue with other second derivative calculations
        cross_rij_m = np.cross(rij, m)
        part2ik_temp_1 = (m1[:, np.newaxis] * rkj / m_square[:, np.newaxis] / rkj_norm[:, np.newaxis] +
                          (rkj_norm / m_square / m_square)[:, np.newaxis] * 
                          (m1[:, np.newaxis] * cross_rij_m + cross_rij_m[:, 0:1] * m))
        part2ik_temp_2 = (m2[:, np.newaxis] * rkj / m_square[:, np.newaxis] / rkj_norm[:, np.newaxis] +
                          (rkj_norm / m_square / m_square)[:, np.newaxis] * 
                          (m2[:, np.newaxis] * cross_rij_m + cross_rij_m[:, 1:2] * m))
        part2ik_temp_3 = (m3[:, np.newaxis] * rkj / m_square[:, np.newaxis] / rkj_norm[:, np.newaxis] +
                          (rkj_norm / m_square / m_square)[:, np.newaxis] * 
                          (m3[:, np.newaxis] * cross_rij_m + cross_rij_m[:, 2:3] * m))
        
        part2ik_temp = np.hstack([part2ik_temp_1, part2ik_temp_2, part2ik_temp_3])
        
        cross_rkl_n = np.cross(rkl, n)
        part2lj_temp_1 = (n1[:, np.newaxis] * rkj / n_square[:, np.newaxis] / rkj_norm[:, np.newaxis] -
                          (rkj_norm / n_square / n_square)[:, np.newaxis] * 
                          (n1[:, np.newaxis] * cross_rkl_n + cross_rkl_n[:, 0:1] * n))
        part2lj_temp_2 = (n2[:, np.newaxis] * rkj / n_square[:, np.newaxis] / rkj_norm[:, np.newaxis] -
                          (rkj_norm / n_square / n_square)[:, np.newaxis] * 
                          (n2[:, np.newaxis] * cross_rkl_n + cross_rkl_n[:, 1:2] * n))
        part2lj_temp_3 = (n3[:, np.newaxis] * rkj / n_square[:, np.newaxis] / rkj_norm[:, np.newaxis] -
                          (rkj_norm / n_square / n_square)[:, np.newaxis] * 
                          (n3[:, np.newaxis] * cross_rkl_n + cross_rkl_n[:, 2:3] * n))
        
        part2lj_temp = np.hstack([part2lj_temp_1, part2lj_temp_2, part2lj_temp_3])
        
        cross_rkj_rij_m = np.cross(rkj - rij, m)
        part2ij_temp_1 = (-m1[:, np.newaxis] * rkj / m_square[:, np.newaxis] / rkj_norm[:, np.newaxis] +
                          (rkj_norm / m_square / m_square)[:, np.newaxis] * 
                          (m1[:, np.newaxis] * cross_rkj_rij_m + cross_rkj_rij_m[:, 0:1] * m))
        part2ij_temp_2 = (-m2[:, np.newaxis] * rkj / m_square[:, np.newaxis] / rkj_norm[:, np.newaxis] +
                          (rkj_norm / m_square / m_square)[:, np.newaxis] * 
                          (m2[:, np.newaxis] * cross_rkj_rij_m + cross_rkj_rij_m[:, 1:2] * m))
        part2ij_temp_3 = (-m3[:, np.newaxis] * rkj / m_square[:, np.newaxis] / rkj_norm[:, np.newaxis] +
                          (rkj_norm / m_square / m_square)[:, np.newaxis] * 
                          (m3[:, np.newaxis] * cross_rkj_rij_m + cross_rkj_rij_m[:, 2:3] * m))
        
        part2ij_temp = np.hstack([part2ij_temp_1, part2ij_temp_2, part2ij_temp_3])
        
        cross_rkj_rkl_n = np.cross(rkj - rkl, n)
        part2lk_temp_1 = (-n1[:, np.newaxis] * rkj / n_square[:, np.newaxis] / rkj_norm[:, np.newaxis] -
                          (rkj_norm / n_square / n_square)[:, np.newaxis] * 
                          (n1[:, np.newaxis] * cross_rkj_rkl_n + cross_rkj_rkl_n[:, 0:1] * n))
        part2lk_temp_2 = (-n2[:, np.newaxis] * rkj / n_square[:, np.newaxis] / rkj_norm[:, np.newaxis] -
                          (rkj_norm / n_square / n_square)[:, np.newaxis] * 
                          (n2[:, np.newaxis] * cross_rkj_rkl_n + cross_rkj_rkl_n[:, 1:2] * n))
        part2lk_temp_3 = (-n3[:, np.newaxis] * rkj / n_square[:, np.newaxis] / rkj_norm[:, np.newaxis] -
                          (rkj_norm / n_square / n_square)[:, np.newaxis] * 
                          (n3[:, np.newaxis] * cross_rkj_rkl_n + cross_rkj_rkl_n[:, 2:3] * n))
        
        part2lk_temp = np.hstack([part2lk_temp_1, part2lk_temp_2, part2lk_temp_3])
        
        parti_1, parti_2, parti_3 = parti[:, 0], parti[:, 1], parti[:, 2]
        partl_1, partl_2, partl_3 = partl[:, 0], partl[:, 1], partl[:, 2]
        
        part2jj_temp_1 = parti_1[:, np.newaxis] * partAj - partl_1[:, np.newaxis] * partBj
        part2jj_temp_2 = parti_2[:, np.newaxis] * partAj - partl_2[:, np.newaxis] * partBj
        part2jj_temp_3 = parti_3[:, np.newaxis] * partAj - partl_3[:, np.newaxis] * partBj
        
        part2jj_temp = np.hstack([part2jj_temp_1, part2jj_temp_2, part2jj_temp_3])
        part2jj_temp = part2jj_temp + (A[:, np.newaxis] - 1) * part2ij_temp - B[:, np.newaxis] * part2lj_temp
        
        part2jk_temp_1 = parti_1[:, np.newaxis] * partAk - partl_1[:, np.newaxis] * partBk
        part2jk_temp_2 = parti_2[:, np.newaxis] * partAk - partl_2[:, np.newaxis] * partBk
        part2jk_temp_3 = parti_3[:, np.newaxis] * partAk - partl_3[:, np.newaxis] * partBk
        
        part2jk_temp = np.hstack([part2jk_temp_1, part2jk_temp_2, part2jk_temp_3])
        part2jk_temp = part2jk_temp + (A[:, np.newaxis] - 1) * part2ik_temp - B[:, np.newaxis] * part2lk_temp
        
        part2kk_temp_1 = partl_1[:, np.newaxis] * partBk - parti_1[:, np.newaxis] * partAk
        part2kk_temp_2 = partl_2[:, np.newaxis] * partBk - parti_2[:, np.newaxis] * partAk
        part2kk_temp_3 = partl_3[:, np.newaxis] * partBk - parti_3[:, np.newaxis] * partAk
        
        part2kk_temp = np.hstack([part2kk_temp_1, part2kk_temp_2, part2kk_temp_3])
        part2kk_temp = part2kk_temp + (B[:, np.newaxis] - 1) * part2lk_temp - A[:, np.newaxis] * part2ik_temp
        
        part2li_temp = np.zeros_like(part2ii_temp)
        
        # Transpose operations for symmetric components
        part2ki_temp = part2ik_temp[:, [0, 3, 6, 1, 4, 7, 2, 5, 8]]
        part2jl_temp = part2lj_temp[:, [0, 3, 6, 1, 4, 7, 2, 5, 8]]
        part2ji_temp = part2ij_temp[:, [0, 3, 6, 1, 4, 7, 2, 5, 8]]
        part2kl_temp = part2lk_temp[:, [0, 3, 6, 1, 4, 7, 2, 5, 8]]
        part2kj_temp = part2jk_temp[:, [0, 3, 6, 1, 4, 7, 2, 5, 8]]
        part2il_temp = part2li_temp[:, [0, 3, 6, 1, 4, 7, 2, 5, 8]]
        
        # Get spring properties
        sprKadj_temp = Cspr[nonzero]
        M_temp = Mspr[nonzero]
        
        np.set_printoptions(precision=4,suppress=True)
        localK_part1=np.zeros((spr_Num,144),dtype=np.float64)
        
        # Calculate local stiffness matrix part 1
        for i in range(spr_Num):  
            localK_row=np.zeros(144,dtype=np.float64)
            for j in range(12):
                localK_row[j*12:(j+1)*12]=sprKadj_temp[i] * gradient[i, j] * gradient[i,:]
            localK_part1[i,:]=localK_row
                   
       
        # Reshape Hessian components       
        part2ii_temp = part2ii_temp.reshape((spr_Num, 3, 3),order='F')
        part2ij_temp = part2ij_temp.reshape((spr_Num, 3, 3),order='F')
        part2ji_temp = part2ji_temp.reshape((spr_Num, 3, 3),order='F')
        part2ik_temp = part2ik_temp.reshape((spr_Num, 3, 3),order='F')
        part2ki_temp = part2ki_temp.reshape((spr_Num, 3, 3),order='F')
        part2il_temp = part2il_temp.reshape((spr_Num, 3, 3),order='F')
        
        part2li_temp = part2li_temp.reshape((spr_Num, 3, 3),order='F')
        part2jj_temp = part2jj_temp.reshape((spr_Num, 3, 3),order='F')
        part2jk_temp = part2jk_temp.reshape((spr_Num, 3, 3),order='F')
        part2kj_temp = part2kj_temp.reshape((spr_Num, 3, 3),order='F')
        part2jl_temp = part2jl_temp.reshape((spr_Num, 3, 3),order='F')
        part2lj_temp = part2lj_temp.reshape((spr_Num, 3, 3),order='F')
        
        part2kk_temp = part2kk_temp.reshape((spr_Num, 3, 3),order='F')
        part2kl_temp = part2kl_temp.reshape((spr_Num, 3, 3),order='F')
        part2lk_temp = part2lk_temp.reshape((spr_Num, 3, 3),order='F')
        part2ll_temp = part2ll_temp.reshape((spr_Num, 3, 3),order='F')
        
        # Assemble Hessian
        hessian_1 = np.concatenate([part2ii_temp, part2ij_temp, part2ik_temp, part2il_temp], axis=1)
        hessian_2 = np.concatenate([part2ji_temp, part2jj_temp, part2jk_temp, part2jl_temp], axis=1)
        hessian_3 = np.concatenate([part2ki_temp, part2kj_temp, part2kk_temp, part2kl_temp], axis=1)
        hessian_4 = np.concatenate([part2li_temp, part2lj_temp, part2lk_temp, part2ll_temp], axis=1)
                
        hessian = np.stack([hessian_1, hessian_2, hessian_3, hessian_4], axis=2)
        
        # Calculate local stiffness matrix part 2
        localK_part2=np.zeros((spr_Num,144),dtype=np.float64)
        hessianReshap=np.reshape(hessian,(spr_Num,144))
        for i in range(spr_Num):
            localK_part2[i,:] = M_temp[i] * hessianReshap[i,:]

        
        # Total local stiffness
        localK_n = localK_part1 + localK_part2
        
                # Assembly into global stiffness matrix
        index1 = 3 * (spr_i)
        index2 = 3 * (spr_j)
        index3 = 3 * (spr_k)
        index4 = 3 * (spr_l)
        
        index_dim = np.column_stack([
            index1, index1+1, index1+2,
            index2, index2+1, index2+2,
            index3, index3+1, index3+2,
            index4, index4+1, index4+2
        ])
        
        #print(index_dim)
       
        index=np.zeros((spr_Num,12,12,2),dtype=np.int64)
        
        index[:,0,:,1]=index_dim
        index[:,1,:,1]=index_dim
        index[:,2,:,1]=index_dim
        index[:,3,:,1]=index_dim
        index[:,4,:,1]=index_dim
        index[:,5,:,1]=index_dim
        index[:,6,:,1]=index_dim
        index[:,7,:,1]=index_dim
        index[:,8,:,1]=index_dim
        index[:,9,:,1]=index_dim
        index[:,10,:,1]=index_dim
        index[:,11,:,1]=index_dim
        
        index[:,:,0,0]=index_dim
        index[:,:,1,0]=index_dim
        index[:,:,2,0]=index_dim
        index[:,:,3,0]=index_dim
        index[:,:,4,0]=index_dim
        index[:,:,5,0]=index_dim
        index[:,:,6,0]=index_dim
        index[:,:,7,0]=index_dim
        index[:,:,8,0]=index_dim
        index[:,:,9,0]=index_dim
        index[:,:,10,0]=index_dim
        index[:,:,11,0]=index_dim
        
        index=np.reshape(index,(spr_Num*144,2),order='F')        
        localK_n=np.reshape(localK_n,spr_Num*144,order='F')

        
        for i in range(spr_Num*144):
            Kspr[index[i,0],index[i,1]] += localK_n[i]           

        return Kspr
    
    def Solve_FK(self, node, U) -> Tuple[np.ndarray, np.ndarray]:
        """
        This function is the main function we use to interact with the
        solver. We use this function to compute the global forces and 
        stiffness of the bar elements.
        
        Parameters:
        -----------
        node : object
            Node object
        U : np.ndarray
            Displacements
            
        Returns:
        --------
        Tspr : np.ndarray
            Global forces
        Kspr : np.ndarray
            Global stiffness matrix
        """
        theta = self.solve_theta(node, U)
        Mspr, Cspr = self.solve_moment(theta)
        
        Tspr = self.solve_global_force(node, U, Mspr)
        Kspr = self.solve_global_stiff(node, U, Mspr, Cspr)
        
        self.theta_current_vec = theta
        self.energy_current_vec = 0.5 * self.rot_spr_K_vec * \
                                  (theta - self.theta_stress_free_vec) * \
                                  (theta - self.theta_stress_free_vec)
        
        return Tspr, Kspr

