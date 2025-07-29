# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 13:42:18 2025

@author: zzhuy
"""

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve

class Solver_NR_Loading:
    """Newton-Raphson Implicit Solver for structural analysis with incremental loading"""
    
    def __init__(self):
        # Assembly of the structure
        self.assembly = None
        
        # Storing the support information
        self.supp = None
        
        # The applied load
        # Example format:
        # load_force = 3
        # load = np.array([[29, 0, 0, load_force],
        #                  [30, 0, 0, load_force],
        #                  [31, 0, 0, load_force],
        #                  [32, 0, 0, load_force]])
        self.load = None
        
        # The total number of incremental steps
        self.incre_step = 50
        
        # The tolerance for each iteration
        self.tol = 1e-5
        
        # The maximum allowed iteration number
        self.iter_max = 30
        
        # The history of displacement field
        self.u_his = None
    
    def Solve(self):
        """
        Newton-Raphson solver for loading
        
        Returns:
            u_his: History of displacements for each increment step
        """
        # Initialize and set up storage matrix for u_his
        incre_step = self.incre_step
        tol = self.tol
        iter_max = self.iter_max
        
        supp = self.supp
        load = self.load
        assembly = self.assembly
        
        # Get current displacement matrix
        U = assembly.node.current_U_mat.copy()
        
        # Get dimensions
        node_num = U.shape[0]
        self.u_his = np.zeros((incre_step, node_num, 3),dtype=np.float64)
        
        print('Loading Analysis Start')
        
        # Find the external forces that are currently applied on the structure
        current_applied_force = np.zeros(3 * node_num,np.float64)
        for i in range(node_num):
            current_applied_force[3*i:3*(i+1)] = assembly.node.current_ext_force_mat[i, :]
        
        # Assemble the load vector
        load_size = load.shape[0]
        load_vec = np.zeros(3 * node_num,dtype=np.float64)
        
        for i in range(load_size):
            temp_node_num = int(load[i, 0])
            print(temp_node_num)
            load_vec[temp_node_num*3 ] = load[i, 1]  # X component
            load_vec[temp_node_num*3 + 1] = load[i, 2]  # Y component  
            load_vec[temp_node_num*3 + 2] = load[i, 3]  # Z component
        
        # Main incremental loading loop
        for i in range(incre_step):
            step = 1
            R = 1
            lambda_factor = i + 1  # Load factor (MATLAB uses 1-based indexing)
            print(f'Increment = {i+1}')
            
            # Newton-Raphson iteration loop
            while step < iter_max and R > tol:
                # Find the internal force and stiffness of system
                T, K = assembly.Solve_FK(U)
                
                # Calculate the unbalanced force
                unbalance = current_applied_force + lambda_factor * load_vec - T
                
                print(unbalance)
                
                # Apply boundary conditions (modify K and unbalance for supports)
                K, unbalance = self.mod_k_for_supp(K, supp, unbalance)

                # Solve for displacement
                dU_temp = np.linalg.solve(K, unbalance)
                
                # Update displacements
                for j in range(node_num):
                    U[j, :] += dU_temp[3*j:3*(j+1)]
                
                # Calculate residual
                R = np.linalg.norm(dU_temp)
                print(f'    Iteration = {step}, R = {R:.6e}')
                step += 1
            
            # Store displacement history
            self.u_his[i, :, :] = U
        
        # Update the assembly with final displacements
        assembly.node.current_U_mat = U
        
        return self.u_his
    
    def mod_k_for_supp(self, K, supp, unbalance):
        K_mod = np.array(K, dtype=float)
        unbalance_mod = np.array(unbalance, dtype=float)
        for s in supp:
            node_id = s[0] 
            for dim in range(3):
                if s[dim+1] == 1:
                    idx = 3*node_id + dim
                    Kvv=max(K_mod[idx,idx],100);
                    K_mod[idx, :] = 0
                    K_mod[:, idx] = 0
                    K_mod[idx, idx] = Kvv
                    unbalance_mod[idx] = 0
        return K_mod, unbalance_mod

