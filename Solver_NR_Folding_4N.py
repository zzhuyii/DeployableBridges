import numpy as np

class Solver_NR_Folding_4N:
    def __init__(self):
        self.assembly = None
        self.supp = []
        self.increStep = 50
        self.tol = 1e-5
        self.iterMax = 30
        self.targetRot = []
        self.Uhis = None

    def Solve(self):
        increStep = self.increStep
        tol = self.tol
        iterMax = self.iterMax
        supp = self.supp
        assembly = self.assembly

        U = np.copy(assembly.node.current_U_mat)
        NodeNum = U.shape[0]
        Uhis = np.zeros((increStep, NodeNum, 3))

        currentAppliedForce = assembly.node.current_ext_force_mat.reshape(-1)

        print('Self Assemble Analysis Start')

        sprZeroStrain_before = assembly.rotSpr.theta_current_vec
        sprZeroStrain_after = np.array(self.targetRot)

        for i in range(increStep):
            sprZeroStrain_current = (i+1)/increStep * sprZeroStrain_after + \
                                     (1 - (i+1)/increStep) * sprZeroStrain_before
            print(f"Increment = {i+1}")
            step = 1
            R = 1

            while step < iterMax and R > tol:
                assembly.rotSpr.theta_stress_free_vec = sprZeroStrain_current
                T, K = assembly.Solve_FK(U)

                unbalance = currentAppliedForce - T
                K_mod, unbalance_mod = self.mod_k_for_supp(K, supp, unbalance)
                
                deltaU = np.linalg.solve(K_mod, unbalance_mod) 

                for j in range(NodeNum):
                    U[j, :] += deltaU[3*j:3*(j+1)]

                R = np.linalg.norm(unbalance_mod)
                print(f"  Iteration = {step}, R = {R:.2e}")
                step += 1

            Uhis[i, :, :] = U

        assembly.node.current_U_mat = U
        return Uhis

    def mod_k_for_supp(self, K, supp, unbalance):
        K_mod = np.array(K, dtype=float)
        unbalance_mod = np.array(unbalance, dtype=float)
        for s in supp:
            node_id = s[0] - 1
            for dim in range(3):
                if s[dim+1] == 1:
                    idx = 3*node_id + dim
                    Kvv=max(K_mod[idx,idx],100);
                    K_mod[idx, :] = 0
                    K_mod[:, idx] = 0
                    K_mod[idx, idx] = Kvv
                    unbalance_mod[idx] = 0
        return K_mod, unbalance_mod
