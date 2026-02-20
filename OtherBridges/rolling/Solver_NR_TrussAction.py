import numpy as np


class Solver_NR_TrussAction:
    """Newton-Raphson implicit solver for truss self-actuation."""

    def __init__(self):
        self.assembly = None
        self.supp = None
        self.increStep = 50
        self.tol = 1e-5
        self.iterMax = 30
        self.targetL0 = None
        self.Uhis = None

    @staticmethod
    def _mod_k_for_supp(K, supp, Tinput):
        Kwsupp = K.copy()
        T = Tinput.copy()

        supp = np.asarray(supp, dtype=float)
        n = K.shape[0]
        for i in range(supp.shape[0]):
            temp_node = int(supp[i, 0])

            if supp[i, 1] == 1:
                idx = temp_node * 3 + 0
                Kvv = K[idx, idx]
                Kwsupp[idx, :] = 0.0
                Kwsupp[:, idx] = 0.0
                Kwsupp[idx, idx] = 1.0 if abs(Kvv) < 100 else Kvv
                T[idx] = 0.0

            if supp[i, 2] == 1:
                idx = temp_node * 3 + 1
                Kvv = K[idx, idx]
                Kwsupp[idx, :] = 0.0
                Kwsupp[:, idx] = 0.0
                Kwsupp[idx, idx] = 1.0 if abs(Kvv) < 100 else Kvv
                T[idx] = 0.0

            if supp[i, 3] == 1:
                idx = temp_node * 3 + 2
                Kvv = K[idx, idx]
                Kwsupp[idx, :] = 0.0
                Kwsupp[:, idx] = 0.0
                Kwsupp[idx, idx] = 1.0 if abs(Kvv) < 100 else Kvv
                T[idx] = 0.0

        return Kwsupp, T

    def Solve(self):
        incre_step = self.increStep
        tol = self.tol
        iter_max = self.iterMax
        supp = self.supp

        assembly = self.assembly
        U = assembly.node.current_U_mat.copy()

        node_num = U.shape[0]
        self.Uhis = np.zeros((incre_step, node_num, 3), dtype=float)

        current_applied_force = np.zeros(3 * node_num, dtype=float)
        for i in range(node_num):
            current_applied_force[3 * i:3 * (i + 1)] = assembly.node.current_ext_force_mat[i, :]

        print("Self Assemble Analysis Start")

        L0_before = assembly.actBar.L0_vec.copy()
        L0_after = np.array(self.targetL0, dtype=float).copy()

        for i in range(incre_step):
            alpha = (i + 1) / incre_step
            L0_current = alpha * L0_after + (1.0 - alpha) * L0_before

            print(f"Increment = {i + 1}")
            step = 1
            R = 1.0

            while step < iter_max and R > tol:
                assembly.actBar.L0_vec = L0_current
                T, K = assembly.Solve_FK(U)

                unbalance = current_applied_force - T
                K, unbalance = self._mod_k_for_supp(K, supp, unbalance)

                try:
                    dU = np.linalg.solve(K, unbalance)
                except np.linalg.LinAlgError:
                    dU = np.linalg.lstsq(K, unbalance, rcond=None)[0]

                for j in range(node_num):
                    U[j, :] += dU[3 * j:3 * (j + 1)]

                R = float(np.linalg.norm(unbalance))
                print(f"\tIteration = {step}, R = {R:e}")
                step += 1

            self.Uhis[i, :, :] = U

        return self.Uhis
