import numpy as np

class Assembly_Scissor_Bridge:
    def __init__(self):
        self.node = None
        self.cst = None
        self.bar = None

        # 4N rotational spring (vector formulation)
        self.rotSpr = None

        # 3N rotational spring (central-difference formulation)
        self.rotSpr3 = None

    def Initialize_Assembly(self):
        # --- state vectors ---
        self.node.current_U_mat = np.zeros_like(self.node.coordinates_mat, dtype=float)
        self.node.current_ext_force_mat = np.zeros_like(self.node.coordinates_mat, dtype=float)

        # --- initialize each element ---
        if self.rotSpr is not None:
            self.rotSpr.Initialize(self.node)

        if self.rotSpr3 is not None:
            self.rotSpr3.Initialize(self.node)

        if self.cst is not None:
            self.cst.Initialize(self.node)

        if self.bar is not None:
            self.bar.Initialize(self.node)

    def Solve_FK(self, U):
        T = 0.0
        K = 0.0

        if self.cst is not None:
            Tcst, Kcst = self.cst.Solve_FK(self.node, U)
            T = T + Tcst
            K = K + Kcst

        if self.rotSpr is not None:
            Trs, Krs = self.rotSpr.Solve_FK(self.node, U)
            T = T + Trs
            K = K + Krs

        if self.rotSpr3 is not None:
            Trs3, Krs3 = self.rotSpr3.Solve_FK(self.node, U)
            T = T + Trs3
            K = K + Krs3

        if self.bar is not None:
            Tb, Kb = self.bar.Solve_FK(self.node, U)
            T = T + Tb
            K = K + Kb

        return T, K
