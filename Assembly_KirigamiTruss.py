import numpy as np

class Assembly_KirigamiTruss:
    def __init__(self):
        self.node = None
        self.rotSpr = None
        self.cst = None
        self.bar = None

    def Initialize_Assembly(self):
        self.node.current_U_mat = np.zeros_like(self.node.coordinates_mat)
        self.node.current_ext_force_mat = np.zeros_like(self.node.coordinates_mat)
        self.rotSpr.Initialize(self.node)
        self.cst.Initialize(self.node)
        self.bar.Initialize(self.node)

    def Solve_FK(self, U):
        Tcst, Kcst = self.cst.Solve_FK(self.node, U)
        Trs, Krs = self.rotSpr.Solve_FK(self.node, U)
        Tb, Kb = self.bar.Solve_FK(self.node, U)
               
        T = Tcst + Trs + Tb
        K = Kcst + Krs + Kb
        return T, K