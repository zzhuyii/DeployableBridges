import numpy as np


class Assembly_Origami:
    def __init__(self):
        self.node = None
        self.rot_spr_4N = None
        self.cst = None
        self.bar = None

    @property
    def rotSpr(self):
        return self.rot_spr_4N

    @rotSpr.setter
    def rotSpr(self, value):
        self.rot_spr_4N = value

    def Initialize_Assembly(self):
        self.node.current_U_mat = np.zeros_like(self.node.coordinates_mat, dtype=float)
        self.node.current_ext_force_mat = np.zeros_like(self.node.coordinates_mat, dtype=float)
        for element in (self.cst, self.bar, self.rot_spr_4N):
            if element is not None:
                element.Initialize(self.node)

    def Solve_FK(self, U):
        node_num = self.node.coordinates_mat.shape[0]
        T = np.zeros(3 * node_num, dtype=float)
        K = np.zeros((3 * node_num, 3 * node_num), dtype=float)

        for element in (self.cst, self.bar, self.rot_spr_4N):
            if element is not None:
                Te, Ke = element.Solve_FK(self.node, U)
                T += Te
                K += Ke

        return T, K
