class Assembly_Rolling_Bridge:
    """Python port of MATLAB @Assembly_Rolling_Bridge for the N=8 rolling bridge."""

    def __init__(self):
        self.node = None
        self.bar = None
        self.actBar = None
        self.rot_spr_4N = None
        self.cst = None

    @property
    def rotSpr(self):
        return self.rot_spr_4N

    @rotSpr.setter
    def rotSpr(self, value):
        self.rot_spr_4N = value

    def Initialize_Assembly(self):
        self.node.current_U_mat = self.node.coordinates_mat * 0.0
        self.node.current_ext_force_mat = self.node.coordinates_mat * 0.0

        if self.bar is not None:
            self.bar.Initialize(self.node)
        if self.actBar is not None:
            self.actBar.Initialize(self.node)
        if self.rot_spr_4N is not None:
            self.rot_spr_4N.Initialize(self.node)
        if self.cst is not None:
            self.cst.Initialize(self.node)

    def Solve_FK(self, U):
        T = 0.0
        K = 0.0

        if self.bar is not None:
            Tbar, Kbar = self.bar.Solve_FK(self.node, U)
            T = T + Tbar
            K = K + Kbar

        if self.actBar is not None:
            Tabar, Kabar = self.actBar.Solve_FK(self.node, U)
            T = T + Tabar
            K = K + Kabar

        if self.rot_spr_4N is not None:
            Tspr, Kspr = self.rot_spr_4N.Solve_FK(self.node, U)
            T = T + Tspr
            K = K + Kspr

        if self.cst is not None:
            Tcst, Kcst = self.cst.Solve_FK(self.node, U)
            T = T + Tcst
            K = K + Kcst

        # MATLAB @Assembly_Rolling_Bridge/Solve_FK.m adds rot_spr_4N a
        # second time as Trs/Krs. Keep that behavior for one-to-one parity.
        if self.rot_spr_4N is not None:
            Trs, Krs = self.rot_spr_4N.Solve_FK(self.node, U)
            T = T + Trs
            K = K + Krs

        return T, K
