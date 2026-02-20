import numpy as np


class Std_Elements_Bars:
    """
    Python equivalent of MATLAB @Std_Elements_Bars

    Notes:
    - node_ij_mat is assumed to be 1-based indexing (MATLAB style).
      Internally we convert to 0-based for numpy.
    - node.coordinates_mat: (NodeNum, 3)
    - U: (NodeNum, 3)
    - Returns:
        Tbar: (3*NodeNum,)  (flat vector)
        Kbar: (3*NodeNum, 3*NodeNum)
    """

    def __init__(self):
        # Connection information (Nb, 2), 1-based IDs
        self.node_ij_mat = np.zeros((0, 2), dtype=int)

        # Area (Nb,)
        self.A_vec = np.zeros((0,), dtype=float)

        # Young's modulus (Nb,)
        self.E_vec = np.zeros((0,), dtype=float)

        # Stress-free length (Nb,)
        self.L0_vec = np.zeros((0,), dtype=float)

        # step size (kept for compatibility; not used in closed-form formulas)
        self.delta = 1e-8

    # ---------------------------
    # Initialize
    # ---------------------------
    def Initialize(self, node):
        """
        MATLAB:
            obj.L0_vec=zeros(size(obj.A_vec));
            for i=1:length(obj.A_vec)
                node1=node.coordinates_mat(obj.node_ij_mat(i,1),:);
                node2=node.coordinates_mat(obj.node_ij_mat(i,2),:);
                obj.L0_vec(i)=norm(node1-node2);
            end
        """
        if self.node_ij_mat.size == 0:
            self.L0_vec = np.zeros_like(self.A_vec, dtype=float)
            return

        coords = np.asarray(node.coordinates_mat, dtype=float)
        bar_num = len(self.A_vec)
        self.L0_vec = np.zeros((bar_num,), dtype=float)

        for i in range(bar_num):
            n1 = int(self.node_ij_mat[i, 0]) - 1  # 0-based
            n2 = int(self.node_ij_mat[i, 1]) - 1
            x1 = coords[n1, :]
            x2 = coords[n2, :]
            self.L0_vec[i] = np.linalg.norm(x1 - x2)

    # ---------------------------
    # Main interface
    # ---------------------------
    def Solve_FK(self, node, U):
        """
        MATLAB:
            [Tbar]=Solve_Global_Force(obj,node,U);
            [Kbar]=Solve_Global_Stiff(obj,node,U);
        """
        Tbar = self.Solve_Global_Force(node, U)
        Kbar = self.Solve_Global_Stiff(node, U)
        return Tbar, Kbar

    # ---------------------------
    # Axial strain (per bar)
    # ---------------------------
    def Solve_Strain(self, node, U):
        """
        Compute axial strain for each bar:
            strain = (l - L0) / L0
        Returns:
            strain_vec: (Nb,)
        """
        coords = np.asarray(node.coordinates_mat, dtype=float)
        U = np.asarray(U, dtype=float)

        bar_num = len(self.A_vec)
        strain_vec = np.zeros((bar_num,), dtype=float)

        for i in range(bar_num):
            node1 = int(self.node_ij_mat[i, 0]) - 1
            node2 = int(self.node_ij_mat[i, 1]) - 1

            x1 = coords[node1, :] + U[node1, :]
            x2 = coords[node2, :] + U[node2, :]

            l = np.linalg.norm(x1 - x2)
            L0 = float(self.L0_vec[i]) if i < len(self.L0_vec) else l
            if L0 <= 0.0:
                strain_vec[i] = 0.0
            else:
                strain_vec[i] = (l - L0) / L0

        return strain_vec

    # ---------------------------
    # Axial stress (per bar)
    # ---------------------------
    def Solve_Stress(self, node, U):
        """
        Compute axial stress for each bar:
            stress = E * strain
        Returns:
            stress_vec: (Nb,)
        """
        strain_vec = self.Solve_Strain(node, U)
        return strain_vec * self.E_vec

    # ---------------------------
    # Global force assembly
    # ---------------------------
    def Solve_Global_Force(self, node, U):
        """
        MATLAB Solve_Global_Force:
            for each bar:
                x1 = coords(node1,:) + U(node1,:)
                x2 = coords(node2,:) + U(node2,:)
                Flocal = Solve_Local_Force(...)
                assemble into Tbar
        """
        coords = np.asarray(node.coordinates_mat, dtype=float)
        U = np.asarray(U, dtype=float)

        node_num = coords.shape[0]
        Tbar = np.zeros((3 * node_num,), dtype=float)

        bar_num = len(self.A_vec)
        for i in range(bar_num):
            node1 = int(self.node_ij_mat[i, 0]) - 1
            node2 = int(self.node_ij_mat[i, 1]) - 1

            x1 = coords[node1, :] + U[node1, :]
            x2 = coords[node2, :] + U[node2, :]

            Flocal = self.Solve_Local_Force(
                x1=x1,
                x2=x2,
                L0=float(self.L0_vec[i]),
                E=float(self.E_vec[i]),
                A=float(self.A_vec[i]),
            )  # (6,)

            # Assemble: node1 contributes Flocal[0:3], node2 contributes Flocal[3:6]
            i1 = slice(3 * node1, 3 * node1 + 3)
            i2 = slice(3 * node2, 3 * node2 + 3)

            Tbar[i1] += Flocal[0:3]
            Tbar[i2] += Flocal[3:6]

        return Tbar

    # ---------------------------
    # Global stiffness assembly
    # ---------------------------
    def Solve_Global_Stiff(self, node, U):
        """
        MATLAB Solve_Global_Stiff:
            assemble Klocal blocks into global Kbar
        """
        coords = np.asarray(node.coordinates_mat, dtype=float)
        U = np.asarray(U, dtype=float)

        node_num = coords.shape[0]
        Kbar = np.zeros((3 * node_num, 3 * node_num), dtype=float)

        bar_num = len(self.A_vec)
        for i in range(bar_num):
            node1 = int(self.node_ij_mat[i, 0]) - 1
            node2 = int(self.node_ij_mat[i, 1]) - 1

            x1 = coords[node1, :] + U[node1, :]
            x2 = coords[node2, :] + U[node2, :]

            Klocal = self.Solve_Local_Stiff(
                x1=x1,
                x2=x2,
                L0=float(self.L0_vec[i]),
                E=float(self.E_vec[i]),
                A=float(self.A_vec[i]),
            )  # (6,6)

            i1 = slice(3 * node1, 3 * node1 + 3)
            i2 = slice(3 * node2, 3 * node2 + 3)

            # Block assembly same as MATLAB:
            # [ (1:3,1:3) (1:3,4:6)
            #   (4:6,1:3) (4:6,4:6) ]
            Kbar[i1, i1] += Klocal[0:3, 0:3]
            Kbar[i1, i2] += Klocal[0:3, 3:6]
            Kbar[i2, i1] += Klocal[3:6, 0:3]
            Kbar[i2, i2] += Klocal[3:6, 3:6]

        return Kbar

    # ---------------------------
    # Local force (closed-form)
    # ---------------------------
    def Solve_Local_Force(self, x1, x2, L0, E, A):
        """
        MATLAB Solve_Local_Force:
            l = norm(x1-x2)
            Flocal = E*A/L0*(l-L0)/l * [x1-x2, x2-x1]'
        Returns a (6,) vector.
        """
        x1 = np.asarray(x1, dtype=float).reshape(3,)
        x2 = np.asarray(x2, dtype=float).reshape(3,)

        dx = x1 - x2
        l = np.linalg.norm(dx)

        # Guard against zero length (should not happen if geometry is valid)
        if l <= 0.0:
            return np.zeros((6,), dtype=float)

        coef = (E * A / L0) * ((l - L0) / l)
        f1 = coef * dx          # force on node1 (3,)
        f2 = -f1                # force on node2 (3,)
        Flocal = np.hstack([f1, f2])  # (6,)
        return Flocal

    # ---------------------------
    # Local stiffness (closed-form)
    # ---------------------------
    def Solve_Local_Stiff(self, x1, x2, L0, E, A):
        """
        MATLAB Solve_Local_Stiff:
            l = norm(x1-x2)
            K1 = E*A/l*(1/l/l) * v^T v   (outer product)
            K2 = E*A/l*(l-L0)/L0 * [ I -I; -I I ]
            Klocal = K1 + K2
        where v = [x1-x2, x2-x1] (1x6)
        Returns a (6,6) matrix.
        """
        x1 = np.asarray(x1, dtype=float).reshape(3,)
        x2 = np.asarray(x2, dtype=float).reshape(3,)

        dx = x1 - x2
        l = np.linalg.norm(dx)

        if l <= 0.0:
            return np.zeros((6, 6), dtype=float)

        v = np.hstack([dx, -dx])  # (6,)

        # K1 = E*A/l * (1/l^2) * (v^T v) = E*A/(l^3) * outer(v,v)
        K1 = (E * A / (l**3)) * np.outer(v, v)

        I3 = np.eye(3)
        Kgeo = np.block([[I3, -I3],
                         [-I3, I3]])  # (6,6)

        # K2 = E*A/l * (l-L0)/L0 * Kgeo
        K2 = (E * A / l) * ((l - L0) / L0) * Kgeo

        return K1 + K2
