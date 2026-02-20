import os
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class Plot_Scissor_Bridge:
    """
    Python port of MATLAB Plot_Scissor_Bridge.

    Expected assembly interface:
      assembly.node.coordinates_mat : (N,3)
      assembly.cst.node_ijk_mat     : (Ncst,3)  1-based indices
      assembly.bar.node_ij_mat      : (Nbar,2)  1-based indices
      assembly.rotSpr3.node_ijk_mat : (Ns3,3)   1-based indices   (3N springs)
      assembly.rotSpr.node_ijkl_mat : (Ns4,4)   1-based indices   (4N springs)
    """

    def __init__(self):
        self.assembly = None

        # View control
        self.viewAngle1 = 45
        self.viewAngle2 = 45
        self.displayRange = 1
        self.displayRangeRatio = 0.2

        # Figure settings
        self.width = 800
        self.height = 600
        self.x0 = 0
        self.y0 = 0

        # GIF controls
        self.holdTime = 0.01  # seconds per frame
        self.fileName = "Animation.gif"

        # internal scale factor (so figsize matches pixels roughly)
        self._sizeFactor = 120

    # -------------------------
    # helpers
    # -------------------------
    def _setup_ax(self):
        fig = plt.figure(figsize=(self.width / self._sizeFactor, self.height / self._sizeFactor))
        ax = fig.add_subplot(111, projection="3d")
        ax.view_init(self.viewAngle1, self.viewAngle2)
        ax.set_facecolor("white")
        ax.set_box_aspect([1, 1, 1])
        self._set_axis_limits(ax)
        return fig, ax

    def _set_axis_limits(self, ax):
        Vsize = self.displayRange
        Vratio = self.displayRangeRatio

        if isinstance(Vsize, (list, tuple, np.ndarray)) and len(np.array(Vsize).reshape(-1)) == 6:
            V = np.array(Vsize).reshape(-1)
            ax.set_xlim(V[0], V[1])
            ax.set_ylim(V[2], V[3])
            ax.set_zlim(V[4], V[5])
        else:
            ax.set_xlim(-Vratio * Vsize, Vsize)
            ax.set_ylim(-Vratio * Vsize, Vsize)
            ax.set_zlim(-Vratio * Vsize, Vsize)

    @staticmethod
    def _to0(i):
        """MATLAB 1-based index -> Python 0-based"""
        return int(i) - 1

    def _plot_cst_faces(self, ax, nodes_xyz, facecolor="yellow", alpha=1.0, edgecolor="k", linewidth=1.0):
        cstIJK = self.assembly.cst.node_ijk_mat
        if cstIJK is None:
            return
        cstIJK = np.array(cstIJK)
        if cstIJK.size == 0:
            return
        for tri in cstIJK:
            v = [nodes_xyz[self._to0(n)] for n in tri]
            patch = Poly3DCollection([v], facecolors=facecolor, edgecolors=edgecolor,
                                     linewidths=linewidth, alpha=alpha)
            ax.add_collection3d(patch)

    def _plot_bars(self, ax, nodes_xyz, color="k", linewidth=1.0):
        barConnect = self.assembly.bar.node_ij_mat
        if barConnect is None:
            return
        barConnect = np.array(barConnect)
        if barConnect.size == 0:
            return
        for (n1, n2) in barConnect:
            p1 = nodes_xyz[self._to0(n1)]
            p2 = nodes_xyz[self._to0(n2)]
            ax.plot([p1[0], p2[0]],
                    [p1[1], p2[1]],
                    [p1[2], p2[2]],
                    color=color, linewidth=linewidth)

    # -------------------------
    # MATLAB-equivalent methods
    # -------------------------
    def Plot_Shape_Node_Number(self):
        assembly = self.assembly
        node0 = assembly.node.coordinates_mat

        fig, ax = self._setup_ax()
        self._plot_cst_faces(ax, node0, facecolor="yellow", alpha=1.0)
        self._plot_bars(ax, node0, color="k", linewidth=1.0)

        N = node0.shape[0]
        for i in range(N):
            ax.text(node0[i, 0], node0[i, 1], node0[i, 2], str(i + 1), color="red", fontsize=8)

        plt.show()
        return fig

    def Plot_Shape_CST_Number(self):
        assembly = self.assembly
        node0 = assembly.node.coordinates_mat
        cstIJK = assembly.cst.node_ijk_mat

        fig, ax = self._setup_ax()
        self._plot_cst_faces(ax, node0, facecolor="yellow", alpha=1.0)

        cstIJK = np.array(cstIJK)
        for i, tri in enumerate(cstIJK):
            idx = [self._to0(n) for n in tri]
            centroid = node0[idx, :].mean(axis=0)
            ax.text(centroid[0], centroid[1], centroid[2], str(i + 1), color="blue")

        self._plot_bars(ax, node0, color="k", linewidth=1.0)
        plt.show()
        return fig

    def Plot_Shape_Bar_Number(self):
        assembly = self.assembly
        node0 = assembly.node.coordinates_mat
        barConnect = assembly.bar.node_ij_mat

        fig, ax = self._setup_ax()
        self._plot_cst_faces(ax, node0, facecolor="yellow", alpha=1.0)
        self._plot_bars(ax, node0, color="k", linewidth=1.0)

        barConnect = np.array(barConnect)
        for i, (n1, n2) in enumerate(barConnect):
            p1 = node0[self._to0(n1)]
            p2 = node0[self._to0(n2)]
            mid = 0.5 * (p1 + p2)
            ax.text(mid[0], mid[1], mid[2], str(i + 1), color="blue")

        plt.show()
        return fig

    def Plot_Shape_RotSpr_3N_Number(self):
        assembly = self.assembly
        node0 = assembly.node.coordinates_mat

        fig, ax = self._setup_ax()
        self._plot_cst_faces(ax, node0, facecolor="yellow", alpha=1.0)
        self._plot_bars(ax, node0, color="k", linewidth=1.0)

        sprIJK = assembly.rotSpr3.node_ijk_mat
        sprIJK = np.array(sprIJK)
        for i, (n1, n2, n3) in enumerate(sprIJK):
            p = node0[self._to0(n2)]
            ax.text(p[0], p[1], p[2], str(i + 1), color="blue")

        plt.show()
        return fig

    def Plot_Shape_RotSpr_4N_Number(self):
        assembly = self.assembly
        node0 = assembly.node.coordinates_mat

        fig, ax = self._setup_ax()
        self._plot_cst_faces(ax, node0, facecolor="yellow", alpha=1.0)
        self._plot_bars(ax, node0, color="k", linewidth=1.0)

        sprIJKL = assembly.rotSpr.node_ijkl_mat
        sprIJKL = np.array(sprIJKL)
        for i, (n1, n2, n3, n4) in enumerate(sprIJKL):
            p2 = node0[self._to0(n2)]
            p3 = node0[self._to0(n3)]
            mid = 0.5 * (p2 + p3)
            ax.text(mid[0], mid[1], mid[2], str(i + 1), color="blue")

        plt.show()
        return fig

    def Plot_Deformed_Shape(self, U):
        assembly = self.assembly
        node0 = assembly.node.coordinates_mat
        deformNode = node0 + U

        fig, ax = self._setup_ax()
        self._plot_cst_faces(ax, node0, facecolor="black", alpha=0.2, edgecolor="k", linewidth=1.0)
        self._plot_bars(ax, node0, color=(0.5, 0.5, 0.5), linewidth=1.0)
        self._plot_bars(ax, deformNode, color="k", linewidth=1.2)
        self._plot_cst_faces(ax, deformNode, facecolor="yellow", alpha=1.0, edgecolor="k", linewidth=1.0)

        plt.show()
        return fig

    def Plot_Deformed_His(self, Uhis):
        """
        Uhis: (Incre, N, 3)
        Save GIF to self.fileName
        """
        if self.assembly is None:
            raise ValueError("Plot_Scissor_Bridge.assembly is None")

        assembly = self.assembly
        node0 = np.array(assembly.node.coordinates_mat, dtype=float)

        Uhis = np.array(Uhis, dtype=float)
        if Uhis.ndim != 3 or Uhis.shape[2] != 3:
            raise ValueError(f"Uhis must be (Incre, N, 3). Got {Uhis.shape}")

        Incre = Uhis.shape[0]
        if Incre == 0:
            raise ValueError("Uhis has 0 increments. Nothing to animate.")

        # abs output path + ensure directory exists
        out_path = os.path.abspath(self.fileName)
        out_dir = os.path.dirname(out_path)
        if out_dir and (not os.path.exists(out_dir)):
            os.makedirs(out_dir, exist_ok=True)

        fig = plt.figure(figsize=(self.width / self._sizeFactor, self.height / self._sizeFactor))
        ax = fig.add_subplot(111, projection="3d")

        images = []
        for i in range(Incre):
            ax.clear()
            ax.view_init(self.viewAngle1, self.viewAngle2)
            ax.set_facecolor("white")
            ax.set_box_aspect([1, 1, 1])
            self._set_axis_limits(ax)

            deformNode = node0 + Uhis[i, :, :]

            self._plot_bars(ax, deformNode, color="k", linewidth=1.0)
            self._plot_cst_faces(ax, deformNode, facecolor="yellow", alpha=1.0, edgecolor="k", linewidth=1.0)

            # Stable frame capture
            fig.canvas.draw()
            frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]  # RGBA -> RGB
            images.append(frame.copy())

        plt.close(fig)

        imageio.mimsave(out_path, images, duration=float(self.holdTime))
        print(f"[Plot_Deformed_His] GIF saved to: {out_path}")

    def Plot_Bar_Force(self, F):
        assembly = self.assembly
        node0 = assembly.node.coordinates_mat
        barConnect = assembly.bar.node_ij_mat
        F = np.array(F).reshape(-1)

        fig, ax = self._setup_ax()
        self._plot_bars(ax, node0, color="k", linewidth=1.0)

        barConnect = np.array(barConnect)
        for i, (n1, n2) in enumerate(barConnect):
            p1 = node0[self._to0(n1)]
            p2 = node0[self._to0(n2)]
            mid = 0.5 * (p1 + p2)
            val = int(np.round(F[i])) if i < len(F) else 0
            ax.text(mid[0], mid[1], mid[2], str(val), color="blue")

        plt.show()
        return fig

    def Plot_Shape_Bar_Stress(self, bar_stress):
        assembly = self.assembly
        node0 = assembly.node.coordinates_mat
        barConnect = assembly.bar.node_ij_mat

        bar_stress = np.array(bar_stress, dtype=float).reshape(-1)
        barConnect = np.array(barConnect)
        bar_num = barConnect.shape[0]

        if bar_stress.size != bar_num:
            raise ValueError(
                f"bar_stress length ({bar_stress.size}) must match number of bars ({bar_num})"
            )

        min_sx = float(np.min(bar_stress))
        max_sx = float(np.max(bar_stress))
        span = max_sx - min_sx

        fig, ax = self._setup_ax()
        self._plot_cst_faces(ax, node0, facecolor="yellow", alpha=1.0, edgecolor="k", linewidth=1.0)

        def _color_for_value(v):
            if span <= 0:
                return (0, 0, 1)
            if v > (4 / 5) * span + min_sx:
                return (1, 0, 0)      # red
            if v > (3 / 5) * span + min_sx:
                return (1, 0.5, 0)    # orange
            if v > (2 / 5) * span + min_sx:
                return (1, 1, 0)      # yellow
            if v > (1 / 5) * span + min_sx:
                return (0, 0.5, 0)    # green
            return (0, 0, 1)          # blue

        for i, (n1, n2) in enumerate(barConnect):
            p1 = node0[self._to0(n1)]
            p2 = node0[self._to0(n2)]
            color = _color_for_value(bar_stress[i])
            ax.plot([p1[0], p2[0]],
                    [p1[1], p2[1]],
                    [p1[2], p2[2]],
                    color=color, linewidth=2.0)

        if span > 0:
            edges = min_sx + span * (np.arange(6) / 5.0)
            colors = np.array([
                [1, 0, 0],
                [1, 0.5, 0],
                [1, 1, 0],
                [0, 0.5, 0],
                [0, 0, 1],
            ])

            lo_idx = [4, 3, 2, 1, 0]
            hi_idx = [5, 4, 3, 2, 1]

            handles = []
            labels = []
            for k in range(5):
                h = ax.plot([np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan],
                            color=colors[k], linewidth=2.0)[0]
                handles.append(h)
                labels.append(
                    f"{edges[lo_idx[k]]/1e6:.1f} to {edges[hi_idx[k]]/1e6:.1f} MPa"
                )

            ax.legend(handles, labels, loc="upper left")

        plt.show()
        return fig
