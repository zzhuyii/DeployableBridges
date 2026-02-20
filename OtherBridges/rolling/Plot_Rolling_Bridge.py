import os
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class Plot_Rolling_Bridge:
    def __init__(self):
        self.assembly = None

        self.viewAngle1 = 45
        self.viewAngle2 = 45
        self.displayRange = 1
        self.displayRangeRatio = 0.2

        self.width = 800
        self.height = 600
        self.x0 = 0
        self.y0 = 0

        self.holdTime = 0.01
        self.fileName = "animation.gif"

        self.activeTrussNum = None
        self.panelConnection = []

        self._sizeFactor = 120

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

    def _plot_bars(self, ax, nodes_xyz, bar_connect, color="k", linewidth=1.0):
        if bar_connect is None:
            return
        bar_connect = np.array(bar_connect)
        if bar_connect.size == 0:
            return
        for (n1, n2) in bar_connect:
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
        node0 = self.assembly.node.coordinates_mat

        fig, ax = self._setup_ax()
        self._plot_cst_faces(ax, node0, facecolor="yellow", alpha=1.0)
        self._plot_bars(ax, node0, self.assembly.bar.node_ij_mat, color="k", linewidth=1.0)

        for i in range(node0.shape[0]):
            ax.text(node0[i, 0], node0[i, 1], node0[i, 2], str(i + 1))

        plt.show()
        return fig

    def Plot_Shape_CST_Number(self):
        node0 = self.assembly.node.coordinates_mat

        fig, ax = self._setup_ax()
        self._plot_cst_faces(ax, node0, facecolor="yellow", alpha=1.0)

        cstIJK = np.array(self.assembly.cst.node_ijk_mat)
        for i, tri in enumerate(cstIJK):
            idx = [self._to0(n) for n in tri]
            centroid = node0[idx, :].mean(axis=0)
            ax.text(centroid[0], centroid[1], centroid[2], str(i + 1), color="blue")

        self._plot_bars(ax, node0, self.assembly.bar.node_ij_mat, color=(0.7, 0.7, 0.7), linewidth=1.0)
        if self.assembly.actBar is not None:
            self._plot_bars(ax, node0, self.assembly.actBar.node_ij_mat, color=(0.7, 0.7, 0.7), linewidth=1.0)

        plt.show()
        return fig

    def Plot_Shape_Bar_Number(self):
        node0 = self.assembly.node.coordinates_mat
        bar_connect = np.array(self.assembly.bar.node_ij_mat)

        fig, ax = self._setup_ax()
        self._plot_bars(ax, node0, bar_connect, color="k", linewidth=1.0)

        for i, (n1, n2) in enumerate(bar_connect):
            p1 = node0[self._to0(n1)]
            p2 = node0[self._to0(n2)]
            mid = 0.5 * (p1 + p2)
            ax.text(mid[0], mid[1], mid[2], str(i + 1), color="blue")

        if self.assembly.actBar is not None:
            self._plot_bars(ax, node0, self.assembly.actBar.node_ij_mat, color="blue", linewidth=1.0)
        self._plot_cst_faces(ax, node0, facecolor="yellow", alpha=1.0)

        plt.show()
        return fig

    def Plot_Shape_ActBar_Number(self):
        node0 = self.assembly.node.coordinates_mat

        fig, ax = self._setup_ax()
        self._plot_cst_faces(ax, node0, facecolor="yellow", alpha=1.0)
        self._plot_bars(ax, node0, self.assembly.bar.node_ij_mat, color="k", linewidth=1.0)

        act_connect = self.assembly.actBar.node_ij_mat
        self._plot_bars(ax, node0, act_connect, color="b", linewidth=1.0)

        act_connect = np.array(act_connect)
        for i, (n1, n2) in enumerate(act_connect):
            p1 = node0[self._to0(n1)]
            p2 = node0[self._to0(n2)]
            mid = 0.5 * (p1 + p2)
            ax.text(mid[0], mid[1], mid[2], str(i + 1), color="blue")

        plt.show()
        return fig

    def Plot_Shape_Spr_Number(self):
        node0 = self.assembly.node.coordinates_mat

        fig, ax = self._setup_ax()
        self._plot_cst_faces(ax, node0, facecolor="yellow", alpha=1.0)
        self._plot_bars(ax, node0, self.assembly.bar.node_ij_mat, color="k", linewidth=1.0)
        if self.assembly.actBar is not None:
            self._plot_bars(ax, node0, self.assembly.actBar.node_ij_mat, color=(0.7, 0.7, 0.7), linewidth=1.0)

        sprIJK = np.array(self.assembly.rot_spr_3N.node_ijk_mat)
        for i, (_, n2, _) in enumerate(sprIJK):
            p = node0[self._to0(n2)]
            ax.text(p[0], p[1], p[2], str(i + 1), color="blue")

        plt.show()
        return fig

    def Plot_Deformed_Shape(self, U):
        assembly = self.assembly
        node0 = assembly.node.coordinates_mat
        deformNode = node0 + U

        fig, ax = self._setup_ax()
        self._plot_bars(ax, node0, assembly.bar.node_ij_mat, color=(0.7, 0.7, 0.7), linewidth=1.0)
        self._plot_cst_faces(ax, node0, facecolor="black", alpha=0.2, edgecolor="k", linewidth=1.0)
        if assembly.actBar is not None:
            self._plot_bars(ax, node0, assembly.actBar.node_ij_mat, color=(0.7, 0.7, 0.7), linewidth=1.0)

        self._plot_bars(ax, deformNode, assembly.bar.node_ij_mat, color="k", linewidth=1.0)
        if assembly.actBar is not None:
            self._plot_bars(ax, deformNode, assembly.actBar.node_ij_mat, color="blue", linewidth=1.0)

        self._plot_cst_faces(ax, deformNode, facecolor="yellow", alpha=1.0, edgecolor="k", linewidth=1.0)

        plt.show()
        return fig

    def Plot_Deformed_His(self, Uhis):
        if self.assembly is None:
            raise ValueError("Plot_Rolling_Bridge.assembly is None")

        assembly = self.assembly
        node0 = np.array(assembly.node.coordinates_mat, dtype=float)

        Uhis = np.array(Uhis, dtype=float)
        if Uhis.ndim != 3 or Uhis.shape[2] != 3:
            raise ValueError(f"Uhis must be (Incre, N, 3). Got {Uhis.shape}")

        Incre = Uhis.shape[0]
        if Incre == 0:
            raise ValueError("Uhis has 0 increments. Nothing to animate.")

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

            self._plot_bars(ax, deformNode, assembly.bar.node_ij_mat, color="k", linewidth=1.0)
            if assembly.actBar is not None:
                self._plot_bars(ax, deformNode, assembly.actBar.node_ij_mat, color="b", linewidth=3.0)
            self._plot_cst_faces(ax, deformNode, facecolor="yellow", alpha=1.0, edgecolor="k", linewidth=1.0)

            fig.canvas.draw()
            frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
            images.append(frame.copy())

        plt.close(fig)
        imageio.mimsave(out_path, images, duration=float(self.holdTime))
        print(f"[Plot_Deformed_His] GIF saved to: {out_path}")

    def Plot_Bar_Force(self, F):
        node0 = self.assembly.node.coordinates_mat
        bar_connect = np.array(self.assembly.bar.node_ij_mat)
        F = np.array(F).reshape(-1)

        fig, ax = self._setup_ax()
        self._plot_bars(ax, node0, bar_connect, color="k", linewidth=1.0)

        for i, (n1, n2) in enumerate(bar_connect):
            p1 = node0[self._to0(n1)]
            p2 = node0[self._to0(n2)]
            mid = 0.5 * (p1 + p2)
            val = int(np.round(F[i])) if i < len(F) else 0
            ax.text(mid[0], mid[1], mid[2], str(val), color="blue")

        plt.show()
        return fig

    def Plot_Shape_Bar_Stress(self, bar_stress):
        node0 = self.assembly.node.coordinates_mat
        bar_connect = np.array(self.assembly.bar.node_ij_mat)

        bar_stress = np.array(bar_stress, dtype=float).reshape(-1)
        if bar_stress.size != bar_connect.shape[0]:
            raise ValueError("bar_stress length must match number of bars")

        min_sx = float(np.min(bar_stress))
        max_sx = float(np.max(bar_stress))
        span = max_sx - min_sx

        fig, ax = self._setup_ax()
        self._plot_cst_faces(ax, node0, facecolor="yellow", alpha=1.0, edgecolor="k", linewidth=1.0)

        def _color_for_value(v):
            if span <= 0:
                return (0, 0, 1)
            if v > (4 / 5) * span + min_sx:
                return (1, 0, 0)
            if v > (3 / 5) * span + min_sx:
                return (1, 0.5, 0)
            if v > (2 / 5) * span + min_sx:
                return (1, 1, 0)
            if v > (1 / 5) * span + min_sx:
                return (0, 0.5, 0)
            return (0, 0, 1)

        for i, (n1, n2) in enumerate(bar_connect):
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
