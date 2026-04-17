import os

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class Plot_Scissor_Bridge:
    """Plot helper for the MATLAB scissor bridge scripts."""

    def __init__(self):
        self.assembly = None
        self.viewAngle1 = 45
        self.viewAngle2 = 45
        self.displayRange = 1
        self.displayRangeRatio = 0.2
        self.width = 900
        self.height = 650
        self.holdTime = 0.03
        self.fileName = "Animation.gif"
        self._sizeFactor = 120

    @staticmethod
    def _to0(node_id):
        return int(node_id) - 1

    def _setup_ax(self):
        fig = plt.figure(figsize=(self.width / self._sizeFactor, self.height / self._sizeFactor))
        ax = fig.add_subplot(111, projection="3d")
        ax.view_init(self.viewAngle1, self.viewAngle2)
        ax.set_facecolor("white")
        ax.set_box_aspect([1, 1, 1])
        self._set_axis_limits(ax)
        return fig, ax

    def _set_axis_limits(self, ax):
        vsize = self.displayRange
        if isinstance(vsize, (list, tuple, np.ndarray)) and len(np.asarray(vsize).reshape(-1)) == 6:
            v = np.asarray(vsize, dtype=float).reshape(-1)
            ax.set_xlim(v[0], v[1])
            ax.set_ylim(v[2], v[3])
            ax.set_zlim(v[4], v[5])
        else:
            ax.set_xlim(-self.displayRangeRatio * vsize, vsize)
            ax.set_ylim(-self.displayRangeRatio * vsize, vsize)
            ax.set_zlim(-self.displayRangeRatio * vsize, vsize)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")

    def _cst(self):
        cst = getattr(self.assembly, "cst", None)
        return None if cst is None else cst.node_ijk_mat

    def _bar(self):
        bar = getattr(self.assembly, "bar", None)
        return None if bar is None else bar.node_ij_mat

    def _act_bar(self):
        act_bar = getattr(self.assembly, "actBar", None)
        return None if act_bar is None else act_bar.node_ij_mat

    def _rot3(self):
        spr = getattr(self.assembly, "rot_spr_3N", None)
        return None if spr is None else spr.node_ijk_mat

    def _rot4(self):
        spr = getattr(self.assembly, "rot_spr_4N", None)
        return None if spr is None else spr.node_ijkl_mat

    def _plot_cst_faces(self, ax, nodes_xyz, facecolor="yellow", alpha=1.0, edgecolor="k", linewidth=0.8):
        cst = self._cst()
        if cst is None:
            return
        cst = np.asarray(cst)
        if cst.size == 0:
            return
        for tri in cst:
            verts = [nodes_xyz[self._to0(n)] for n in tri]
            patch = Poly3DCollection([verts], facecolors=facecolor, edgecolors=edgecolor,
                                     linewidths=linewidth, alpha=alpha)
            ax.add_collection3d(patch)

    def _plot_lines(self, ax, nodes_xyz, connect, color="k", linewidth=1.0):
        if connect is None:
            return
        connect = np.asarray(connect)
        if connect.size == 0:
            return
        for n1, n2 in connect:
            p1 = nodes_xyz[self._to0(n1)]
            p2 = nodes_xyz[self._to0(n2)]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                    color=color, linewidth=linewidth)

    def Plot_Shape_Node_Number(self):
        node0 = self.assembly.node.coordinates_mat
        fig, ax = self._setup_ax()
        self._plot_cst_faces(ax, node0)
        self._plot_lines(ax, node0, self._bar(), color="k")
        self._plot_lines(ax, node0, self._act_bar(), color="tab:blue", linewidth=1.2)
        for i, xyz in enumerate(node0):
            ax.text(xyz[0], xyz[1], xyz[2], str(i + 1), color="red", fontsize=7)
        return fig

    def Plot_Shape_CST_Number(self):
        node0 = self.assembly.node.coordinates_mat
        fig, ax = self._setup_ax()
        self._plot_cst_faces(ax, node0)
        self._plot_lines(ax, node0, self._bar(), color="0.25")
        cst = np.asarray(self._cst())
        for i, tri in enumerate(cst):
            xyz = node0[[self._to0(n) for n in tri]].mean(axis=0)
            ax.text(xyz[0], xyz[1], xyz[2], str(i + 1), color="blue", fontsize=7)
        return fig

    def Plot_Shape_Bar_Number(self):
        node0 = self.assembly.node.coordinates_mat
        fig, ax = self._setup_ax()
        self._plot_cst_faces(ax, node0)
        self._plot_lines(ax, node0, self._bar(), color="k")
        for i, (n1, n2) in enumerate(np.asarray(self._bar())):
            mid = 0.5 * (node0[self._to0(n1)] + node0[self._to0(n2)])
            ax.text(mid[0], mid[1], mid[2], str(i + 1), color="blue", fontsize=7)
        return fig

    def Plot_Shape_ActBar_Number(self):
        node0 = self.assembly.node.coordinates_mat
        fig, ax = self._setup_ax()
        self._plot_cst_faces(ax, node0)
        self._plot_lines(ax, node0, self._bar(), color="0.35")
        self._plot_lines(ax, node0, self._act_bar(), color="tab:blue", linewidth=2.0)
        for i, (n1, n2) in enumerate(np.asarray(self._act_bar())):
            mid = 0.5 * (node0[self._to0(n1)] + node0[self._to0(n2)])
            ax.text(mid[0], mid[1], mid[2], str(i + 1), color="tab:blue", fontsize=7)
        return fig

    def Plot_Shape_RotSpr_3N_Number(self):
        node0 = self.assembly.node.coordinates_mat
        fig, ax = self._setup_ax()
        self._plot_cst_faces(ax, node0)
        self._plot_lines(ax, node0, self._bar(), color="k")
        for i, (_, n2, _) in enumerate(np.asarray(self._rot3())):
            p = node0[self._to0(n2)]
            ax.text(p[0], p[1], p[2], str(i + 1), color="blue", fontsize=7)
        return fig

    def Plot_Shape_RotSpr_4N_Number(self):
        node0 = self.assembly.node.coordinates_mat
        fig, ax = self._setup_ax()
        self._plot_cst_faces(ax, node0)
        self._plot_lines(ax, node0, self._bar(), color="k")
        for i, (_, n2, n3, _) in enumerate(np.asarray(self._rot4())):
            mid = 0.5 * (node0[self._to0(n2)] + node0[self._to0(n3)])
            ax.text(mid[0], mid[1], mid[2], str(i + 1), color="blue", fontsize=7)
        return fig

    def Plot_Deformed_Shape(self, U):
        node0 = self.assembly.node.coordinates_mat
        deform_node = node0 + np.asarray(U, dtype=float)
        fig, ax = self._setup_ax()
        self._plot_cst_faces(ax, node0, facecolor="black", alpha=0.18)
        self._plot_lines(ax, node0, self._bar(), color="0.65", linewidth=0.8)
        self._plot_lines(ax, node0, self._act_bar(), color="0.65", linewidth=0.8)
        self._plot_lines(ax, deform_node, self._bar(), color="k", linewidth=1.1)
        self._plot_lines(ax, deform_node, self._act_bar(), color="tab:blue", linewidth=1.8)
        self._plot_cst_faces(ax, deform_node, facecolor="yellow", alpha=0.92)
        return fig

    def Plot_Deformed_His(self, Uhis):
        node0 = self.assembly.node.coordinates_mat
        Uhis = np.asarray(Uhis, dtype=float)
        if Uhis.ndim != 3 or Uhis.shape[2] != 3:
            raise ValueError(f"Uhis must be (steps, nodes, 3), got {Uhis.shape}")

        out_path = os.path.abspath(self.fileName)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        fig = plt.figure(figsize=(self.width / self._sizeFactor, self.height / self._sizeFactor))
        ax = fig.add_subplot(111, projection="3d")
        images = []
        for U in Uhis:
            ax.clear()
            ax.view_init(self.viewAngle1, self.viewAngle2)
            ax.set_facecolor("white")
            ax.set_box_aspect([1, 1, 1])
            self._set_axis_limits(ax)
            deform_node = node0 + U
            self._plot_lines(ax, deform_node, self._bar(), color="k", linewidth=1.0)
            self._plot_lines(ax, deform_node, self._act_bar(), color="tab:blue", linewidth=1.8)
            self._plot_cst_faces(ax, deform_node, facecolor="yellow", alpha=0.92)
            fig.canvas.draw()
            images.append(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
        plt.close(fig)
        imageio.mimsave(out_path, images, duration=float(self.holdTime))
        print(f"[Plot_Deformed_His] GIF saved to: {out_path}")

    def Plot_Bar_Force(self, F):
        node0 = self.assembly.node.coordinates_mat
        F = np.asarray(F).reshape(-1)
        fig, ax = self._setup_ax()
        self._plot_lines(ax, node0, self._bar(), color="k")
        for i, (n1, n2) in enumerate(np.asarray(self._bar())):
            mid = 0.5 * (node0[self._to0(n1)] + node0[self._to0(n2)])
            ax.text(mid[0], mid[1], mid[2], f"{F[i]:.0f}", color="blue", fontsize=7)
        return fig

    def Plot_Shape_Bar_Stress(self, bar_stress):
        node0 = self.assembly.node.coordinates_mat
        bar_stress = np.asarray(bar_stress, dtype=float).reshape(-1)
        fig, ax = self._setup_ax()
        self._plot_cst_faces(ax, node0, facecolor="yellow", alpha=0.75)
        vmin = float(np.min(bar_stress))
        vmax = float(np.max(bar_stress))
        span = max(vmax - vmin, 1e-12)
        cmap = plt.get_cmap("coolwarm")
        for stress, (n1, n2) in zip(bar_stress, np.asarray(self._bar())):
            color = cmap((stress - vmin) / span)
            p1 = node0[self._to0(n1)]
            p2 = node0[self._to0(n2)]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=color, linewidth=2.0)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin / 1e6, vmax=vmax / 1e6))
        sm.set_array([])
        fig.colorbar(sm, ax=ax, shrink=0.65, label="Bar stress (MPa)")
        return fig

    def Plot_Shape_Bar_Failure(self, pass_yn):
        node0 = self.assembly.node.coordinates_mat
        pass_yn = np.asarray(pass_yn, dtype=bool).reshape(-1)
        fig, ax = self._setup_ax()
        self._plot_cst_faces(ax, node0, facecolor="yellow", alpha=0.45)
        for passed, (n1, n2) in zip(pass_yn, np.asarray(self._bar())):
            p1 = node0[self._to0(n1)]
            p2 = node0[self._to0(n2)]
            color = "green" if passed else "red"
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=color, linewidth=2.2)
        return fig
