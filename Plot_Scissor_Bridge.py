import os
from matplotlib.patches import Patch
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

    def _plot_cst_faces(self, ax, nodes_xyz, facecolor="yellow", alpha=0.5, edgecolor="k", linewidth=0):
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


    def Plot_Shape_Bar_Stress(self, bar_stress, U):
        
        node0 = self.assembly.node.coordinates_mat
        deformNode = node0 + U  # U is (N,3) numpy array
        bar_connect = self.assembly.bar.node_ij_mat
        
        bar_stress = np.asarray(bar_stress, dtype=float).reshape(-1)
        fig, ax = self._setup_ax()
        self._plot_cst_faces(ax, deformNode, facecolor="yellow", alpha=0.5)
        vmin = float(np.min(bar_stress))
        vmax = float(np.max(bar_stress))
        span = max(vmax - vmin, 1e-12)
        cmap = plt.get_cmap("coolwarm")
        for stress, (n1, n2) in zip(bar_stress, np.asarray(self._bar())):
            color = cmap((stress - vmin) / span)
            p1 = deformNode[self._to0(n1)]
            p2 = deformNode[self._to0(n2)]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=color, linewidth=2.0)
            
        min_sx = float(np.min(bar_stress))
        max_sx = float(np.max(bar_stress))
        span = max(max_sx - min_sx, 1.0)

        for stress, (n1, n2) in zip(bar_stress, bar_connect):
            if stress > 4 / 5 * span + min_sx:
                color = 'red'
            elif stress > 3 / 5 * span + min_sx:
                color = 'orange'
            elif stress > 2 / 5 * span + min_sx:
                color = 'yellow'
            elif stress > 1 / 5 * span + min_sx:
                color = 'green'
            else:
                color = 'blue'
            p1 = deformNode[n1 - 1]
            p2 = deformNode[n2 - 1]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                    color=color, linewidth=2)

        legend_patches = [
            Patch(color="red", label="{:.1f} to {:.1f} MPa".format((4 / 5 * span + min_sx) / 1e6, max_sx / 1e6)),
            Patch(color="orange", label="{:.1f} to {:.1f} MPa".format((3 / 5 * span + min_sx) / 1e6, (4 / 5 * span + min_sx) / 1e6)),
            Patch(color="yellow", label="{:.1f} to {:.1f} MPa".format((2 / 5 * span + min_sx) / 1e6, (3 / 5 * span + min_sx) / 1e6)),
            Patch(color="green", label="{:.1f} to {:.1f} MPa".format((1 / 5 * span + min_sx) / 1e6, (2 / 5 * span + min_sx) / 1e6)),
            Patch(color="blue", label="{:.1f} to {:.1f} MPa".format(min_sx / 1e6, (1 / 5 * span + min_sx) / 1e6)),
        ]
        ax.legend(handles=legend_patches, loc='upper left', bbox_to_anchor=(0, 1))    
        plt.gca().set_aspect('equal')    
        plt.show()    
        return fig

    def Plot_Shape_Bar_Failure(self, pass_yn,U_end):
        node0 = self.assembly.node.coordinates_mat
        deformedNode = node0 + U_end
        pass_yn = np.asarray(pass_yn, dtype=bool).reshape(-1)
        fig, ax = self._setup_ax()
        self._plot_cst_faces(ax, deformedNode, facecolor="yellow", alpha=0.45)
        for passed, (n1, n2) in zip(pass_yn, np.asarray(self._bar())):
            p1 = deformedNode[self._to0(n1)]
            p2 = deformedNode[self._to0(n2)]
            color = "green" if passed else "red"
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=color, linewidth=2.2)
        plt.gca().set_aspect('equal')  
        return fig
