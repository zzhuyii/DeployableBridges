import os
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class Plot_Rolling_Bridge:
    def __init__(self):
        self.assembly = None

        self.viewAngle1 = 10
        self.viewAngle2 = -75
        self.displayRange = 1
        self.displayRangeRatio = 0.2

        self.width = 8
        self.height = 4
        self.x0 = 0
        self.y0 = 0

        self.holdTime = 0.01
        self.fileName = "animation.gif"

        self.activeTrussNum = None
        self.panelConnection = []


    def _setup_ax(self):
        fig = plt.figure(figsize=(self.width, self.height))
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

    def _plot_cst_faces(self, ax, nodes_xyz, facecolor="yellow", alpha=0.5, edgecolor="k", linewidth=0):
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



    def Plot_Shape_Bar_Failure(self, pass_yn, U_end):
        node0 = self.assembly.node.coordinates_mat
        deformedNode = node0 + U_end
        
        bar_connect = np.array(self.assembly.bar.node_ij_mat)
        pass_yn = np.asarray(pass_yn, dtype=bool).reshape(-1)

        fig, ax = self._setup_ax()
        self._plot_cst_faces(ax, deformedNode, facecolor="yellow", alpha=0.5, edgecolor="k", linewidth=0)

        for ok, (n1, n2) in zip(pass_yn, bar_connect):
            p1 = deformedNode[self._to0(n1)]
            p2 = deformedNode[self._to0(n2)]
            color = "green" if ok else "red"
            ax.plot([p1[0], p2[0]],
                    [p1[1], p2[1]],
                    [p1[2], p2[2]],
                    color=color, linewidth=2.0)
            
        ax.view_init(15, -75)
        plt.gca().set_aspect('equal')    

        plt.show()
        return fig


    def Plot_Shape_Bar_Stress(self, bar_stress,U_end):
        node0 = self.assembly.node.coordinates_mat
        deformedNode = node0 + U_end
        
        bar_connect = np.array(self.assembly.bar.node_ij_mat)

        bar_stress = np.array(bar_stress, dtype=float).reshape(-1)
        if bar_stress.size != bar_connect.shape[0]:
            raise ValueError("bar_stress length must match number of bars")

        min_sx = float(np.min(bar_stress))
        max_sx = float(np.max(bar_stress))
        span = max_sx - min_sx

        fig, ax = self._setup_ax()
        self._plot_cst_faces(ax, deformedNode, facecolor="yellow", alpha=0.5, edgecolor="k", linewidth=0)

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
            p1 = deformedNode[self._to0(n1)]
            p2 = deformedNode[self._to0(n2)]
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
        
        ax.view_init(15, -75)

        plt.gca().set_aspect('equal')
        plt.show()
        return fig
