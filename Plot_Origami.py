import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class Plot_Origami:
    def __init__(self):
        self.assembly = None

        self.viewAngle1 = 45
        self.viewAngle2 = 45
        self.displayRange = 1
        self.displayRangeRatio = 0.2

        self.width = 8
        self.height = 4
        self.x0 = 0
        self.y0 = 0

        self.holdTime = 0.01
        self.fileName = "Animation.gif"

        self.sizeFactor = 40

    # -----------------------------
    # Helpers
    # -----------------------------
    def _set_axes(self, ax):
        ax.view_init(self.viewAngle1, self.viewAngle2)
        ax.set_facecolor('white')
        ax.set_box_aspect([1, 1, 1])

        vsize = self.displayRange
        if isinstance(vsize, (list, tuple, np.ndarray)) and np.asarray(vsize).size == 6:
            ax.set_xlim(vsize[0], vsize[1])
            ax.set_ylim(vsize[2], vsize[3])
            ax.set_zlim(vsize[4], vsize[5])
        else:
            ax.set_xlim(-self.displayRangeRatio * vsize, vsize)
            ax.set_ylim(-self.displayRangeRatio * vsize, vsize)
            ax.set_zlim(-self.displayRangeRatio * vsize, vsize)

 

    def Plot_Bar_Stress(self, bar_stress, U_end):
        assembly = self.assembly
        node0 = assembly.node.coordinates_mat
        deformedNode = node0 + U_end


        fig = plt.figure(figsize=(self.width, self.height))
        ax = fig.add_subplot(111, projection='3d')
        self._set_axes(ax)

        # Plot CST panels
        cstIJK = assembly.cst.node_ijk_mat
        for k in range(cstIJK.shape[0]):
            nodeNumVec = cstIJK[k]
            v = [deformedNode[nn - 1] for nn in nodeNumVec]
            patch = Poly3DCollection([v], facecolors='yellow', linewidths=0, edgecolors='k', alpha=0.5)
            ax.add_collection3d(patch)

        # Color-code bar stress
        barConnect = assembly.bar.node_ij_mat
        min_sx = float(np.min(bar_stress))
        max_sx = float(np.max(bar_stress))
        span = max_sx - min_sx if max_sx != min_sx else 1.0

        for j in range(barConnect.shape[0]):
            v = bar_stress[j]
            if v > (4 / 5) * span + min_sx:
                color = (1, 0, 0)          # red
            elif v > (3 / 5) * span + min_sx:
                color = (1, 0.5, 0)        # orange
            elif v > (2 / 5) * span + min_sx:
                color = (1, 1, 0)          # yellow
            elif v > (1 / 5) * span + min_sx:
                color = (0, 0.5, 0)        # green
            else:
                color = (0, 0, 1)          # blue

            n1, n2 = barConnect[j]
            node1 = deformedNode[n1 - 1]
            node2 = deformedNode[n2 - 1]
            ax.plot([node1[0], node2[0]], [node1[1], node2[1]], [node1[2], node2[2]],
                    color=color, linewidth=2)

        # Legend
        edges = min_sx + span * np.arange(6) / 5.0
        colors = [
            (1, 0, 0),
            (1, 0.5, 0),
            (1, 1, 0),
            (0, 0.5, 0),
            (0, 0, 1),
        ]
        lo_idx = [4, 3, 2, 1, 0]
        hi_idx = [5, 4, 3, 2, 1]

        handles = []
        labels = []
        for k in range(5):
            h = ax.plot([np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan],
                        color=colors[k], linewidth=2)[0]
            handles.append(h)
            labels.append(f"{edges[lo_idx[k]]/1e6:.1f} to {edges[hi_idx[k]]/1e6:.1f} MPa")

        ax.legend(handles, labels, loc='upper left')
        plt.gca().set_aspect('equal')
        plt.show()
        return fig


    def Plot_Shape_Bar_Failure(self, pass_yn, U_end):
        assembly = self.assembly
        node0 = assembly.node.coordinates_mat
        deformedNode = node0 + U_end
        
        bar_connect = assembly.bar.node_ij_mat
        pass_yn = np.asarray(pass_yn, dtype=bool).reshape(-1)

        fig = plt.figure(figsize=(self.width, self.height ))
        ax = fig.add_subplot(111, projection='3d')
        self._set_axes(ax)

        for ok, (n1, n2) in zip(pass_yn, bar_connect):
            node1 = deformedNode[n1 - 1]
            node2 = deformedNode[n2 - 1]
            color = 'green' if ok else 'red'
            ax.plot([node1[0], node2[0]], [node1[1], node2[1]], [node1[2], node2[2]],
                    color=color, linewidth=2)

        plt.gca().set_aspect('equal')
        plt.show()
        return fig
