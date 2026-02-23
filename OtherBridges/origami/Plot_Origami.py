import numpy as np
import matplotlib.pyplot as plt
import imageio
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class Plot_Origami:
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

    def _canvas_to_rgb(self, fig):
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        try:
            buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            return buf.reshape(h, w, 3)
        except Exception:
            buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
            argb = buf.reshape(h, w, 4)
            rgb = argb[:, :, 1:4]
            return rgb

    # -----------------------------
    # Plot helpers for numbering
    # -----------------------------
    def Plot_Shape_Node_Number(self):
        assembly = self.assembly
        node0 = assembly.node.coordinates_mat

        fig = plt.figure(figsize=(self.width / self.sizeFactor, self.height / self.sizeFactor))
        ax = fig.add_subplot(111, projection='3d')
        self._set_axes(ax)

        # Plot CST panels
        cstIJK = assembly.cst.node_ijk_mat
        for k in range(cstIJK.shape[0]):
            nodeNumVec = cstIJK[k]
            v = [node0[nn - 1] for nn in nodeNumVec]
            patch = Poly3DCollection([v], facecolors='yellow', linewidths=1, edgecolors='k', alpha=0.5)
            ax.add_collection3d(patch)

        # Plot bars
        barConnect = assembly.bar.node_ij_mat
        for j in range(barConnect.shape[0]):
            n1, n2 = barConnect[j]
            node1 = node0[n1 - 1]
            node2 = node0[n2 - 1]
            ax.plot([node1[0], node2[0]], [node1[1], node2[1]], [node1[2], node2[2]], color='k')

        # Plot node numbers
        for i in range(node0.shape[0]):
            ax.text(node0[i, 0], node0[i, 1], node0[i, 2], str(i + 1), color='red', fontsize=8)
            ax.scatter(node0[i, 0], node0[i, 1], node0[i, 2], color='blue', s=10)

        plt.show()
        return fig

    def Plot_Shape_Spr_Number(self):
        assembly = self.assembly
        node0 = assembly.node.coordinates_mat

        fig = plt.figure(figsize=(self.width / self.sizeFactor, self.height / self.sizeFactor))
        ax = fig.add_subplot(111, projection='3d')
        self._set_axes(ax)

        cstIJK = assembly.cst.node_ijk_mat
        for k in range(cstIJK.shape[0]):
            nodeNumVec = cstIJK[k]
            v = [node0[nn - 1] for nn in nodeNumVec]
            patch = Poly3DCollection([v], facecolors='yellow', linewidths=1, edgecolors='k', alpha=0.5)
            ax.add_collection3d(patch)

        barConnect = assembly.bar.node_ij_mat
        for j in range(barConnect.shape[0]):
            n1, n2 = barConnect[j]
            node1 = node0[n1 - 1]
            node2 = node0[n2 - 1]
            ax.plot([node1[0], node2[0]], [node1[1], node2[1]], [node1[2], node2[2]], color='k')

        sprIJKL = assembly.rotSpr.node_ijkl_mat
        for i in range(sprIJKL.shape[0]):
            n2 = sprIJKL[i][1] - 1
            n3 = sprIJKL[i][2] - 1
            x = 0.5 * (node0[n2, 0] + node0[n3, 0])
            y = 0.5 * (node0[n2, 1] + node0[n3, 1])
            z = 0.5 * (node0[n2, 2] + node0[n3, 2])
            ax.text(x, y, z, str(i + 1), color='blue')

        plt.show()
        return fig

    def Plot_Shape_CST_Number(self):
        assembly = self.assembly
        node0 = assembly.node.coordinates_mat
        cstIJK = assembly.cst.node_ijk_mat

        fig = plt.figure(figsize=(self.width / self.sizeFactor, self.height / self.sizeFactor))
        ax = fig.add_subplot(111, projection='3d')
        self._set_axes(ax)

        for k in range(cstIJK.shape[0]):
            nodeNumVec = cstIJK[k]
            v = [node0[nn - 1] for nn in nodeNumVec]
            patch = Poly3DCollection([v], facecolors='yellow', linewidths=1, edgecolors='k', alpha=0.5)
            ax.add_collection3d(patch)

        for i in range(cstIJK.shape[0]):
            idxs = [n - 1 for n in cstIJK[i]]
            x = sum(node0[idx, 0] for idx in idxs) / 3
            y = sum(node0[idx, 1] for idx in idxs) / 3
            z = sum(node0[idx, 2] for idx in idxs) / 3
            ax.text(x, y, z, str(i + 1), color='blue')

        plt.show()
        return fig

    def Plot_Shape_Bar_Number(self):
        assembly = self.assembly
        node0 = assembly.node.coordinates_mat

        fig = plt.figure(figsize=(self.width / self.sizeFactor, self.height / self.sizeFactor))
        ax = fig.add_subplot(111, projection='3d')
        self._set_axes(ax)

        cstIJK = assembly.cst.node_ijk_mat
        for k in range(cstIJK.shape[0]):
            nodeNumVec = cstIJK[k]
            v = [node0[nn - 1] for nn in nodeNumVec]
            patch = Poly3DCollection([v], facecolors='yellow', linewidths=1, edgecolors='k', alpha=0.5)
            ax.add_collection3d(patch)

        barConnect = assembly.bar.node_ij_mat
        for j in range(barConnect.shape[0]):
            n1, n2 = barConnect[j]
            node1 = node0[n1 - 1]
            node2 = node0[n2 - 1]
            ax.plot([node1[0], node2[0]], [node1[1], node2[1]], [node1[2], node2[2]], color='k')

        for i in range(barConnect.shape[0]):
            n1, n2 = barConnect[i]
            x = 0.5 * (node0[n1 - 1, 0] + node0[n2 - 1, 0])
            y = 0.5 * (node0[n1 - 1, 1] + node0[n2 - 1, 1])
            z = 0.5 * (node0[n1 - 1, 2] + node0[n2 - 1, 2])
            ax.text(x, y, z, str(i + 1), color='blue')

        plt.show()
        return fig

    # -----------------------------
    # Deformed plots
    # -----------------------------
    def Plot_Deformed_Shape(self, U):
        assembly = self.assembly
        undeformedNode = assembly.node.coordinates_mat

        fig = plt.figure(figsize=(self.width / self.sizeFactor, self.height / self.sizeFactor))
        ax = fig.add_subplot(111, projection='3d')
        self._set_axes(ax)

        node0 = assembly.node.coordinates_mat
        cstIJK = assembly.cst.node_ijk_mat

        # Undeformed CST (black, alpha=0.2)
        for k in range(cstIJK.shape[0]):
            nodeNumVec = cstIJK[k]
            v = [node0[nn - 1] for nn in nodeNumVec]
            patch = Poly3DCollection([v], facecolors='black', alpha=0.2, linewidths=1, edgecolors='k')
            ax.add_collection3d(patch)

        # Undeformed bars (gray)
        barConnect = assembly.bar.node_ij_mat
        for j in range(barConnect.shape[0]):
            n1, n2 = barConnect[j]
            node1 = node0[n1 - 1]
            node2 = node0[n2 - 1]
            ax.plot([node1[0], node2[0]], [node1[1], node2[1]], [node1[2], node2[2]], color=(0.5, 0.5, 0.5))

        deformNode = undeformedNode + U

        # Deformed CST (yellow)
        for k in range(cstIJK.shape[0]):
            nodeNumVec = cstIJK[k]
            v = [deformNode[nn - 1] for nn in nodeNumVec]
            patch = Poly3DCollection([v], facecolors='yellow', linewidths=1, edgecolors='k')
            ax.add_collection3d(patch)

        # Deformed bars (black)
        for j in range(barConnect.shape[0]):
            n1, n2 = barConnect[j]
            node1 = deformNode[n1 - 1]
            node2 = deformNode[n2 - 1]
            ax.plot([node1[0], node2[0]], [node1[1], node2[1]], [node1[2], node2[2]], color='k')

        plt.show()
        return fig

    def Plot_Deformed_His(self, Uhis):
        assembly = self.assembly
        undeformedNode = assembly.node.coordinates_mat

        images = []
        fig = plt.figure(figsize=(self.width / self.sizeFactor, self.height / self.sizeFactor))
        ax = fig.add_subplot(111, projection='3d')

        for i in range(Uhis.shape[0]):
            ax.clear()
            self._set_axes(ax)

            tempU = Uhis[i, :, :]
            deformNode = undeformedNode + tempU

            # Bars
            barConnect = assembly.bar.node_ij_mat
            for j in range(barConnect.shape[0]):
                n1, n2 = barConnect[j]
                node1 = deformNode[n1 - 1]
                node2 = deformNode[n2 - 1]
                ax.plot([node1[0], node2[0]], [node1[1], node2[1]], [node1[2], node2[2]], color='k')

            # CST
            cstIJK = assembly.cst.node_ijk_mat
            for k in range(cstIJK.shape[0]):
                nodeNumVec = cstIJK[k]
                v = [deformNode[nn - 1] for nn in nodeNumVec]
                patch = Poly3DCollection([v], facecolors='yellow', linewidths=1, edgecolors='k')
                ax.add_collection3d(patch)

            images.append(self._canvas_to_rgb(fig))

        plt.close(fig)
        imageio.mimsave(self.fileName, images, duration=self.holdTime)

    def Plot_Bar_Stress(self, U_or_stress):
        assembly = self.assembly
        node0 = assembly.node.coordinates_mat

        # Accept either displacement U (NodeNum x 3) or direct bar stress (Nb,)
        if isinstance(U_or_stress, np.ndarray) and U_or_stress.ndim == 2:
            ex = assembly.bar.solve_strain(assembly.node, U_or_stress)
            sx, _ = assembly.bar.solve_stress(ex)
            bar_stress = sx
        else:
            bar_stress = np.asarray(U_or_stress, dtype=float).reshape(-1)

        fig = plt.figure(figsize=(self.width / self.sizeFactor, self.height / self.sizeFactor))
        ax = fig.add_subplot(111, projection='3d')
        self._set_axes(ax)

        # Plot CST panels
        cstIJK = assembly.cst.node_ijk_mat
        for k in range(cstIJK.shape[0]):
            nodeNumVec = cstIJK[k]
            v = [node0[nn - 1] for nn in nodeNumVec]
            patch = Poly3DCollection([v], facecolors='yellow', linewidths=1, edgecolors='k')
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
            node1 = node0[n1 - 1]
            node2 = node0[n2 - 1]
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
        plt.show()
        return fig
