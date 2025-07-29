import numpy as np
import matplotlib.pyplot as plt
import imageio
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class Plot_KirigamiTruss:
    def __init__(self, view_angle1=15, view_angle2=70, display_range=10, display_range_ratio=1,
                 x0=100, y0=100, width=800, height=600, hold_time=0.5, filename="deformed.gif"):
        self.assembly = None
        self.view_angle1 = view_angle1
        self.view_angle2 = view_angle2
        self.display_range = display_range
        self.display_range_ratio = display_range_ratio
        self.x0 = x0
        self.y0 = y0
        self.width = width
        self.height = height
        self.hold_time = hold_time
        self.file_name = filename
        self.sizeFactor=40

    def Plot_Shape_Cst_Number(self):
        # Unpack display settings
        view1 = self.view_angle1
        view2 = self.view_angle2
        Vsize = self.display_range
        Vratio = self.display_range_ratio
    
        assembly = self.assembly
        node0 = assembly.node.coordinates_mat
        cstIJK = assembly.cst.node_ijk_mat
        panelNum = len(cstIJK)
    
        fig = plt.figure(figsize=(self.width / self.sizeFactor, self.height / self.sizeFactor))
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(view1, view2)
        ax.set_facecolor('white')
        plt.gca().set_aspect('auto', adjustable='box')
    
        # Set axis limits
        if isinstance(Vsize,(list, tuple, np.ndarray)):
            # single row, like [-1, L*(N+1), -1, 2, -1, 2]
            ax.set_xlim(Vsize[0], Vsize[1])
            ax.set_ylim(Vsize[2], Vsize[3])
            ax.set_zlim(Vsize[4], Vsize[5])
        else:
            ax.set_xlim(-Vsize*Vratio, Vsize)
            ax.set_ylim(-Vsize*Vratio, Vsize)
            ax.set_zlim(-Vsize*Vratio, Vsize)
    
        # Plot CST panels
        for k in range(panelNum):
            nodeNumVec = cstIJK[k]
            f = list(range(len(nodeNumVec)))
            v = [node0[nn-1] for nn in nodeNumVec]  # -1 for MATLAB-to-Python index
            verts = [v]
            patch = Poly3DCollection(verts, facecolors='yellow', linewidths=1, edgecolors='k', alpha=0.5)
            ax.add_collection3d(patch)
    
        # Number CST panels
        for i in range(panelNum):
            idxs = [n-1 for n in cstIJK[i]]  # MATLAB to Python index
            x = sum(node0[idx, 0] for idx in idxs) / 3
            y = sum(node0[idx, 1] for idx in idxs) / 3
            z = sum(node0[idx, 2] for idx in idxs) / 3
            ax.text(x, y, z, str(i+1), color='blue')
    
        plt.show()
        

    def Plot_Shape_Bar_Number(self):
        # Unpack display settings
        view1 = self.view_angle1
        view2 = self.view_angle2
        Vsize = self.display_range
        Vratio = self.display_range_ratio
    
        assembly = self.assembly
        node0 = assembly.node.coordinates_mat
        cstIJK = assembly.cst.node_ijk_mat
        panelNum = len(cstIJK)
    
        fig = plt.figure(figsize=(self.width / self.sizeFactor, self.height / self.sizeFactor))
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(view1, view2)
        ax.set_facecolor('white')
        plt.gca().set_aspect('auto', adjustable='box')
    
        # Set axis limits
        if isinstance(Vsize,(list, tuple, np.ndarray)):
            # single row, like [-1, L*(N+1), -1, 2, -1, 2]
            ax.set_xlim(Vsize[0], Vsize[1])
            ax.set_ylim(Vsize[2], Vsize[3])
            ax.set_zlim(Vsize[4], Vsize[5])
        else:
            ax.set_xlim(-Vsize*Vratio, Vsize)
            ax.set_ylim(-Vsize*Vratio, Vsize)
            ax.set_zlim(-Vsize*Vratio, Vsize)
    
        # Plot CST panels (yellow faces)
        for k in range(panelNum):
            nodeNumVec = cstIJK[k]
            v = [node0[nn-1] for nn in nodeNumVec]  # -1 for MATLAB to Python indexing
            verts = [v]
            patch = Poly3DCollection(verts, facecolors='yellow', linewidths=1, edgecolors='k', alpha=0.5)
            ax.add_collection3d(patch)
    
        # Plot bars (black lines)
        barConnect = assembly.bar.node_ij_mat
        barNum = len(barConnect)
        for j in range(barNum):
            n1, n2 = barConnect[j]
            node1 = node0[n1-1]  # -1 for 0-based Python indexing
            node2 = node0[n2-1]
            ax.plot([node1[0], node2[0]],
                    [node1[1], node2[1]],
                    [node1[2], node2[2]], color='k')
    
        # Number bars (midpoint)
        for i in range(barNum):
            n1, n2 = barConnect[i]
            x = 0.5 * (node0[n1-1, 0] + node0[n2-1, 0])
            y = 0.5 * (node0[n1-1, 1] + node0[n2-1, 1])
            z = 0.5 * (node0[n1-1, 2] + node0[n2-1, 2])
            ax.text(x, y, z, str(i+1), color='blue')
    
        plt.show()

    def Plot_Shape_Node_Number(self):
        # Unpack display settings
        view1 = self.view_angle1
        view2 = self.view_angle2
        Vsize = self.display_range
        Vratio = self.display_range_ratio
    
        assembly = self.assembly
        node0 = assembly.node.coordinates_mat
        cstIJK = assembly.cst.node_ijk_mat
        panelNum = len(cstIJK)
    
        fig = plt.figure(figsize=(self.width / self.sizeFactor, self.height / self.sizeFactor))
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(view1, view2)
        ax.set_facecolor('white')
        plt.gca().set_aspect('auto', adjustable='box')
    
        # Set axis limits
        if isinstance(Vsize,(list, tuple, np.ndarray)):
            # single row, like [-1, L*(N+1), -1, 2, -1, 2]
            ax.set_xlim(Vsize[0], Vsize[1])
            ax.set_ylim(Vsize[2], Vsize[3])
            ax.set_zlim(Vsize[4], Vsize[5])
        else:
            ax.set_xlim(-Vsize*Vratio, Vsize)
            ax.set_ylim(-Vsize*Vratio, Vsize)
            ax.set_zlim(-Vsize*Vratio, Vsize)
    
        # Plot CST panels (yellow faces)
        for k in range(panelNum):
            nodeNumVec = cstIJK[k]
            v = [node0[nn-1] for nn in nodeNumVec]  # -1 for MATLAB to Python index
            verts = [v]
            patch = Poly3DCollection(verts, facecolors='yellow', linewidths=1, edgecolors='k', alpha=0.5)
            ax.add_collection3d(patch)
    
        # Plot bars (black lines)
        barConnect = assembly.bar.node_ij_mat
        barNum = len(barConnect)
        for j in range(barNum):
            n1, n2 = barConnect[j]
            node1 = node0[n1-1]
            node2 = node0[n2-1]
            ax.plot([node1[0], node2[0]],
                    [node1[1], node2[1]],
                    [node1[2], node2[2]], color='k')
    
        # Plot and number all nodes
        N = node0.shape[0]
        for i in range(N):
            ax.text(node0[i,0], node0[i,1], node0[i,2], str(i+1), color='red', fontsize=8)
            # Optionally, use a scatter to highlight node positions
            ax.scatter(node0[i,0], node0[i,1], node0[i,2], color='blue', s=10)
    
        plt.show()

    def Plot_Shape_Spr_Number(self):
        # Unpack display settings
        view1 = self.view_angle1
        view2 = self.view_angle2
        Vsize = self.display_range
        Vratio = self.display_range_ratio
    
        assembly = self.assembly
        node0 = assembly.node.coordinates_mat
        cstIJK = assembly.cst.node_ijk_mat
        panelNum = len(cstIJK)
    
        fig = plt.figure(figsize=(self.width / self.sizeFactor, self.height / self.sizeFactor))
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(view1, view2)
        ax.set_facecolor('white')
        plt.gca().set_aspect('auto', adjustable='box')
    
        # Set axis limits
        if isinstance(Vsize,(list, tuple, np.ndarray)):
            # single row, like [-1, L*(N+1), -1, 2, -1, 2]
            ax.set_xlim(Vsize[0], Vsize[1])
            ax.set_ylim(Vsize[2], Vsize[3])
            ax.set_zlim(Vsize[4], Vsize[5])
        else:
            ax.set_xlim(-Vsize*Vratio, Vsize)
            ax.set_ylim(-Vsize*Vratio, Vsize)
            ax.set_zlim(-Vsize*Vratio, Vsize)
    
        # Plot CST panels (yellow faces)
        for k in range(panelNum):
            nodeNumVec = cstIJK[k]
            v = [node0[nn-1] for nn in nodeNumVec]  # -1 for MATLAB to Python index
            verts = [v]
            patch = Poly3DCollection(verts, facecolors='yellow', linewidths=1, edgecolors='k', alpha=0.5)
            ax.add_collection3d(patch)
    
        # Plot bars (black lines)
        barConnect = assembly.bar.node_ij_mat
        barNum = len(barConnect)
        for j in range(barNum):
            n1, n2 = barConnect[j]
            node1 = node0[n1-1]
            node2 = node0[n2-1]
            ax.plot([node1[0], node2[0]],
                    [node1[1], node2[1]],
                    [node1[2], node2[2]], color='k')
    
        # Number springs (at midpoints of 2nd and 3rd node in each spring)
        sprIJKL = assembly.rotSpr.node_ijkl_mat
        sprNum = len(sprIJKL)
        for i in range(sprNum):
            n2, n3 = sprIJKL[i][1]-1, sprIJKL[i][2]-1  # MATLAB to Python
            x = 0.5 * (node0[n2, 0] + node0[n3, 0])
            y = 0.5 * (node0[n2, 1] + node0[n3, 1])
            z = 0.5 * (node0[n2, 2] + node0[n3, 2])
            ax.text(x, y, z, str(i+1), color='blue')
    
        plt.show()


    def Plot_Deformed_Shape(self, U):
        # Unpack display settings
        view1 = self.view_angle1
        view2 = self.view_angle2
        Vsize = self.display_range
        Vratio = self.display_range_ratio
    
        assembly = self.assembly
        undeformedNode = assembly.node.coordinates_mat
    
        fig = plt.figure(figsize=(self.width / self.sizeFactor, self.height / self.sizeFactor))
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(view1, view2)
        ax.set_facecolor('white')
        plt.gca().set_aspect('equal')
    
        # Set axis limits
        if isinstance(Vsize,(list, tuple, np.ndarray)):
            # single row, like [-1, L*(N+1), -1, 2, -1, 2]
            ax.set_xlim(Vsize[0], Vsize[1])
            ax.set_ylim(Vsize[2], Vsize[3])
            ax.set_zlim(Vsize[4], Vsize[5])
        else:
            ax.set_xlim(-Vsize*Vratio, Vsize)
            ax.set_ylim(-Vsize*Vratio, Vsize)
            ax.set_zlim(-Vsize*Vratio, Vsize)
    
        node0 = assembly.node.coordinates_mat
        cstIJK = assembly.cst.node_ijk_mat
        panelNum = len(cstIJK)
    
        # Plot undeformed panels (black, alpha=0.2)
        for k in range(panelNum):
            nodeNumVec = cstIJK[k]
            v = [node0[nn-1] for nn in nodeNumVec]  # MATLAB to Python
            verts = [v]
            patch = Poly3DCollection(verts, facecolors='black', alpha=0.2, linewidths=1, edgecolors='k')
            ax.add_collection3d(patch)
    
        deformNode = undeformedNode + U  # U is (N,3) numpy array
    
        # Plot deformed bars (black lines)
        barConnect = assembly.bar.node_ij_mat
        barNum = len(barConnect)
        for j in range(barNum):
            n1, n2 = barConnect[j]
            node1 = deformNode[n1-1]
            node2 = deformNode[n2-1]
            ax.plot([node1[0], node2[0]],
                    [node1[1], node2[1]],
                    [node1[2], node2[2]], color='k')
    
        # Plot deformed panels (yellow, alpha=0.2)
        for k in range(panelNum):
            nodeNumVec = cstIJK[k]
            v = [deformNode[nn-1] for nn in nodeNumVec]  # MATLAB to Python
            verts = [v]
            patch = Poly3DCollection(verts, facecolors='yellow', alpha=0.2, linewidths=1, edgecolors='k')
            ax.add_collection3d(patch)
            
        plt.gca().set_aspect('equal')   
    
        plt.show()
        return fig

    def Plot_Deformed_History(self, Uhis):
        view1 = self.view_angle1
        view2 = self.view_angle2
        Vsize = self.display_range
        Vratio = self.display_range_ratio
    
        assembly = self.assembly
        undeformedNode = assembly.node.coordinates_mat
        pauseTime = getattr(self, 'holdTime', 0.1)
        filename = self.file_name
    
        Incre = Uhis.shape[0]
        images = []
    
        fig = plt.figure(figsize=(self.width / self.sizeFactor, self.height / self.sizeFactor))
        ax = fig.add_subplot(111, projection='3d')
    
        for i in range(Incre):
            ax.clear()
            ax.view_init(view1, view2)
            ax.set_facecolor('white')
            plt.gca().set_aspect('auto', adjustable='box')
    
            # Set axis limits
            if isinstance(Vsize,(list, tuple, np.ndarray)):
                # single row, like [-1, L*(N+1), -1, 2, -1, 2]
                ax.set_xlim(Vsize[0], Vsize[1])
                ax.set_ylim(Vsize[2], Vsize[3])
                ax.set_zlim(Vsize[4], Vsize[5])
            else:
                ax.set_xlim(-Vsize*Vratio, Vsize)
                ax.set_ylim(-Vsize*Vratio, Vsize)
                ax.set_zlim(-Vsize*Vratio, Vsize)
    
            tempU = Uhis[i, :, :]
            deformNode = undeformedNode + tempU
    
            # Plot bars (black lines)
            barConnect = assembly.bar.node_ij_mat
            barNum = len(barConnect)
            for j in range(barNum):
                n1, n2 = barConnect[j]
                node1 = deformNode[n1-1]
                node2 = deformNode[n2-1]
                ax.plot([node1[0], node2[0]],
                        [node1[1], node2[1]],
                        [node1[2], node2[2]], color='k')
    
            # Plot CST panels (grey faces)
            cstIJK = assembly.cst.node_ijk_mat
            panelNum = len(cstIJK)
            for k in range(panelNum):
                nodeNumVec = cstIJK[k]
                v = [deformNode[nn-1] for nn in nodeNumVec]  # MATLAB to Python index
                verts = [v]
                patch = Poly3DCollection(verts, facecolors=[(0.7, 0.7, 0.7)], linewidths=1, edgecolors='k')
                ax.add_collection3d(patch)
    
            # Save frame
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            images.append(image.copy())
    
        plt.close(fig)    
        # Write to GIF
        imageio.mimsave(filename, images, duration=pauseTime)