from Elements_Nodes import Elements_Nodes
from CD_Elements_Bars import CD_Elements_Bars
from CD_Elements_CST import CD_Elements_CST
from CD_Elements_RotSprings_3N import CD_Elements_RotSprings_3N
from CD_Elements_RotSprings_4N import CD_Elements_RotSprings_4N
from Vec_Elements_Bars import Vec_Elements_Bars
from Vec_Elements_RotSprings_4N import Vec_Elements_RotSprings_4N
from Vec_Elements_CST import Vec_Elements_CST

from Assembly_KirigamiTruss import Assembly_KirigamiTruss
from Plot_KirigamiTruss import Plot_KirigamiTruss

import numpy as np

node=Elements_Nodes()
bar=Vec_Elements_Bars()
cst=Vec_Elements_CST()
rs3=CD_Elements_RotSprings_3N()
rs4=Vec_Elements_RotSprings_4N()

node.coordinates_mat=np.array([[0, 0, 0.0],
                               [0, 1, 1],
                               [0, 0, 2],
                               [0, -1, 1]])

disp1=0.1
disp2=0

nodispMat = np.array([[0,0,0],
                  [0,0,0],
                  [0,0,0],
                  [0,0,0]])

dispMat = np.array([[0,0,0],
                  [disp1,disp2,0],
                  [0,0,0],
                  [0,0,0]])

dispHis = np.array([nodispMat,dispMat])

bar.node_ij_mat=np.array([[1,2],
                         [2,3]])
bar.A_vec=np.array([1,1],dtype=np.float64)
bar.E_vec=np.array([1,1],dtype=np.float64)
bar.Initialize(node)
[Fb,Kb]=bar.Solve_FK(node, dispMat)

#print(Fb)
#print(Kb)


cst.node_ijk_mat=np.array([[1,2,3],
                           [1,3,4]])
cst.E_vec=np.array([1,1],dtype=np.float64)
cst.t_vec=np.array([1,1],dtype=np.float64)
cst.v_vec=np.array([0.2,0.2],dtype=np.float64)
cst.Initialize(node)
[Fcst,Kcst]=cst.Solve_FK(node, dispMat)

#print(Fcst)

rs3.node_ijk_mat=np.array([[1,2,3],
                           [1,3,4]])
rs3.rot_spr_K_vec=np.array([1,1])
rs3.Initialize(node)
[Frs3,Krs3]=rs3.Solve_FK(node, dispMat)


rs4.node_ijkl_mat=np.array([[1,2,4,3],
                            [2,1,3,4]])
rs4.rot_spr_K_vec=np.array([1,1],dtype=np.float64)
rs4.Initialize(node)
[Frs4,Krs4]=rs4.Solve_FK(node, dispMat)

#print(Frs4)
#print(Krs4)

assembly=Assembly_KirigamiTruss()
assembly.node = node
assembly.bar = bar
assembly.rotSpr = rs4
assembly.cst = cst

assembly.Initialize_Assembly()

[F,K]=assembly.Solve_FK(dispMat)

remain = F-Fb-Fcst-Frs4
#print(remain)


plotKT=Plot_KirigamiTruss()
plotKT.display_range=2
plotKT.assembly=assembly

plotKT.Plot_Shape_Cst_Number()
plotKT.Plot_Shape_Bar_Number()
plotKT.Plot_Shape_Node_Number()
plotKT.Plot_Shape_Spr_Number()
plotKT.Plot_Deformed_Shape(dispMat)
plotKT.Plot_Deformed_History(dispHis)