import numpy as np
from Elements_Nodes import Elements_Nodes
from Vec_Elements_Bars import Vec_Elements_Bars
from Vec_Elements_RotSprings_4N import Vec_Elements_RotSprings_4N
from Vec_Elements_CST import Vec_Elements_CST
import streamlit as st
from Assembly_KirigamiTruss import Assembly_KirigamiTruss
from Plot_KirigamiTruss import Plot_KirigamiTruss
from Solver_NR_Loading import Solver_NR_Loading

def SolveBridgeDeformation(N,load,view1,view2,barA,L):    
    
    # Define Geometry
    w = 0.1*L
    gap = 0

    node = Elements_Nodes()
    node.coordinates_mat = np.array([
        [-w, 0, 0],
        [-w, L, 0],
        [-w, 0, L],
        [-w, L, L]
    ])

    for i in range(1, N+1):
        base = (w + L) * (i - 1)
        node.coordinates_mat = np.vstack([
            node.coordinates_mat,
            [base, 0, 0], [base, L, 0], [base, 0, L], [base, L, L],
            [base + L/2, 0, 0], [base + L/2, 0, gap],
            [base + L/2, L, 0], [base + L/2, L, gap],
            [base + L/2, 0, L], [base + L/2, gap, L],
            [base + L/2, L, L], [base + L/2, L, L - gap],
            [base + L/2, L/2, 0], [base + L/2, L/2, L],
            [base + L/2, 0, L/2], [base + L/2, L, L/2],
            [base + L, 0, 0], [base + L, L, 0], [base + L, 0, L], [base + L, L, L]
        ])

    node.coordinates_mat = np.vstack([
        node.coordinates_mat,
        [(w + L)*N, 0, 0], [(w + L)*N, L, 0],
        [(w + L)*N, 0, L], [(w + L)*N, L, L]
    ])

    # Define assembly
    assembly = Assembly_KirigamiTruss()
    assembly.node = node
    
    cst = Vec_Elements_CST()
    bar = Vec_Elements_Bars()
    rotSpr = Vec_Elements_RotSprings_4N()
    
    assembly.cst = cst
    assembly.bar = bar
    assembly.rotSpr = rotSpr

    # Define Triangle Elements
    for i in range(N+1):
        if i==0:
            cst.node_ijk_mat=np.array([
                [1, 3, 7], [1, 5, 7], [3, 4, 8], [3, 7, 8],
                [2, 4, 6], [4, 6, 8], [3, 4, 8], [1, 2, 6],
                [1, 5, 6]
            ])  
        else:
            idx = 20*i
            cst.node_ijk_mat=np.vstack([cst.node_ijk_mat,
                [idx+1, idx+3, idx+7], [idx+1, idx+5, idx+7], [idx+3, idx+4, idx+8], [idx+3, idx+7, idx+8],
                [idx+2, idx+4, idx+6], [idx+4, idx+6, idx+8], [idx+3, idx+4, idx+8], [idx+1, idx+2, idx+6],
                [idx+1, idx+5, idx+6]
            ])
    
    for i in range(N):
        
        idx = 20*i
        cst.node_ijk_mat=np.vstack([cst.node_ijk_mat,
            [idx+5, idx+6, idx+17], [idx+5, idx+9, idx+17], [idx+11, idx+6, idx+17],
            [idx+9, idx+21, idx+17], [idx+11, idx+22, idx+17], [idx+21, idx+22, idx+17]
        ])
        
    cstNum = len(cst.node_ijk_mat)
    cst.t_vec = 0.1 * np.ones(cstNum)
    cst.E_vec = 200.0e9 * np.ones(cstNum)
    cst.v_vec = 0.2 * np.ones(cstNum)



    # Define Bars
    for i in range(N):
        if i==0:
            idx=0
            bar.node_ij_mat=np.array([
                [idx+5, idx+10], [idx+19, idx+10], [idx+5, idx+19], [idx+7, idx+13], [idx+7, idx+19], [idx+13, idx+19],
                [idx+13, idx+23], [idx+19, idx+23], [idx+19, idx+21], [idx+10, idx+21],             
                [idx+7, idx+14], [idx+7, idx+18],
                [idx+14, idx+18], [idx+8, idx+15], [idx+15, idx+18], [idx+8, idx+18], [idx+14, idx+23], [idx+18, idx+23],
                [idx+18, idx+24], [idx+15, idx+24], [idx+8, idx+16], [idx+16, idx+20], [idx+8, idx+20], [idx+24, idx+16],
                [idx+24, idx+20], [idx+6, idx+20], [idx+6, idx+12], [idx+12, idx+20], [idx+22, idx+12], [idx+22, idx+20],
                [idx+5, idx+9], [idx+9, idx+17], [idx+5, idx+17], [idx+6, idx+11], [idx+11, idx+17], [idx+6, idx+17],
                [idx+11, idx+22], [idx+17, idx+22], [idx+9, idx+21], [idx+17, idx+21]
            ])
        else:        
            idx = 20*i
            bar.node_ij_mat=np.vstack([bar.node_ij_mat,
                [idx+5, idx+10], [idx+19, idx+10], [idx+5, idx+19], [idx+7, idx+13], [idx+7, idx+19], [idx+13, idx+19],
                [idx+13, idx+23], [idx+19, idx+23], [idx+19, idx+21], [idx+10, idx+21], [idx+7, idx+14], [idx+7, idx+18],
                [idx+14, idx+18], [idx+8, idx+15], [idx+15, idx+18], [idx+8, idx+18], [idx+14, idx+23], [idx+18, idx+23],
                [idx+18, idx+24], [idx+15, idx+24], [idx+8, idx+16], [idx+16, idx+20], [idx+8, idx+20], [idx+24, idx+16],
                [idx+24, idx+20], [idx+6, idx+20], [idx+6, idx+12], [idx+12, idx+20], [idx+22, idx+12], [idx+22, idx+20],
                [idx+5, idx+9], [idx+9, idx+17], [idx+5, idx+17], [idx+6, idx+11], [idx+11, idx+17], [idx+6, idx+17],
                [idx+11, idx+22], [idx+17, idx+22], [idx+9, idx+21], [idx+17, idx+21]
            ])
        
        
        
    barNum = len(bar.node_ij_mat)
    bar.A_vec = barA * np.ones(barNum)
    bar.E_vec = 200e9 * np.ones(barNum)


    # Define Rotational Springs
    for i in range(N):
        if i==0:
            rotSpr.node_ijkl_mat=np.array([
            [20*(i)+5,  20*(i)+1,  20*(i)+7,  20*(i)+2],
            [20*(i)+1,  20*(i)+7,  20*(i)+3,  20*(i)+8],
            [20*(i)+7,  20*(i)+3,  20*(i)+8,  20*(i)+4],
            [20*(i)+3,  20*(i)+4,  20*(i)+8,  20*(i)+6],
            [20*(i)+2,  20*(i)+4,  20*(i)+6,  20*(i)+8],
            [20*(i)+4,  20*(i)+2,  20*(i)+6,  20*(i)+1],
            [20*(i)+2,  20*(i)+6,  20*(i)+1,  20*(i)+5],
            [20*(i)+6,  20*(i)+1,  20*(i)+5,  20*(i)+7],
            [20*(i)+1,  20*(i)+7,  20*(i)+5,  20*(i)+19],
            [20*(i)+5,  20*(i)+7,  20*(i)+19,  20*(i)+13],
            [20*(i)+7,  20*(i)+13,  20*(i)+19,  20*(i)+23],
            [20*(i)+19, 20*(i)+23, 20*(i)+21, 20*(i)+27],
            [20*(i)+5,  20*(i)+10, 20*(i)+19, 20*(i)+21],
            [20*(i)+7,  20*(i)+5,  20*(i)+19, 20*(i)+10],
            [20*(i)+13, 20*(i)+19, 20*(i)+23, 20*(i)+21],
            [20*(i)+10, 20*(i)+19, 20*(i)+21, 20*(i)+23],
            [20*(i)+3,  20*(i)+8,  20*(i)+7,  20*(i)+18],
            [20*(i)+8,  20*(i)+7,  20*(i)+18, 20*(i)+14],
            [20*(i)+7,  20*(i)+8,  20*(i)+18, 20*(i)+15],
            [20*(i)+8,  20*(i)+15, 20*(i)+18, 20*(i)+24],
            [20*(i)+7,  20*(i)+14, 20*(i)+18, 20*(i)+23],
            [20*(i)+14, 20*(i)+18, 20*(i)+23, 20*(i)+24],
            [20*(i)+15, 20*(i)+18, 20*(i)+24, 20*(i)+23],
            [20*(i)+18, 20*(i)+23, 20*(i)+24, 20*(i)+28],
            [20*(i)+4,  20*(i)+6,  20*(i)+8,  20*(i)+20],
            [20*(i)+6,  20*(i)+8,  20*(i)+20, 20*(i)+16],
            [20*(i)+8,  20*(i)+6,  20*(i)+20, 20*(i)+12],
            [20*(i)+8,  20*(i)+16, 20*(i)+20, 20*(i)+24],
            [20*(i)+6,  20*(i)+12, 20*(i)+20, 20*(i)+22],
            [20*(i)+24, 20*(i)+20, 20*(i)+22, 20*(i)+12],
            [20*(i)+16, 20*(i)+20, 20*(i)+24, 20*(i)+22],
            [20*(i)+26, 20*(i)+24, 20*(i)+22, 20*(i)+20],
            [20*(i)+1,  20*(i)+5,  20*(i)+6,  20*(i)+17],
            [20*(i)+6,  20*(i)+5,  20*(i)+17, 20*(i)+9],
            [20*(i)+5,  20*(i)+6,  20*(i)+17, 20*(i)+11],
            [20*(i)+5,  20*(i)+9,  20*(i)+17, 20*(i)+21],
            [20*(i)+6,  20*(i)+17, 20*(i)+11, 20*(i)+22],
            [20*(i)+11, 20*(i)+17, 20*(i)+22, 20*(i)+21],
            [20*(i)+9,  20*(i)+17, 20*(i)+21, 20*(i)+22],
            [20*(i)+17, 20*(i)+21, 20*(i)+22, 20*(i)+26]
            ]) 
        else:
            rotSpr.node_ijkl_mat=np.vstack([rotSpr.node_ijkl_mat,
                    [20*(i)+5,  20*(i)+1,  20*(i)+7,  20*(i)+2],
                    [20*(i)+1,  20*(i)+7,  20*(i)+3,  20*(i)+8],
                    [20*(i)+7,  20*(i)+3,  20*(i)+8,  20*(i)+4],
                    [20*(i)+3,  20*(i)+4,  20*(i)+8,  20*(i)+6],
                    [20*(i)+2,  20*(i)+4,  20*(i)+6,  20*(i)+8],
                    [20*(i)+4,  20*(i)+2,  20*(i)+6,  20*(i)+1],
                    [20*(i)+2,  20*(i)+6,  20*(i)+1,  20*(i)+5],
                    [20*(i)+6,  20*(i)+1,  20*(i)+5,  20*(i)+7],
                    [20*(i)+1,  20*(i)+7,  20*(i)+5,  20*(i)+19],
                    [20*(i)+5,  20*(i)+7,  20*(i)+19,  20*(i)+13],
                    [20*(i)+7,  20*(i)+13,  20*(i)+19,  20*(i)+23],
                    [20*(i)+19, 20*(i)+23, 20*(i)+21, 20*(i)+27],
                    [20*(i)+5,  20*(i)+10, 20*(i)+19, 20*(i)+21],
                    [20*(i)+7,  20*(i)+5,  20*(i)+19, 20*(i)+10],
                    [20*(i)+13, 20*(i)+19, 20*(i)+23, 20*(i)+21],
                    [20*(i)+10, 20*(i)+19, 20*(i)+21, 20*(i)+23],
                    [20*(i)+3,  20*(i)+8,  20*(i)+7,  20*(i)+18],
                    [20*(i)+8,  20*(i)+7,  20*(i)+18, 20*(i)+14],
                    [20*(i)+7,  20*(i)+8,  20*(i)+18, 20*(i)+15],
                    [20*(i)+8,  20*(i)+15, 20*(i)+18, 20*(i)+24],
                    [20*(i)+7,  20*(i)+14, 20*(i)+18, 20*(i)+23],
                    [20*(i)+14, 20*(i)+18, 20*(i)+23, 20*(i)+24],
                    [20*(i)+15, 20*(i)+18, 20*(i)+24, 20*(i)+23],
                    [20*(i)+18, 20*(i)+23, 20*(i)+24, 20*(i)+28],
                    [20*(i)+4,  20*(i)+6,  20*(i)+8,  20*(i)+20],
                    [20*(i)+6,  20*(i)+8,  20*(i)+20, 20*(i)+16],
                    [20*(i)+8,  20*(i)+6,  20*(i)+20, 20*(i)+12],
                    [20*(i)+8,  20*(i)+16, 20*(i)+20, 20*(i)+24],
                    [20*(i)+6,  20*(i)+12, 20*(i)+20, 20*(i)+22],
                    [20*(i)+24, 20*(i)+20, 20*(i)+22, 20*(i)+12],
                    [20*(i)+16, 20*(i)+20, 20*(i)+24, 20*(i)+22],
                    [20*(i)+26, 20*(i)+24, 20*(i)+22, 20*(i)+20],
                    [20*(i)+1,  20*(i)+5,  20*(i)+6,  20*(i)+17],
                    [20*(i)+6,  20*(i)+5,  20*(i)+17, 20*(i)+9],
                    [20*(i)+5,  20*(i)+6,  20*(i)+17, 20*(i)+11],
                    [20*(i)+5,  20*(i)+9,  20*(i)+17, 20*(i)+21],
                    [20*(i)+6,  20*(i)+17, 20*(i)+11, 20*(i)+22],
                    [20*(i)+11, 20*(i)+17, 20*(i)+22, 20*(i)+21],
                    [20*(i)+9,  20*(i)+17, 20*(i)+21, 20*(i)+22],
                    [20*(i)+17, 20*(i)+21, 20*(i)+22, 20*(i)+26]
            ]) 
            
    # Add final rotSpr set for end cap
    rotSpr.node_ijkl_mat=np.vstack([rotSpr.node_ijkl_mat,
                    [20*N+5,  20*N+1,  20*N+7,  20*N+2],
                    [20*N+1,  20*N+7,  20*N+3,  20*N+8],
                    [20*N+7,  20*N+3,  20*N+8,  20*N+4],
                    [20*N+3,  20*N+4,  20*N+8,  20*N+6],
                    [20*N+2,  20*N+4,  20*N+6,  20*N+8],
                    [20*N+4,  20*N+2,  20*N+6,  20*N+1],
                    [20*N+2,  20*N+6,  20*N+1,  20*N+5],
                    [20*N+6,  20*N+1,  20*N+5,  20*N+7]])
        
    rotNum = len(rotSpr.node_ijkl_mat)
    rotSpr.rot_spr_K_vec = 10000.0 * np.ones(rotNum)
    
    
    factor = 1000
    for i in range(N+1):
        for offset in [1, 3, 5, 7]:
            rotSpr.rot_spr_K_vec[i*40 + offset] *= factor        
            
    # Initialize Assembly
    assembly.Initialize_Assembly()                
            
    # Define Plotting
    plots = Plot_KirigamiTruss()
    plots.assembly = assembly
    
    plots.display_range = np.array([-0.2*L, L*(N)*1.2, -0.2*L, 1.2*L, -0.2*L, 1.2*L])
    plots.view_angle1=view1
    plots.view_angle2=view2
    
    plots.Plot_Shape_Node_Number()
    plots.Plot_Shape_Cst_Number()
    plots.Plot_Shape_Bar_Number()        
    plots.Plot_Shape_Spr_Number()    
   
    # Solver Setup
    nr = Solver_NR_Loading()
    nr.assembly = assembly
    nr.supp = [[0,1,1,1],[1,1,1,1],[20*N+5-1,1,1,1],[20*N+6-1,1,1,1]]

    
    nr.incre_step = 1
    nr.iter_max = 20
    nr.tol = 1e-8
    
    nr.load=np.array([[20*N/2+5-1, 0, 0, -load/4],
             [20*N/2+6-1, 0, 0, -load/4],
             [20*N/2+1-1, 0, 0, -load/4],
             [20*N/2+2-1, 0, 0, -load/4]])
        
    
    Uhis = nr.Solve()
    fig1=plots.Plot_Deformed_Shape(10*Uhis[-1])    
    fig2=plots.Plot_Bar_Stress(Uhis[-1])
    
    return fig1,fig2

    # plots.fileName = 'OrigamiTruss_deploy.gif'
    # plots.plot_deformed_history(Uhis[::10])

st.subheader("Simulate motion and load-carrying of deployable bridge")

st.text('Developer: Dr. Yi Zhu')

st.text('This is a demo for using the Sim-FAST package to simulate the deployment ' + 
        'and load carrying capacity of kirigami truss bridges. We assume that ' +
        'connections are rigid, all members share the same cross-section, and ' +
        'ignore buckling related failure mode when calculating the loading.')

st.subheader("Setting up the deployable bridge")

N = st.selectbox(
     "Select a number for sections:",
     [2,4,6,8,10,12])

barA = st.selectbox("Bar Area (m2)):",         
     [0.0001, 0.0004, 0.001, 0.004, 0.01, 0.04])

L = st.selectbox(
     "Length of the sections (m):",
     [1.0,1.5,2.0,2.5,3.0])

st.text('Here we quickly set up the deployable kirigami truss bridge by picking ' +
        'the number of sections and truss areas. The following pre-simulated ' +
        'GIF will show the deployment kinematics. This GIF shows a bridge with.' +
        'a section length to be 1 meter')

if N == 2:
    st.image("Kirigami_Truss_2Sec_Deploy.gif")
elif N == 4:
    st.image("Kirigami_Truss_4Sec_Deploy.gif")
elif N == 6:
    st.image("Kirigami_Truss_6Sec_Deploy.gif")
elif N == 8:
    st.image("Kirigami_Truss_8Sec_Deploy.gif")
elif N == 10:
    st.image("Kirigami_Truss_10Sec_Deploy.gif")
elif N == 12:
    st.image("Kirigami_Truss_12Sec_Deploy.gif")


st.subheader("Load-carrying simulation of the deployable bridge")

st.text('Here we quickly set up the loading of the bridge. ' + 
        'We assume that the bridge is simply supported at both ends, ' +
        'and the bridge is loaded at the mid-span with a concentrated load. ' +
        'The following figures show the loading results. Deformation is ' +
        'Scaled up by 10 time when plotting. Self weight is neglected.')

load = st.selectbox(
     "Applied Loads (kN):",
     [100.0, 200.0, 300.0, 400.0, 500.0, 600.0])
load=load*1000

view1 = st.slider("View angle 1:",         
        min_value=0.0,
        max_value=90.0,
        value=15.0,
        step=5.0)

view2 = st.slider("View angle 2:",         
        min_value=0.0,
        max_value=180.0,
        value=70.0,
        step=5.0)

fig1,fig2=SolveBridgeDeformation(N,load,view1,view2,barA,L)

st.pyplot(fig1)
st.pyplot(fig2)



