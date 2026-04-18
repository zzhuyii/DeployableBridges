import numpy as np
from Elements_Nodes import Elements_Nodes
from Vec_Elements_Bars import Vec_Elements_Bars
from Vec_Elements_RotSprings_4N import Vec_Elements_RotSprings_4N
from Vec_Elements_CST import Vec_Elements_CST
import streamlit as st
from Assembly_KirigamiTruss import Assembly_KirigamiTruss
from Plot_KirigamiTruss import Plot_KirigamiTruss
from Solver_NR_Loading import Solver_NR_Loading

from Kirigami_Truss_Strength_During_Deploy import kirigami_deploy
from Kirigami_Truss_Load_To_Fail import kirigami_fail

st.subheader("Simulation of of deployable bridge")

st.text('This is a demo for using the Sim-FAST package to simulate the deployment ' + 
        'and load carrying capacity of different deployable bridges. We assume that ' +
        'connections are rigid, all members share the same cross-section, and ' +
        'ignore buckling related failure mode when calculating the loading.' +
        'For more detailed control of the simulation, please find the MATLAB' +
        'code from: https://github.com/zzhuyii/Sim-FAST')

st.subheader("Setting up the deployable bridge")

BridgeType = st.selectbox("Select type of deployable bridges:",['kirigami','origami','scissor','improved scissor','rolling'])


N = st.selectbox(
     "Select a number for sections:",
     [2,4,6,8])

barA = st.selectbox("Bar Area (m2)):",         
     [0.0001, 0.0004, 0.001, 0.004, 0.01, 0.04])

L = st.selectbox(
     "Length of the sections (m):",
     [1.0,1.5,2.0,2.5,3.0])

st.subheader("Strength during deployment")

st.text('Here we quickly set up the deployable kirigami truss bridge by picking ' +
        'the number of sections and truss areas. The following pre-simulated ' +
        'GIF will show the deployment kinematics. This GIF shows a bridge with.' +
        'a section length to be 1 meter')

if BridgeType=='kirigami':        
    fig1,fig2=kirigami_deploy(L, N,)    
    # st.image("Kirigami_Truss_Strength_During_Deploy_Bar_Failure.png")
    # st.image("Kirigami_Truss_Strength_During_Deploy_Bar_Stress.png")    
    st.pyplot(fig1)
    st.pyplot(fig2)


st.subheader("Load to failure after deployment")

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





