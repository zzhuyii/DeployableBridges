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

from Origami_Bridge_Strength_During_Deploy import origami_deploy


st.subheader("Simulation of of deployable bridge")

st.text('This is a demo for using the Sim-FAST package to simulate the deployment ' + 
        'and load carrying capacity of different deployable bridges. We assume that ' +
        'connections are rigid, all members share the same cross-section, and ' +
        'ignore buckling related failure mode when calculating the loading.' +
        'For more detailed control of the simulation, please find the MATLAB ' +
        'code from: https://github.com/zzhuyii/Sim-FAST')

st.subheader("Setting up the deployable bridge")

BridgeType = st.selectbox("Select type of deployable bridges:",['kirigami','origami','scissor','improved scissor','rolling'])

if BridgeType =='origami':
    N = st.selectbox(
         "Select a number for sections:",
         [2,3,4])
else:
    N = st.selectbox(
         "Select a number for sections:",
         [2,4,6,8])

L = 2.0

st.subheader("Strength during deployment")

st.text('Here, we check the truss strength during the deployment process.' + 
        'You can change deployment ratio for different stage of deployment' +
        'The load applied is the bridge self-weight with AASHTO factors.')

DepRate = st.selectbox(
     "Deployment Ratio:",
     [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])


if BridgeType=='kirigami':        
    fig1,fig2=kirigami_deploy(L, N, DepRate)    
    # st.image("Kirigami_Truss_Strength_During_Deploy_Bar_Failure.png")
    # st.image("Kirigami_Truss_Strength_During_Deploy_Bar_Stress.png")    
    st.pyplot(fig1)
    st.pyplot(fig2)
elif BridgeType=='origami':   
    fig1,fig2=origami_deploy(L, N, DepRate)    
    # st.image("Kirigami_Truss_Strength_During_Deploy_Bar_Failure.png")
    # st.image("Kirigami_Truss_Strength_During_Deploy_Bar_Stress.png")    
    st.pyplot(fig1)
    st.pyplot(fig2)


st.subheader("Load to failure after deployment")

st.text('Here we load the bridge all the way to failure. We will study' + 
        ' the capacity of the bridge and the efficiency of the bridge.' +
        ' When bridge is short, failure may not happen after the software ' +
        'reach the maximum 100 step for loading. Because an incremental ' +
        'iterative nonlinear loading solver is used, finding ultimate ' +
        'load can be slower.')

if BridgeType=='kirigami':        
    fig1,fig2=kirigami_fail(L, N)    
    # st.image("Kirigami_Truss_Strength_During_Deploy_Bar_Failure.png")
    # st.image("Kirigami_Truss_Strength_During_Deploy_Bar_Stress.png")    
    st.pyplot(fig1)
    st.pyplot(fig2)




