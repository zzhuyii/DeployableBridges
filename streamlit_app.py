import streamlit as st

from Kirigami_Truss_Strength_During_Deploy import kirigami_deploy
from Kirigami_Truss_Load_To_Fail import kirigami_fail

from Origami_Bridge_Strength_During_Deploy import origami_deploy
from Origami_Bridge_Load_To_Fail import origami_fail

from Scissor_Bridge_Strength_During_Deploy import scissor_deploy
from Scissor_Bridge_Load_To_Fail import scissor_fail

from Scissor_Bridge_2_Strength_During_Deploy import improvedScissor_deploy
from Scissor_Bridge_2_Load_To_Fail import improvedScissor_fail

from Rolling_Bridge_Strength_During_Deploy import rolling_deploy
from Rolling_Bridge_Load_To_Fail import rolling_fail


st.subheader("Simulation of of deployable bridge")

st.text('Developer: Zhongqi Fan & Yi Zhu')

st.text('This is a demo for using the Sim-FAST package to simulate the deployment ' + 
        'and load carrying capacity of different deployable bridges. We assume that ' +
        'connections are rigid, all members share the same cross-section, and ' +
        'ignore buckling related failure mode when calculating the loading.' )

st.markdown("Please find the MATLAB code from: https://github.com/zzhuyii/Sim-FAST")

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
    fig1,fig2,tip=kirigami_deploy(L, N, DepRate)    
    # st.image("Kirigami_Truss_Strength_During_Deploy_Bar_Failure.png")
    # st.image("Kirigami_Truss_Strength_During_Deploy_Bar_Stress.png")   
    st.write("The tip deflection of kirigami bridge in meter unit is:", tip)
    st.pyplot(fig1)
    st.pyplot(fig2)
elif BridgeType=='origami':   
    fig1,fig2,tip=origami_deploy(L, N, DepRate)    
    # st.image("Kirigami_Truss_Strength_During_Deploy_Bar_Failure.png")
    # st.image("Kirigami_Truss_Strength_During_Deploy_Bar_Stress.png")  
    st.write("The tip deflection of origami bridge in meter unit is:", tip)
    st.pyplot(fig1)
    st.pyplot(fig2)
elif BridgeType=='scissor':   
    fig1,fig2,tip=scissor_deploy(N, DepRate)    
    # st.image("Kirigami_Truss_Strength_During_Deploy_Bar_Failure.png")
    # st.image("Kirigami_Truss_Strength_During_Deploy_Bar_Stress.png")    
    st.write("The tip deflection of scissor bridge in meter unit is:", tip)
    st.pyplot(fig1)
    st.pyplot(fig2)
elif BridgeType=='improved scissor':   
    fig1,fig2,tip=improvedScissor_deploy(N, DepRate)    
    # st.image("Kirigami_Truss_Strength_During_Deploy_Bar_Failure.png")
    # st.image("Kirigami_Truss_Strength_During_Deploy_Bar_Stress.png")   
    st.write("The tip deflection of improved scissor bridge in meter unit is:", tip)
    st.pyplot(fig1)
    st.pyplot(fig2)    
elif BridgeType=='rolling':   
    fig1,fig2,tip=rolling_deploy(N, DepRate)    
    # st.image("Kirigami_Truss_Strength_During_Deploy_Bar_Failure.png")
    # st.image("Kirigami_Truss_Strength_During_Deploy_Bar_Stress.png")  
    st.write("The tip deflection of rolling bridge in meter unit is:", tip)
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
    fig1, fig2, F, Weight =kirigami_fail(L, N)    
    # st.image("Kirigami_Truss_Strength_During_Deploy_Bar_Failure.png")
    # st.image("Kirigami_Truss_Strength_During_Deploy_Bar_Stress.png")  
    st.write("The the maximum load kirigami bridge can carry is:", F/1000, "kN")
    st.write("The the load over self-weight of kirigami bridge can carry is:", F/Weight)
    st.pyplot(fig1)
    st.pyplot(fig2)
elif BridgeType=='origami':   
    fig1,fig2, F, Weight=origami_fail(L, N)    
    # st.image("Kirigami_Truss_Strength_During_Deploy_Bar_Failure.png")
    # st.image("Kirigami_Truss_Strength_During_Deploy_Bar_Stress.png")  
    st.write("The the maximum load origami bridge can carry is:", F/1000, "kN")
    st.write("The the load over self-weight of origami bridge can carry is:", F/Weight)
    st.pyplot(fig1)
    st.pyplot(fig2)
elif BridgeType=='scissor':   
    fig1,fig2, F, Weight=scissor_fail(N)    
    # st.image("Kirigami_Truss_Strength_During_Deploy_Bar_Failure.png")
    # st.image("Kirigami_Truss_Strength_During_Deploy_Bar_Stress.png")   
    st.write("The the maximum load scissor bridge can carry is:", F/1000, "kN")
    st.write("The the load over self-weight of scissor bridge can carry is:", F/Weight)
    st.pyplot(fig1)
    st.pyplot(fig2)
elif BridgeType=='improved scissor':   
    fig1,fig2, F, Weight=improvedScissor_fail(N)    
    # st.image("Kirigami_Truss_Strength_During_Deploy_Bar_Failure.png")
    # st.image("Kirigami_Truss_Strength_During_Deploy_Bar_Stress.png")    
    st.write("The the maximum load improved scissor bridge can carry is:", F/1000, "kN")
    st.write("The the load over self-weight of improved scissor bridge can carry is:", F/Weight)
    st.pyplot(fig1)
    st.pyplot(fig2)   
elif BridgeType=='rolling':   
    fig1,fig2, F, Weight=rolling_fail(N)    
    # st.image("Kirigami_Truss_Strength_During_Deploy_Bar_Failure.png")
    # st.image("Kirigami_Truss_Strength_During_Deploy_Bar_Stress.png") 
    st.write("The the maximum load rolling bridge can carry is:", F/1000, "kN")
    st.write("The the load over self-weight of rolling bridge can carry is:", F/Weight)
    st.pyplot(fig1)
    st.pyplot(fig2)   


