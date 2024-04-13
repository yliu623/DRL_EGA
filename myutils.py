""" Utility Functions """
import numpy as np

#################################################################### Simulate OTC prices #######################################
def get_OTC_path(T_training, np_seed, T_testing, nmax):
    """
    Generating OTC prices for later simulation

    Parameters
    ----------
    nmax :  
        number of paths.  
    T_training :  
        Training period. 
    T_testing :  
        Testing period. 

    Returns
    -------
    p_c : contract price.
    p_2 : open market price for regular-weight hogs.
    p_1 : open market price for under-weight hogs.
    c_h : holding cost.
    c_p : penalty cost.
    W1 : under-weight hogs quantity.
    W2 : regular-weight hogs quantity.
    """


    q = 98107
    alpha = 0.154
    beta = 0.02
    Twarmup = 101
    start_training = 10    
    Tmax = T_training + start_training + T_testing
    #simulate factor prices
    W0_factor = np.array([[-0.050083, -0.13667, 0.0020002], 
                          [0.62053, 0.36688, 0.8887],
                          [0.52539, 0.39767, -0.70058],
                          [4.6059, 3.7998, 0.05894]]) 
                           
    #simulate OTC prices
    W0_OTC = np.array([[14.617, -34.846, -8.940, 1.037, -0.089], 
                      [0.165, -0.274, -0.183, 0.806, -0.183], 
                      [0.358, 0.342, 0.454, -0.253, 0.454],
                      [0.11, 0.006, -0.027, 0.026, -0.027], 
                      [0.876, 2.207, 2.284, 0.005, 0.023], 
                      [0.673, 0.471, 0.328, -0.007, 0.003], 
                      [0.237, 1.101, 0.191, 0.001, 0.002],
                      [4.839, 47.704, 16.58, 0.020227, 0.0016579]])              
    
    W1 = np.zeros((nmax,Tmax))
    W2 = np.zeros((nmax,Tmax))
    
    def bimodal(n):
        mu1 = 24270.46318226
        sigma1 = np.sqrt(28405922.00056706)
        mu2 = 36815.52113443
        sigma2 = np.sqrt(16641717.58409381) 
              
        weight1 = 0.62629896 # Proportion of samples from first distribution
        weight2 = 0.37370103 # Proportion of samples from second distribution

        n1 = round(n*weight1) 
        n2 = round(n*weight2) 
    
        X = np.concatenate((np.random.normal(mu1, sigma1, n1), np.random.normal(mu2, sigma2, n2))) # sigma is the standard deviation
        X[X<0] = 0
        return X
    np.random.seed(np_seed)
    for T in range(Tmax-1, Tmax):
        pi_o = np.zeros((nmax,T+1)) 
        pi_m = np.zeros((nmax,T+1))
        pi_f = np.zeros((nmax,T+1))
        p_c = np.zeros((nmax,T+1))
        p_2 = np.zeros((nmax,T+1))
        p_1 = np.zeros((nmax,T+1))
        c_h = np.zeros((nmax,T+1))
        c_p = np.zeros((nmax,T+1))
        
        tempPi_O = np.zeros((1,T+1))
        tempPi_M = np.zeros((1,T+1))
        tempPi_F = np.zeros((1,T+1))               
        for n in range(nmax):
            gen_cont = True
            while gen_cont:
                w1_gen = bimodal(T+1)
                w2_gen = np.random.normal(62906.31683,9551.438123,T+1)
                if w2_gen.all() + w1_gen.all() <= 150000:
                    W1[n,:] = w1_gen 
                    W2[n,:] = w2_gen 
                    gen_cont = False

            # Warm up factor price
            u_O = np.random.normal(0,np.sqrt(W0_factor[3,0]),Twarmup+53)
            u_M = np.random.normal(0,np.sqrt(W0_factor[3,1]),Twarmup+53)
            u_F = np.random.normal(0,np.sqrt(W0_factor[3,2]),Twarmup+53)
            Pi_O = np.zeros(Twarmup+53) #warm-up period factor market price
            Pi_M = np.zeros(Twarmup+53)
            Pi_F = np.zeros(Twarmup+53)
            
            for t in range(2,len(u_O)):
                Pi_O[t] = W0_factor[0,0] + (1 + W0_factor[1,0])*Pi_O[t-1] - W0_factor[1,0]*Pi_O[t-2] + u_O[t] + W0_factor[2,0]*u_O[t-1]
            
            for t in range(2,len(u_M)):
                Pi_M[t] = W0_factor[0,1] + (1 + W0_factor[1,1])*Pi_M[t-1] - W0_factor[1,1]*Pi_M[t-2] + u_M[t] + W0_factor[2,1]*u_M[t-1]
            
            for t in range(2,len(u_F)):
                Pi_F[t] = W0_factor[0,2] + (1 + W0_factor[1,2])*Pi_F[t-1] - W0_factor[1,2]*Pi_F[t-2] + u_F[t] + W0_factor[2,2]*u_F[t-1]
            
            Pi_O[Pi_O<0]=0
            Pi_M[Pi_M<0]=0
            Pi_F[Pi_F<0]=0
            
            #warm up OTC prices
            w_C = np.random.normal(0,np.sqrt(W0_OTC[7,0]),Twarmup+53)
            w_1 = np.random.normal(0,np.sqrt(W0_OTC[7,1]),Twarmup+53)
            w_2 = np.random.normal(0,np.sqrt(W0_OTC[7,2]),Twarmup+53)
            w_H = np.random.normal(0,np.sqrt(W0_OTC[7,3]),Twarmup+53)
            w_P = np.random.normal(0,np.sqrt(W0_OTC[7,4]),Twarmup+53)
            P_C = np.zeros(Twarmup+53)  #OTC prices for warm-up period
            P_1 = np.zeros(Twarmup+53)  
            P_2 = np.zeros(Twarmup+53)  
            C_H = np.zeros(Twarmup+53)  
            C_P = np.zeros(Twarmup+53) 
            
            for t in range(53,len(w_C)):        
                P_C[t] = W0_OTC[0,0] + P_C[t-1]*W0_OTC[1,0] + Pi_O[t]*W0_OTC[4,0] + Pi_M[t]*W0_OTC[5,0] + Pi_F[t]*W0_OTC[6,0] + w_C[t] + W0_OTC[3,0]*w_C[t-52] + W0_OTC[2,0]*w_C[t-1] + W0_OTC[2,0]*W0_OTC[3,0]*w_C[t-53]
            
            for t in range(53,len(w_1)):
                P_1[t] = W0_OTC[0,1] + P_1[t-1]*W0_OTC[1,1] + Pi_O[t]*W0_OTC[4,1] + Pi_M[t]*W0_OTC[5,1] + Pi_F[t]*W0_OTC[6,1] + w_1[t] + W0_OTC[3,1]*w_1[t-52] + W0_OTC[2,1]*w_1[t-1] + W0_OTC[2,1]*W0_OTC[3,1]*w_1[t-53]
            
            for t in range(53,len(w_2)):
                P_2[t] = W0_OTC[0,2] + P_2[t-1]*W0_OTC[1,2] + Pi_O[t]*W0_OTC[4,2] + Pi_M[t]*W0_OTC[5,2] + Pi_F[t]*W0_OTC[6,2] + w_2[t] + W0_OTC[3,2]*w_2[t-52] + W0_OTC[2,2]*w_2[t-1] + W0_OTC[2,2]*W0_OTC[3,2]*w_2[t-53]
            
            for t in range(53,len(w_H)):
                C_H[t] = W0_OTC[0,3] + C_H[t-1]*W0_OTC[1,3] + Pi_O[t]*W0_OTC[4,3] + Pi_M[t]*W0_OTC[5,3] + Pi_F[t]*W0_OTC[6,3] + w_H[t] + W0_OTC[3,3]*w_H[t-52] + W0_OTC[2,3]*w_H[t-1] + W0_OTC[2,3]*W0_OTC[3,3]*w_H[t-53]
                
            for t in range(53,len(w_P)):
                C_P[t] = W0_OTC[0,4] + C_P[t-1]*W0_OTC[1,4] + Pi_O[t]*W0_OTC[4,4] + Pi_M[t]*W0_OTC[5,4] + Pi_F[t]*W0_OTC[6,4] + w_P[t] + W0_OTC[3,4]*w_P[t-52] + W0_OTC[2,4]*w_P[t-1] + W0_OTC[2,4]*W0_OTC[3,4]*w_P[t-53]
            
            P_C[P_C<0]=0
            P_1[P_1<0]=0
            P_2[P_2<0]=0
            C_H[C_H<0]=0
            C_P[C_P<0]=0
            # Simulate prices
            # Factor prices first period
            pi_o[n,0] = Pi_O[-1]
            pi_m[n,0] = Pi_M[-1]
            pi_f[n,0] = Pi_F[-1]
            # OTC prices fist period             
            p_c[n,0] = P_C[-1]
            p_2[n,0] = P_2[-1]
            p_1[n,0] = P_1[-1]
            c_h[n,0] = C_H[-1]
            c_p[n,0] = C_P[-1]
    
            # generate 2nd to 100 period prices
            W_factor = np.array([[-0.050083, 1+0.62053, -0.62053],
                        [-0.13667, 1+0.36688, -0.36688],
                        [0.0020002, 1+0.8887, -0.8887]])
            tempPi_O[0,0] = W_factor[0,:]@np.array([1,Pi_O[-1],0]) #inner product
            tempPi_M[0,0] = W_factor[1,:]@np.array([1, Pi_M[-1], 0])
            tempPi_F[0,0] = W_factor[2,:]@np.array([1, Pi_F[-1], 0])
            tempPi_O[tempPi_O<0]=0
            tempPi_M[tempPi_M<0]=0
            tempPi_F[tempPi_F<0]=0
            #generate forecast for the OTC 
            W_OTC = np.array([[14.617, 0.16547, 0.87606, 0.67273, 0.23732], #P_C
                      [-8.9395, -0.18333, 2.2838, 0.32753, 0.19068], #P_2
                      [-34.846, -0.27437, 2.2071, 0.47132, 1.1006], #P_1
                      [1.0374, 0.80573, 0.004734, -0.0065202, 0.0012593], #C_H
                      [-0.089304, -0.18324, 0.022836, 0.0032739, 0.0019056]]) #C_P
            
            for i in range(1,T):   
                #Factor prices DGP
                inno_Pi_O = np.random.normal(0,np.sqrt(4.6059),T+1) 
                inno_Pi_M = np.random.normal(0,np.sqrt(3.7998),T+1) 
                inno_Pi_F = np.random.normal(0,np.sqrt(0.05894),T+1) 
                MA_factor = np.array([0.52539, 0.39767, -0.70058]) 
                if i == 1:
                    pi_o[n,i] = W_factor[0,:]@np.hstack([1, pi_o[n,i-1], 0]) + inno_Pi_O[i] + MA_factor[0]*inno_Pi_O[i-1]
                    pi_m[n,i] = W_factor[1,:]@np.hstack([1, pi_m[n,i-1], 0]) + inno_Pi_M[i] + MA_factor[1]*inno_Pi_M[i-1]
                    pi_f[n,i] = W_factor[2,:]@np.hstack([1, pi_f[n,i-1], 0]) + inno_Pi_F[i] + MA_factor[2]*inno_Pi_F[i-1]
                else:
                    pi_o[n,i] = W_factor[0,:]@np.hstack([1, pi_o[n,i-1], pi_o[n,i-2]])+ inno_Pi_O[i] + MA_factor[0]*inno_Pi_O[i-1]
                    pi_m[n,i] = W_factor[1,:]@np.hstack([1, pi_m[n,i-1], pi_m[n,i-2]]) + inno_Pi_M[i] + MA_factor[1]*inno_Pi_M[i-1]
                    pi_f[n,i] = W_factor[2,:]@np.hstack([1, pi_f[n,i-1], pi_f[n,i-2]]) + inno_Pi_F[i] + MA_factor[2]*inno_Pi_F[i-1]
                  
                pi_o[pi_o<0]=0 
                pi_m[pi_m<0]=0
                pi_f[pi_f<0]=0
                # generate forecast/expectation for the next period's factor price, t=2    
                tempPi_O[0,i] = W_factor[0,:]@np.array([1, pi_o[n,i], pi_o[n,i-1]])#forecast for the ith period
                tempPi_M[0,i] = W_factor[1,:]@np.array([1, pi_m[n,i], pi_m[n,i-1]]) 
                tempPi_F[0,i] = W_factor[2,:]@[1, pi_f[n,i], pi_f[n,i-1]] 
                tempPi_O[tempPi_O<0]=0 
                tempPi_M[tempPi_M<0]=0
                tempPi_F[tempPi_F<0]=0
                #OTC prices DGP             
                #generate forecast for the OTC
                inno_P_C = np.random.normal(0,np.sqrt(4.839),T+52) 
                inno_P_2 = np.random.normal(0,np.sqrt(16.58),T+52) 
                inno_P_1 = np.random.normal(0,np.sqrt(47.704),T+52) 
                inno_C_H = np.random.normal(0,np.sqrt(0.020227),T+52) 
                inno_C_P = np.random.normal(0,np.sqrt(0.0016579),T+52) 
                MA_OTC = np.array([0.35802,  0.45427,  0.34237,  -0.25309,  0.45424]) 
                SMA_OTC = np.array([0.11019,  -0.027154,  0.0063966,  0.024557,  -0.027257])
                      
                p_c[n,i] = W_OTC[0,:]@np.hstack([1, p_c[n,i-1],  tempPi_O[:,i], tempPi_M[:,i], tempPi_F[:,i]]) + inno_P_C[i+52] + MA_OTC[0]*inno_P_C[i+51] + SMA_OTC[0]*inno_P_C[i] + MA_OTC[0]*SMA_OTC[0]*inno_P_C[i-1] #P_C
                p_2[n,i] = W_OTC[1,:]@np.hstack([1, p_2[n,i-1],  tempPi_O[:,i], tempPi_M[:,i], tempPi_F[:,i]]) + inno_P_2[i+52] + MA_OTC[1]*inno_P_2[i+51] + SMA_OTC[1]*inno_P_2[i] + MA_OTC[1]*SMA_OTC[1]*inno_P_2[i-1] #P_2
                p_1[n,i] = W_OTC[2,:]@np.hstack([1, p_1[n,i-1],  tempPi_O[:,i], tempPi_M[:,i], tempPi_F[:,i]]) + inno_P_1[i+52] + MA_OTC[2]*inno_P_1[i+51] + SMA_OTC[2]*inno_P_1[i] + MA_OTC[2]*SMA_OTC[2]*inno_P_1[i-1] #P_1
                c_h[n,i] = W_OTC[3,:]@np.hstack([1, c_h[n,i-1],  tempPi_O[:,i], tempPi_M[:,i], tempPi_F[:,i]]) + inno_C_H[i+52] + MA_OTC[3]*inno_C_H[i+51] + SMA_OTC[3]*inno_C_H[i] + MA_OTC[3]*SMA_OTC[3]*inno_C_H[i-1] #C_H
                c_p[n,i] = W_OTC[4,:]@np.hstack([1, c_p[n,i-1],  tempPi_O[:,i], tempPi_M[:,i], tempPi_F[:,i]]) + inno_C_P[i+52] + MA_OTC[4]*inno_C_P[i+51] + SMA_OTC[4]*inno_C_P[i] + MA_OTC[4]*SMA_OTC[4]*inno_C_P[i-1] #C_P
                p_c[p_c<0]=0
                p_2[p_2<0]=0
                p_1[p_1<0]=0
                c_h[c_h<0]=0
                c_p[c_p<0]=0       
            
            ############################################################################################################################# 
            #terminal DGP          
            if T == 1:
                pi_o[n,T] = W_factor[0,:]@np.hstack([1, pi_o[n,T-1], 0]) 
                pi_m[n,T] = W_factor[1,:]@np.hstack([1, pi_m[n,T-1], 0])
                pi_f[n,T] = W_factor[2,:]@np.hstack([1, pi_f[n,T-1], 0])  
                pi_o[pi_o<0]=0 
                pi_m[pi_m<0]=0 
                pi_f[pi_f<0]=0 
            else:
                pi_o[n,T] = W_factor[0,:]@np.hstack([1, pi_o[n,T-1], pi_o[0,T-2]]) + inno_Pi_O[T] + MA_factor[0]*inno_Pi_O[T-1] 
                pi_m[n,T] = W_factor[1,:]@np.hstack([1, pi_m[n,T-1], pi_m[0,T-2]]) + inno_Pi_M[T] + MA_factor[1]*inno_Pi_M[T-1] 
                pi_f[n,T] = W_factor[2,:]@np.hstack([1, pi_f[n,T-1], pi_f[0,T-2]]) + inno_Pi_F[T] + MA_factor[2]*inno_Pi_F[T-1]    
                pi_o[pi_o<0]=0  
                pi_m[pi_m<0]=0
                pi_f[pi_o<0]=0
            #Forecast for last period 
            tempPi_O[0,T] = W_factor[0,:]@np.hstack([1, pi_o[n,T], pi_o[n,T-1]]) #forecast for the ith period
            tempPi_M[0,T] = W_factor[1,:]@np.hstack([1, pi_m[n,T], pi_m[n,T-1]])
            tempPi_F[0,T] = W_factor[2,:]@np.stack([1, pi_f[n,T], pi_f[n,T-1]])
            tempPi_O[tempPi_O<0]=0 
            tempPi_M[tempPi_M<0]=0 
            tempPi_F[tempPi_F<0]=0 
            
            inno_P_C = np.random.normal(0,np.sqrt(4.839),T+52+1) 
            inno_P_2 = np.random.normal(0,np.sqrt(16.58),T+52+1) 
            inno_P_1 = np.random.normal(0,np.sqrt(47.704),T+52+1) 
            inno_C_H = np.random.normal(0,np.sqrt(0.020227),T+52+1) 
            inno_C_P = np.random.normal(0,np.sqrt(0.0016579),T+52+1) 
            MA_OTC = np.array([0.35802,  0.45427,  0.34237,  -0.25309,  0.45424]) 
            SMA_OTC = np.array([0.11019,  -0.027154,  0.0063966,  0.024557,  -0.027257])
                
            p_c[n,T] = W_OTC[0,:]@np.hstack([1, p_c[n,T-1],  tempPi_O[:,T], tempPi_M[:,T], tempPi_F[:,T]]) + inno_P_C[T+52] + MA_OTC[0]*inno_P_C[T+51] + SMA_OTC[0]*inno_P_C[T] + MA_OTC[0]*SMA_OTC[0]*inno_P_C[T-1]  #P_C
            p_2[n,T] = W_OTC[1,:]@np.hstack([1, p_2[n,T-1],  tempPi_O[:,T], tempPi_M[:,T], tempPi_F[:,T]]) + inno_P_2[T+52] + MA_OTC[1]*inno_P_2[T+51] + SMA_OTC[1]*inno_P_2[T] + MA_OTC[1]*SMA_OTC[1]*inno_P_2[T-1]  #P_2
            p_1[n,T] = W_OTC[2,:]@np.hstack([1, p_1[n,T-1],  tempPi_O[:,T], tempPi_M[:,T], tempPi_F[:,T]]) + inno_P_1[T+52] + MA_OTC[2]*inno_P_1[T+51] + SMA_OTC[2]*inno_P_1[T] + MA_OTC[2]*SMA_OTC[2]*inno_P_1[T-1]  #P_1
            c_h[n,T] = W_OTC[3,:]@np.hstack([1, c_h[n,T-1],  tempPi_O[:,T], tempPi_M[:,T], tempPi_F[:,T]]) + inno_C_H[T+52] + MA_OTC[3]*inno_C_H[T+51] + SMA_OTC[3]*inno_C_H[T] + MA_OTC[3]*SMA_OTC[3]*inno_C_H[T-1]  #C_H
            c_p[n,T] = W_OTC[4,:]@np.hstack([1, c_p[n,T-1],  tempPi_O[:,T], tempPi_M[:,T], tempPi_F[:,T]]) + inno_C_P[T+52] + MA_OTC[4]*inno_C_P[T+51] + SMA_OTC[4]*inno_C_P[T] + MA_OTC[4]*SMA_OTC[4]*inno_C_P[T-1]  #C_P
            p_c[p_c<0]=0 
            p_2[p_2<0]=0 
            p_1[p_1<0]=0 
            c_h[c_h<0]=0 
            c_p[c_p<0]=0 
            

    for T in range(T_training-1 , T_training):
        Vo = np.zeros((1,nmax)) #optimal policy

            
        for n in range(nmax): 
            P = np.vstack((p_c[n,start_training:T+start_training+1], p_2[n,start_training:T+start_training+1], p_1[n,start_training:T+start_training+1], c_h[n,start_training:T+start_training+1], c_p[n,start_training:T+start_training+1]))
            w1 = W1[n,start_training:T+start_training+1]
            w2 = W2[n,start_training:T+start_training+1]
            w1[w1<0] = 0
            w2[w2<0] = 0
            ################################################################ Optimal Policy ###########################################################
            # optimal policy assumes the same structure of the optimal policy in prop1 in kouvelis et al 2023
            s1_T = w1[T]*np.ones((1,30)) 
            s2_T = w2[T]+np.array([np.arange(0,30000,1000)])
            R_star = np.zeros((T+1,30)) 
            x1 = np.zeros((1,30)) #sell
            x2 = np.zeros((1,30)) #sell
            y1 = np.zeros((1,30)) #fulfill
            y2 = np.zeros((1,30)) #fulfill
            z1 = np.zeros((1,30)) #hold
            z2 = np.zeros((1,30)) #hold         
            #Case 1: OM dominates
            if P[0,T]+P[4,T]<P[1,T]:
                if P[2,T]>(1-alpha)*P[0,T]+P[4,T]:
                    x1 = np.maximum(s1_T,0)
                    x2 = np.maximum(s2_T,0) 
                else:
                     y1 = np.maximum(np.minimum(s1_T,q*np.ones((1,30))),0) 
                     x1 = np.maximum(s1_T - y1,0) 
                     x2 = np.maximum(s2_T,0) 
                   
            #Case 2: CM dominates
            else:
                if P[2,T]>(1-alpha)*P[0,T]+P[4,T]:
                    y2 = np.maximum(np.minimum(s2_T,q*np.ones((1,30))),0) 
                    x1 = np.maximum(s1_T,0) 
                    x2 = np.maximum(s2_T - y2,0) 
                elif P[2,T]<P[1,T]-alpha*P[0,T]:
           
                    y2 = np.maximum(np.minimum(s2_T,q*np.ones((1,30))),0) 
                    y1 = np.maximum(q - y2,0) 
                    x1 = np.maximum(s1_T - y1,0) 
                    x2 = np.maximum(s2_T - y2,0) 
                else:
                    y1 = np.maximum(np.minimum(s1_T,q*np.ones((1,30))),0) 
                    y2 = np.maximum(q - y1,0) 
                    x1 = np.maximum(s1_T - y1,0) 
                    x2 = np.maximum(s2_T - y2,0) 
                  
               
             #reward
            R_star[T,:] = P[0,T]*(y2+(1-alpha)*y1)+P[1,T]*x2+P[2,T]*x1-P[3,T]*(z1+z2)-P[4,T]*(q-y1-y2)-P[3,T]*(w1[T]+w2[T]) 
            for i in np.arange(T-1,-1,-1):
                s1 = w1[i]*np.ones((1,30)) 
                s2 = w2[i]+np.arange(0,30000,1000)
                dim_1 = int(max(2,np.ceil(min(w1[i],19000)/1000))) #ceil: round up
                dim_2 = int(max(2,np.ceil(min(w2[i],9999)/1000)))
                r_temp = np.zeros((30, dim_1,dim_2))  #python 3-D array indexing, the first argument selects matrix, the second selects the row
                for j in range(0,dim_1):############5
                    z_1 = 1000*j   # try every 1000 level of holding under-weight hogs [0,1000,min(w1[i],20000)]
                    for k in range(0,dim_2):######################
                        z_2 = 1000*k #try every 1000 level of holding regular-weight hogs
                        #initialization
                        x1 = np.zeros((1,30)) #sell
                        x2 = np.zeros((1,30)) #sell
                        y1 = np.zeros((1,30)) #fulfill
                        y2 = np.zeros((1,30)) #fulfill
                        z1 = z_1 * np.ones((1,30)) #hold
                        z2 = z_2 * np.ones((1,30)) #hold
                        s1 = s1 - z1 
                        s2 = s2 - z2 
                        #Case 1: OM dominates
                        if P[0,i]+P[4,i]<P[1,i]:
                            if P[2,i]>(1-alpha)*P[0,i]+P[4,i]:
                                x1 = np.maximum(s1,0) 
                                x2 = np.maximum(s2,0) 
                            else:
                                y1 = np.maximum(np.minimum(s1,q*np.ones((1,30))),0) 
                                x1 = np.maximum(s1 - y1,0) 
                                x2 = np.maximum(s2,0) 
                             
                         #Case 2: CM dominates
                        else:
                            if P[2,i]>(1-alpha)*P[0,i]+P[4,i]:
                                y2 = np.maximum(np.minimum(s2,q*np.ones((1,30))),0) 
                                x1 = np.maximum(s1,0) 
                                x2 = np.maximum(s2 - y2,0) 
                            elif P[2,i]<P[1,i]-alpha*P[0,i]:
    
                                y2 = np.maximum(np.minimum(s2,q*np.ones((1,30))),0) 
                                y1 = np.maximum(q - y2,0) 
                                x1 = np.maximum(s1 - y1,0) 
                                x2 = np.maximum(s2 - y2,0) 
                            else: 
                                y1 = np.maximum(np.minimum(s1,q*np.ones((1,30))),0) 
                                y2 = np.maximum(q - y1,0) 
                                x1 = np.maximum(s1 - y1,0) 
                                x2 = np.maximum(s2 - y2,0) 
                             
                           
                        #reward                    
                        r_temp[:,j,k] = P[0,i]*(y2+(1-alpha)*y1)+P[1,i]*x2+P[2,i]*x1-P[3,i]*(z1+z2)-P[4,i]*(q-y1-y2)+(1-beta)*R_star[i+1,int((z_1+z_2)/1000+1)]-P[3,i]*(w1[i]+w2[i]) #Bellman Eqn 
                    
                 
                
                R_star[i,:] = r_temp.max(axis = 1).max(axis = 1)
                   
            Vo[0,n] = min(R_star[0,:]) 
        V_o = np.zeros((1,1)) 
        V_o[0,0] = np.max(Vo) 
     



     
    return p_c[:,start_training:T+start_training+2], p_2[:,start_training:T+start_training+2], p_1[:,start_training:T+start_training+2], c_h[:,start_training:T+start_training+2], c_p[:,start_training:T+start_training+2], W1[:,start_training:T+start_training+2], W2[:,start_training:T+start_training+2], Vo 




















