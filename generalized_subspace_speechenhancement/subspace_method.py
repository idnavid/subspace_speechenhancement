from tools import *



def sub_space_enhancement(xframes, Rn, mu=0.01):
    # First estimate covariance of clean data, Rx, and
    # observatoins, Ry, based on the assumption that
    # noise and speech are uncorrelated (i.e., Ry = Rx + Rn)
    Ry = covariance(xframes)
    Sigma = np.dot(np.linalg.inv(Rn),Ry) - np.eye(Ry.shape[0])
    L,V  = np.linalg.eig(Sigma)
    K = len(L)
    M = sum(L>0)
    Q = L[:M]
    Q_denom = Q+mu
    Q = np.divide(Q,Q_denom)
    Q11 = np.diag(abs(Q))
    Q12 = np.zeros((Q11.shape[0],K-M))
    Q21 = np.zeros((K-M,Q11.shape[0]))
    Q22 = np.zeros((K-M,K-M))
    top_rows = np.hstack((Q11,Q12))
    bot_rows = np.hstack((Q21,Q22))
    Q = np.vstack((top_rows,bot_rows))
    H_ls = np.dot(np.linalg.inv(V.T),np.dot(Q,V.T))
    return np.dot(xframes,H_ls)



def p_sub_space_enhancement(xframes,xframes_ls,labels):
    res = xframes - xframes_ls # residual
    Sigma_s = covariance(res)
    L,V  = np.linalg.eig(Sigma_s)
    K = sum(L>0.01)
    K = max(10,K)
    Vr = V[:,:K]
    Lr = L[:K]
    zframes = np.dot(xframes,Vr)
    #labels = voice_detection(xframes)
    Rnz = estimate_noise_cov(zframes,labels)
    zframes_ls = sub_space_enhancement(zframes,Rnz)
    return np.dot(zframes_ls,Vr.T)




def varcov_principle_components(xframes):
    '''
        This function uses the principle component estimation technique
        proposed by Ching, Seghouane, Salleh for fMRI dimension reduction. 
        See SLP letter: Estimating Effective Connectivity from fMRI Data
        Using Factor-based Subspace Autoregressive Models, 2015. 
        '''
    y = xframes
    #plot_this(np.abs(np.fft.fft(y[600,:])))
    (T,N) = y.shape # T: num of samples, N: size of each observatoin
    y = zero_mean(y)
    Sigma_y = auto_covariance(y)
    L_full, Q_full = np.linalg.eig(Sigma_y) # L: EigVals, Q: EigVecs
    # Note: must create function to automatically estimate r.
    r = 70
    L_r = L_full[:r]
    plot_this(L_full)
    Q_r = Q_full[:,:r]
    f = np.dot(Q_r.T,y.T)
    y_est = np.dot(Q_r,f)
    #plot_this(np.abs(np.fft.fft(y_est[600,:])))
    return np.real(y_est)


if __name__=='__main__':
    test_wav = "../data/sa1-falr0.wav"
    fs,x = read_wav(test_wav)
    # test_wav = "../data/sa1-falr0_pls.wav"
    # fs,x2 = read_wav(test_wav)
    # x = np.hstack(x1,x2)
    y = add_wgn(x,0.1)
    win = 0.04
    inc = 0.02
    xframes = enframe(x,int(win*fs),int(inc*fs))
    nframes = xframes.shape[0]
    size_frame = xframes.shape[1]
    R = covariance(xframes)
    # We need to be sure we've chosen the right dimention
    assert (size_frame==int(win*fs))
    Rn = 0.01*np.eye(size_frame)
    for mu in [5]:
        xframes_ls = varcov_principle_components(xframes)
        Rn = covariance(xframes) - covariance(xframes_ls.T)
        xframes_ls = sub_space_enhancement(xframes,Rn,mu)
        X = power_spectrum(xframes)
        Xls = power_spectrum(xframes_ls)
        plot_this(np.log(Xls.T+1e-5),title='LS')
        plot_this(np.log(X.T+1e-5),title='orig')

    print("Done!")
