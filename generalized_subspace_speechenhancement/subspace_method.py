from tools import *



def sub_space_enhancement(xframes, Rn, mu=0.01):
    # First estimate covariance of clean data, Rx, and
    # observatoins, Ry, based on the assumption that
    # noise and speech are uncorrelated (i.e., Ry = Rx + Rn)
    xframes = zero_mean(xframes)
    Ry = np.cov(xframes.T)
    Rx = Ry - Rn

    Sigma = np.dot(np.linalg.inv(Rn),Rx)
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



def p_sub_space_enhancement(xframes,xframes_ls,Rn,labels):
    res = xframes - xframes_ls # residual
    Sigma_s = np.cov(res.T)
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







if __name__=='__main__':
    test_wav = "../data/sa1-falr0.wav"
    fs,s = read_wav(test_wav)
    x = add_wgn(s,0.1)
    y = add_wgn(s,0.1)
    win = 0.04
    inc = 0.02
    xframes = enframe(x,win*fs,inc*fs)
    yframes = enframe(y,win*fs,inc*fs)
    for mu in [5]:
        xframes_ls = sub_space_enhancement(xframes,mu)
        #plot_these(deframe(xframes_ls,win*fs,inc*fs),s)
        #plot_these(s,deframe(xframes_ls,win*fs,inc*fs))
        X = power_spectrum(xframes)
        Y = power_spectrum(yframes)
        Xls = power_spectrum(xframes_ls)
        plot_this(np.log(Xls.T+1e-5),title='LS')
        plot_this(np.log(X.T+1e-5),title='orig')
        print np.linalg.norm(X-Xls)
    print "Done!"
