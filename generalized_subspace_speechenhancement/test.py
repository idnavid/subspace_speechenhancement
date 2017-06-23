from tools import *
from subspace_method import *

if __name__=='__main__':
    test_wav = "../data/sa1-falr0.wav"
    win = 0.04
    inc = 0.02
    
    # Ground truth
    fs,s = read_wav(test_wav)
    sframes = enframe(s,int(win*fs),int(inc*fs))
    ## Voice activity threshold has been arbitrarily set:
    voice_labels = np.diagonal(np.dot(sframes,sframes.T)) > 4.0
    S = power_spectrum(sframes)

    # create noisy data
    ls_err = []
    els_err = []
    for snr in  [0., 10., 20., 30., 40., 50.]:
        power_s = compute_power(s)
        power_n = snr_to_power(snr,power_s)
        x = add_wgn(s,np.sqrt(power_n))
        xframes = enframe(x,win*fs,inc*fs)
        
        # Clean noisy data
        ## 1. Least Squares
        mu = 1.
        Rn = np.identity(xframes.shape[1]) * power_n
        Rn = estimate_noise_cov(xframes)
        xframes_ls = sub_space_enhancement(xframes,Rn, mu)
        
        ## 2. Efficient Least Squares
        xframes_els = p_sub_space_enhancement(xframes,xframes_ls,Rn,voice_labels)
        
        Xels = power_spectrum(xframes_els)
        Xls = power_spectrum(xframes_ls)
        els_err.append(np.mean(itakura_saito_dist(Xels,S)))
        ls_err.append(np.mean(itakura_saito_dist(Xls,S)))
    
    plot_these(ls_err,els_err)
    print("Done!")
