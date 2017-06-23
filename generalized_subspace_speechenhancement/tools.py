import numpy as np
from scipy.io import wavfile
from scipy.cluster.vq import vq, kmeans
import pylab


#### Display tools
def plot_this(s,title=''):
    s = s.squeeze()
    if s.ndim ==1:
        pylab.plot(s)
    else:
        pylab.imshow(s,aspect='auto')
        pylab.title(title)
    pylab.show()

def plot_these(s1,s2):
    try:
        # If values are numpy arrays
        pylab.plot(s1/max(abs(s1)),color='red')
        pylab.plot(s2/max(abs(s2)),color='blue')
    except:
        # Values are lists
        pylab.plot(s1,color='red')
        pylab.plot(s2,color='blue')
    pylab.legend()
    pylab.show()

#### Signal level tools
def add_wgn(s,var=1e-4):
    np.random.seed(0)
    # Add white Gaussian noise to signal
    # If no variance is given, simply add jitter
    noise = np.random.normal(0,var,len(s))
    return s + noise

def read_wav(filename):
    """
        read wav file. 
        Normalizes signal to values between -1 and 1. 
        Also add some jitter to remove all-zero segments."""
    fs, s = wavfile.read(filename) # scipy reads int
    s = np.array(s)/float(max(abs(s)))
    s = add_wgn(s) # Add jitter for numerical stability
    return fs,s

def enframe(x, winlen, hoplen):
    """
        receives a 1D numpy array and divides it into frames.
        outputs a numpy matrix with the frames in rows.
        """
    x = np.squeeze(x)
    if x.ndim != 1:
        raise TypeError("enframe input must be a 1-dimensional array.")
    n_frames = 1 + np.int(np.floor((len(x) - winlen) / float(hoplen)))
    xf = np.zeros((n_frames, winlen))
    for ii in range(n_frames):
        #xf[ii] = np.multiply(x[ii * hoplen : ii * hoplen + winlen],np.hamming(winlen))
        xf[ii] = x[ii * hoplen : ii * hoplen + winlen]
    return xf

def deframe(x_frames, winlen, hoplen):
    '''
        Implementation of Overlap-add to reconstruct signal from frames.
        '''
    n_frames = x_frames.shape[0]
    n_samples = n_frames*hoplen + winlen
    x_samples = np.zeros((n_samples,1))
    for ii in range(n_frames):
        tmp = np.real(x_frames[ii,:].reshape((winlen,1)))
        tmp1 = np.multiply(tmp.T,np.hamming(winlen))
        x_samples[ii*hoplen : ii*hoplen + winlen]+= tmp
    return x_samples



def voice_detection(xframes):
    """
        Apply k-means to frame energies to detect voiced frames. 
        i.e., Frames with higher energy are considered voiced. 
        labels has the same no.of.rows as xframes, with 0s for unvoiced 
        and 1s for voiced frames. 
        """
    xframes_0 = zero_mean(xframes)
    frame_nrgs = np.mean(np.multiply(xframes,xframes),axis=1)
    centroids,_ = kmeans(frame_nrgs,2)
    labels,_ = vq(frame_nrgs,centroids)
    if np.mean(frame_nrgs[labels]) < np.mean(frame_nrgs[1 - labels]):
        labels = 1 - labels
    return labels > 0


def compute_power(x):
    """
        calculate the average power of the signal by average energy of frames.
        """
    if x.squeeze().ndim==1:
        # xframes is not blocked into frames
        return np.dot(x,x)/float(len(x))
    
    # Otherwise, x contains signal frames
    xframes = zero_mean(x)
    labels = voice_detection(xframes)
    xframes = xframes[labels,:]
    nrg = 0
    L = xframes.shape[0]
    for i in range(L):
        nrg+=np.dot(xframes[i,:],xframes[i,:])
    return nrg/float(L)


def snr_to_power(snr,power_s):
    """
        Compute the variance from snr by comparing
        the power of the signal, power_s.
        """
    return power_s*((10**(snr/10.))**(-1))

def estimate_noise_cov(xframes,labels=[-1]):
    """
        This function takes signal frames as input
        and estimates noise covariance by applying 
        voice-unvoice detection.
        """
    if labels[0]<0:
        labels = voice_detection(xframes)
    xframes_voiced = xframes[labels,:].T
    return np.cov(xframes_voiced)


def power_spectrum(x):
    """
        x: input signal, each row is one frame
        """
    X = np.fft.fft(x,axis=1)
    X = np.abs(X[:,:X.shape[1]/2])**2
    return np.sqrt(X)


#### Error Calculation tools
def L2norm(P1,P2):
    D = P1-P2
    return np.mean(np.sum(np.sqrt(np.multiply(D,D)),axis=1))

def relative_L2norm(P1,P2):
    """
        Doesn't seem like a good measure.
        Except that it tells us that whether
        an algorithm is especially detrimental
        in high SNR (i.e., low noise).
    """
    return L2norm(P1,P2)/L2norm(P2,0)

def itakura_saito_dist(x1,x2):
    power_ratio = np.divide(x1+1e-5,x2+1e-5)
    log_power_ratio = np.log(power_ratio)
    return np.mean(power_ratio - log_power_ratio - np.ones((log_power_ratio.shape)),axis=1)

def itakura_saito_dist_vbox(pf1,pf2):
    nframes,nfreq = pf1.shape
    r = np.divide(pf1,pf2+1e-5)
    q = r - np.log(r)
    d = np.mean(q,axis=1) - 1
    return d

def compute_snr(xs,xn):
    return 10*np.log10(frame_power(xs)/(frame_power(xn)+1e-4))
