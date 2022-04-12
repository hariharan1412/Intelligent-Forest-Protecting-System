# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 16:58:31 2019
child of pitch_srh_nikos.m, 
does not return SRHmat,
use only for the needs of VAD
@author: nstefana
"""
import numpy.matlib
import numpy as np
from scipy.signal import lfilter
import librosa

def lpcresidual(x,L,shift,order):
    '''
    Function for LPC residual calculation  
    (translation of  lpcresidual.m in covarep/glottalsource)

        Parameters:
                x (1D array): Input Signal 
                L (int): Analysis window length
                shift (int) : Analysis window hop size 
                order (int): order of AR model for the LPC calculation 
        Returns:
                res (1D array:  Residual (Inverse filtered) of the input signal 
    '''
#    x=np.reshape(x,(x.size,1))
    shift=int(np.round(shift))
    order=int(np.round(order))
    start=0
    L=int(np.round(L))
    stop=start+L
    res=np.zeros((len(x),),dtype=float)
    #LPCcoef=np.zeros((order+1,np.round(len(x)/shift)),dtype=float)
    
    win=np.hanning(L)
#    win=np.reshape(win,(L,1))
    while stop<x.size:
        segment=x[start:stop]
        segment=np.multiply(segment,win)
#        print(segment)     
#        A,e,k=lpc(segment,order) #<-------- lpc based on 
        A=librosa.core.lpc(segment,order)
        #LPCcoeff= not implemented 
        inv=lfilter(A,1,segment)
        numerator=np.sum(np.power(segment,2))
        denominator=np.sum(np.power(inv,2))
        inv=inv*np.sqrt(numerator/denominator)
        res[start:stop]=res[start:stop]+inv
        start=start+shift
        stop=stop+shift
    return res



def SRH(specMat, nHarmonics, f0min, f0max):
    '''
    Function to compute Summation of Residual harmonics function
    on a spectrogram matrix, with each column corresponding to one
    spectrum.

        Parameters:
                specMat (2D array): Spectrogram of the corresponding signal 
                nHarmonics (int): Number of harmonics for the analysis
                f0min (int): Lowest analysis frequency (Hz)
                f0max (int): Highest analysis frequency (Hz)
        Returns:
                F0 (1D array): Fundamental frequency estimation for each frame 
                SRHVal (1D array): Voiced/Unvoiced probability
                SRHmat (2D array): SRH spectrogram feature  
    '''
    
    # Initial settings
    N = int(np.shape(specMat)[1])
    SRHmat = np.zeros((int(f0max),N),dtype=float)
    
    fSeq = np.arange(f0min,f0max,1)
    fSeq=fSeq.astype(int)
    fLen = len(fSeq)
    
    # Prepare harmonic indeces matrices.
    v = np.arange(1,nHarmonics+1,1,dtype=int)
    vT = np.reshape(v,(nHarmonics,1))
    q=np.arange(1,nHarmonics,1,dtype=int)
    qT=np.reshape(q,(nHarmonics-1,1))
    plusIdx1 = np.matlib.repmat(vT,1,fLen)
    plusIdx2 = np.matlib.repmat(fSeq,nHarmonics,1)
    plusIdx=np.multiply(plusIdx1,plusIdx2)
    plusIdx=plusIdx.astype(int)
    subtrIdx1=np.matlib.repmat(qT+0.5,1,fLen)
    subtrIdx2=np.matlib.repmat(fSeq,nHarmonics-1,1)
    subtrIdx12 = np.round(np.multiply(subtrIdx1,subtrIdx2))
    subtrIdx = subtrIdx12.astype(int)
    # avoid costly repmat operation by adjusting indices
    plusIdx = np.mod(plusIdx-1,np.shape(specMat)[0]) #+1;
    subtrIdx = np.mod(subtrIdx-1,np.shape(specMat)[0]) #+1
    # Do harmonic summation
    for n in range(0,N):
        specMatCur = specMat[:,n]
        SRHmat[fSeq,n] = np.conj(np.sum(specMatCur[plusIdx],axis=0) - np.sum(specMatCur[subtrIdx],axis=0))
        

    # Retrieve f0 and SRH value
    SRHVal= np.max(SRHmat, axis=0)
    F0=np.argmax(SRHmat, axis=0)
    return  F0, SRHVal, SRHmat
#    return SRHmat[f0min:,:]


def SRH_feature(specMat, nHarmonics, f0min, f0max):
    '''
    Function to compute Summation of Residual harmonics function
    on a spectrogram matrix, with each column corresponding to one
    spectrum.

        Parameters:
            specMat (array): Spectrogram of the corresponding signal 
            nHarmonics (int): Number of harmonics for the analysis
            f0min (int): Lowest analysis frequency (Hz)
            f0max (int): Highest analysis frequency (Hz)
   
       Returns:
            SRHmat (2D array): SRH spectrogram feature  
    '''
    # return [F0,SRHVal]

    
    # Initial settings
    N = int(np.shape(specMat)[1])
    SRHmat = np.zeros((int(f0max),N),dtype=float)
    
    fSeq = np.arange(f0min,f0max,1)
    fSeq=fSeq.astype(int)
    fLen = len(fSeq)
    
    # Prepare harmonic indeces matrices.
    v = np.arange(1,nHarmonics+1,1,dtype=int)
    vT = np.reshape(v,(nHarmonics,1))
    q=np.arange(1,nHarmonics,1,dtype=int)
    qT=np.reshape(q,(nHarmonics-1,1))
    plusIdx1 = np.matlib.repmat(vT,1,fLen)
    plusIdx2 = np.matlib.repmat(fSeq,nHarmonics,1)
    plusIdx=np.multiply(plusIdx1,plusIdx2)
    plusIdx=plusIdx.astype(int)
    subtrIdx1=np.matlib.repmat(qT+0.5,1,fLen)
    subtrIdx2=np.matlib.repmat(fSeq,nHarmonics-1,1)
    subtrIdx12 = np.round(np.multiply(subtrIdx1,subtrIdx2))
    subtrIdx = subtrIdx12.astype(int)
    # avoid costly repmat operation by adjusting indices
    plusIdx = np.mod(plusIdx-1,np.shape(specMat)[0]) #+1;
    subtrIdx = np.mod(subtrIdx-1,np.shape(specMat)[0]) #+1
    # Do harmonic summation
    for n in range(0,N):
        specMatCur = specMat[:,n]
        SRHmat[fSeq,n] = np.conj(np.sum(specMatCur[plusIdx],axis=0) - np.sum(specMatCur[subtrIdx],axis=0))
        
    return  SRHmat[f0min:,:]



def pitch_srh_preselect_vad(wave,fs,f0min,f0max,hopsize,frameDur,Nbins):
    '''
        Parameters:
                wave (1D array): Initial signal (waveform)
                fs (int): Samping frequency of signal
                f0min (int): Lowest SRH analysis frequency (Hz)
                f0max (int): Highest SRH analysis frequency (Hz)
                hopsize (float): Hop size for the STFT (ms)
                frameDur (float): Frame size for the STFT (ms)
                Nbins (int): Number of frequency bins in the resulting SRH spectrogram feature  
                 
        Returns:
                SRHVal (1D array): Voiced/Unvoiced probability
                timeIN (1D array): Center of each analysis segment (in sec) 
                sampleIdxsIN (1D array): Center of each analysis segment (in samples)
    '''
#%% Important settings    
    nHarmonics=4
    LPCorder=int(np.round(0.75*fs/1000));
    
#%% Compute LP residual
    res = lpcresidual(wave,int(np.round(25*fs/1000)),int(np.round(5*fs/1000)), LPCorder)
    
    ## Create frame matrix
    waveLen =len(wave)
    del wave

    frameDuration = int(np.round(frameDur*fs/1000))-2; # for better chainsaw detection use 250    
    frameDuration =int(np.round(frameDuration/2))*2 # Enforce evenness of the frame's length
    shift = int(np.round(hopsize*fs/1000))
    halfDur = int(np.round(frameDuration/2))
    
    sampleIdxsIN=np.arange(halfDur+1,waveLen-halfDur,shift)  
    
    N = len(sampleIdxsIN)
    Nkeep=N

    frameMat=np.zeros((frameDuration,N),dtype=float)

    for n in range(0,N):
        frameMat[:,n] = res[sampleIdxsIN[n]-halfDur:sampleIdxsIN[n]+halfDur]#-1]
    
    ## Create window matrix and apply to frames
    win = np.blackman(frameDuration)
    win = np.reshape(win,(len(win),1))
    frameMatWin = np.multiply(frameMat, win)    
    ## Do mean subtraction
    frameMean = frameMatWin.mean(axis=0)    
    #frameMatWinMean = bsxfun(@minus, frameMatWin, frameMean);
    frameMatWinMean = frameMatWin-frameMean    
    del frameMean,frameMatWin,frameMat
    ## Compute spectrogram matrix
    specMat = np.zeros((int(np.floor(fs/2)), int(np.shape(frameMatWinMean)[1])),dtype=float)
    idx = np.arange(0,np.floor(fs/2),1,dtype=int)

    for i in range(0,N): 
        frameIN=frameMatWinMean[:,i]
        fftFrame=np.fft.fft(frameIN,n=fs,axis=0)
        tmp = np.abs(fftFrame)
        specMat[:,i] = tmp[idx]
        
        
    del frameMatWinMean,tmp,idx;
    specDenom = np.sqrt(np.sum(np.power(specMat,2),axis=0))
    
    specMat = np.divide(specMat, specDenom)
    del specDenom
    
 #%% 

    SRHmatIN= SRH_feature(specMat, nHarmonics, f0min, f0max)                 
    del specMat
    SRHmatIN=SRHmatIN[:,:Nkeep] 
    timeIN=1.0*sampleIdxsIN/fs

    SRHval= np.max(SRHmatIN, axis=0)
        
    return SRHval, timeIN, sampleIdxsIN
