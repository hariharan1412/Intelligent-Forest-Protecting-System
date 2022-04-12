# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 16:25:35 2019
@author: FORTH-ICS
"""
import os
import numpy as np
import librosa
import pickle
import time
#from pitch_srh_preselect import pitch_srh_preselect
from functions.pitch_srh_preselect_vad import pitch_srh_preselect_vad as pitch_srh_preselect

#%%
def compression_matrix(compRatio,Nbins):
    '''
    Helper function
    '''
    Ncols=int(np.floor(Nbins/compRatio))
    Nrem=int(np.mod(Nbins,compRatio))
    M=np.zeros((Nbins,Ncols),dtype=np.float32) 
    for r in range(Ncols-1):
        M[r*compRatio:(r+1)*compRatio,r]=np.ones((compRatio,),dtype=np.float32)         
    M[(r+1)*compRatio:Nbins,Ncols-1]=np.ones((compRatio+Nrem,),dtype=np.float32)  
    return M 

#%%
def my_pcen_smooth(Sin,e,bias,alpha,power,ampFactor,step):
    '''
    Helper function
    '''
    s=0.15
    Sin=ampFactor*np.abs(Sin)
    Nbins=np.shape(Sin)[0]
    Nfr=np.shape(Sin)[1]
    Ein=Sin**2
    CM=compression_matrix(step,Nbins)
    CSin=np.sqrt(np.matmul(CM.T,Ein))
    Nbands=np.shape(CM)[1]
    Mean=np.mean(CSin,axis=1)

    Mt=np.zeros((Nbands,Nfr+1))
    Mt[:,0]=Mean
    Sout=np.zeros(np.shape(CSin))
    
    for t in range(1,Nfr+1):
        Mt[:,t]=(1-s)*Mt[:,t-1]+s*CSin[:,t-1] 
    Mt=Mt[:,1:]  
    
    CSuse=np.zeros((Nbands,Nfr))
    for b in range(Nbands):
        CSuse[b,:]=np.max(Sin[np.arange(step*b,step*(b+1)),:],axis=0)       
        
    for b in range(Nbands):    
        Sout[b,:]=(CSuse[b,:]/((e+Mt[b,:])**alpha) + bias)**power-bias**power
    return Sout


# #%% necessary functions -- replaced by a lambda function (lines 169-170)
# def find_active_segments(srh,threshold):
#     Nu=srh.size
#     keptIdxs=np.argwhere(srh>threshold)
    
#     return keptIdxs, Nu

#%% necessary functions
def find_utterance_Idxs(keptIdxs,Ntf,NcontAct,Nelastic,Nfr): # ,maxUtterLength): #chanded in 16/7/2020
    '''
    Helper function for the feature extraction procedure

    '''
    Ns=keptIdxs[-1]
    Nstart=keptIdxs[0]
    utterIdxs=np.array(([Nstart,0]),dtype=int,ndmin=2)    
    
    for i in range(int(Nstart)+1,int(Ns)):
        if np.any(keptIdxs==i):
            if np.any(keptIdxs==i-1):
                utterIdxs[-1,1]=i
            else:
                utterIdxs=np.vstack((utterIdxs, np.array((i, i),dtype=int)))
                
    Nutt=np.shape(utterIdxs)[0]    
    utIdxs=np.zeros((0,2),dtype=int)
    for u in range(Nutt):
            Nloc=utterIdxs[u,1]-utterIdxs[u,0]
            if Nloc>=NcontAct: 
                utIdxs=np.vstack((utIdxs, [utterIdxs[u,0],utterIdxs[u,1]]))
    
    N3=np.shape(utIdxs)[0] 
    uIdxs=np.zeros((0,2),dtype=int)
    for u in range(N3):
        Nloc=utIdxs[u,1]-utIdxs[u,0]
        if Nloc<Nfr:
            Nadd=np.ceil((Nfr-Nloc)/2)
            if (utIdxs[u,0]-Nadd)>=0 and (utIdxs[u,1]+Nadd)<=(Ntf-1):
                idxMin=utIdxs[u,0]-Nadd    
                idxMax=utIdxs[u,1]+Nadd
                uIdxs=np.vstack((uIdxs, [int(idxMin),int(idxMax)]))
            
        else:
            uIdxs=np.vstack((uIdxs, [utIdxs[u,0],utIdxs[u,1]]))                
    
    N2=np.shape(uIdxs)[0]   
    if N2>1:         
        m=1            
        while m < np.shape(uIdxs)[0]:
            if uIdxs[m,0]-uIdxs[m-1,1]<=Nelastic:
                uIdxs[m-1,1]=uIdxs[m,1]
                uIdxs=np.delete(uIdxs,m,axis=0)
            else:
                m+=1     

    Nutters=np.shape(uIdxs)[0]
    return  uIdxs, Nutters

#%% define idx Borders (>Feb 2021)
def defineIdxBorders(Ns,frameStepFTR,Nfr,hopDurFTR_s,durationFTR,timeUtterStart):
    '''
    Helper function for the feature extraction procedure

    '''
    Nframes=0
    startIdxs=np.arange(0,Ns,frameStepFTR,dtype=int)
    timeGrid=timeUtterStart+startIdxs*hopDurFTR_s
    endIdxs=startIdxs+Nfr
    validBorders=np.argwhere(endIdxs<=Ns)
    Nframes=len(validBorders)
    idxBorders=np.zeros((Nframes,2),dtype=int)
    ftrTimeInterval=np.zeros((Nframes,2),dtype=float)
    for n in range(Nframes):
        idxBorders[n,:]=[startIdxs[n],endIdxs[n]] 
        ftrTimeInterval[n,:]=[timeGrid[n],timeGrid[n]+durationFTR]   
    return Nframes,idxBorders,ftrTimeInterval  

#%%
def extract_pcen_feature(wavName,outputFilePath,timeBorders,segmIdx,VADthresh):    
    '''
    Function to extract PCEN features for each wav file.
    '''
    find_active_segments = \
            lambda srh,threshold: (np.argwhere(srh>threshold), srh.size)

    NcontAct=3 #6
    Nelastic=1 # 21
    Nfr=46 #46 #depending on Nfr23 or Nfr46
    frameStepFTR=16 #can be varied according to classw
    frameDurFTR_ms=90.0
    hopDurFTR_ms=30.0
    hopDurFTR_s=hopDurFTR_ms/1000.0
    timeStep=frameStepFTR*hopDurFTR_ms/1000.0
    durationFTR=(frameDurFTR_ms+(Nfr-1)*hopDurFTR_ms)/1000.0
    Fs=8000
    F0min = 20
    F0max = 760
    NbinsSRH=120
    hopDurVAD_ms=60 #30
    frameDurVAD_ms=180  
    frameDurVAD_s=frameDurVAD_ms/1000.0        
    fftLength=1024
    frameLength=int(frameDurFTR_ms/1000*Fs)
    
    hopLength=int(hopDurFTR_ms/1000*Fs) # default=0.06
    binsIN=np.arange(0,220,dtype=int) # (5,130)
    NbinsIN=len(binsIN)
    step=2
    NbinsOUT=int(np.floor(NbinsIN/step))
    bins4USE=np.arange(4,107)  #final models use this idxs    
    Nbins4USE=len(bins4USE)
    Nfr_vad=np.ceil(((Nfr-1)*hopDurFTR_ms+frameDurFTR_ms-frameDurVAD_ms)/hopDurVAD_ms)+1

#%%    
    eps=10**(-6)
    bias=10
    alpha=0.8
    power=0.25
    ampFactor=(1/8)*2**(31) 
       
#%%    
    Ncount=0
    ftrTimeBorders=np.zeros((0,2),dtype=float)    
    featMatrix=np.zeros((0,Nfr,Nbins4USE), dtype=np.float16) #for 6000 samples.
    
    fullwavName=wavName.split(os.sep)[-1]
    recName=fullwavName.split('.wav')[0]  
    segmDur=timeBorders[segmIdx+1]-timeBorders[segmIdx]
    segmTimeStart=timeBorders[segmIdx]    
    name2save=(outputFilePath + '/' + 'Features/' + recName + '_start_' + format(int(segmTimeStart),'05d') + '.obj') 
    
#%% Check if feature objects  are already extracted with the same VAD threshold, if yes, do not produce them again 
    if os.path.exists(name2save) == False:
        run_segment=True
    else:
        with open(name2save,'rb') as fid:
            dataIN = pickle.load(fid,encoding='latin1')  
        if dataIN['VADthresh']==threshold:
            run_segment=False
            print('Features for file ' + name2save + ' already extracted, will not re-calculate them')
        else:
            run_segment=True   
#%%
    if run_segment:
        print('running now segment ' + str(segmIdx) +  ' of file ' + wavName)
        sIN, sr = librosa.load(wavName, sr=Fs, offset=segmTimeStart,duration=segmDur); 
        srh,timePoints,sampleIdxs =  pitch_srh_preselect(sIN,Fs,F0min,F0max,hopDurVAD_ms,frameDurVAD_ms,NbinsSRH)
        del sIN
        
        keptIdxs,Nu=find_active_segments(srh,VADthresh)
        if keptIdxs.size>0:
            utIdxs,Nutters=find_utterance_Idxs(keptIdxs,Nu,NcontAct,Nelastic,Nfr_vad) #,maxUtterLength)  
            for m in range(Nutters):
                utterTimeBorder=np.array((timePoints[utIdxs[m,0]]-0.5*frameDurVAD_s,timePoints[utIdxs[m,1]]+0.5*frameDurVAD_s))+segmTimeStart   #+1           
                utterDur=utterTimeBorder[1]-utterTimeBorder[0]
                sIN, sr = librosa.load(wavName, sr=Fs, offset=utterTimeBorder[0],duration=utterDur); Ns=len(sIN)  
                sIN=sIN-np.mean(sIN)
                sUSE=sIN*np.sqrt(Ns)/np.linalg.norm(sIN) #<------
                Sin=librosa.core.stft(sUSE,n_fft=fftLength,hop_length=hopLength,win_length=frameLength,window='hann',center=False)
                Sin=np.abs(Sin[binsIN,:])  
                Spcen=my_pcen_smooth(Sin,eps,bias,alpha,power,ampFactor,step)
                featMatrixUtterance=np.zeros((0,Nfr,Nbins4USE), dtype=np.float16) #for 6000 samples                
                NfrIN=np.shape(Sin)[1]
                utterStart=timePoints[utIdxs[m,0]]-0.5*frameDurVAD_s+segmTimeStart
                
                Nframes,idxBorders,ftrTimeInterval=defineIdxBorders(NfrIN,frameStepFTR,Nfr,hopDurFTR_s,durationFTR,utterStart) 
                for n in range(Nframes):    
                    sPortion=np.transpose(Spcen[bins4USE,idxBorders[n,0]:idxBorders[n,1]])
                    featMatrixUtterance=np.concatenate((featMatrixUtterance,np.float16(sPortion[None,...])),axis=0)                       
                featMatrix=np.concatenate((featMatrix,featMatrixUtterance),axis=0)    
                ftrTimeBorders=np.concatenate((ftrTimeBorders,ftrTimeInterval),axis=0) 
                    
        Nsamples=np.shape(featMatrix)[0] 
        with open(name2save, 'wb') as fid:        
            pickle.dump({'X' : featMatrix,'ftrTB':ftrTimeBorders,'Nw': wavName,
                         'Ns': Nsamples, 'Nr':recName,'VADthresh':VADthresh}, fid) #, 'probs':sawProbs

     
