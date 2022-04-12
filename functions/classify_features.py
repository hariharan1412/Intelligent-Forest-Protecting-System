
import os
import numpy as np
import librosa
import pickle
import soundfile as sf 
import time
import glob
import shutil

#%%
def create_timestamp(eventTime): 
    minutes,seconds=divmod(eventTime,60)  
    hours, minutes=divmod(minutes,60) 
    s=format(int(seconds), '02d')
    m=format(int(minutes), '02d')
    h=format(int(hours), '02d')
        
    return h,m,s   

#%% 
def classify_features(pathIN,modelfileName,Nclasses,probThresh):
    
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.models import load_model

    directory = os.getcwd()
    print(directory)

    Fs=8000
    featurePathIN=(pathIN + '/' + 'Features')
    pathOUT=(pathIN + '/' + 'Extracted_segments')
    model = load_model(modelfileName)
    filenamesIN = glob.glob(os.path.join(featurePathIN,'*.obj'))  
    positiveSegments=np.zeros((0,2))     
    totalDetectedDuration=0
    for filename in filenamesIN:
        with open(filename,'rb') as fid:
            dataIN = pickle.load(fid,encoding='latin1')            
        featMatrix=dataIN['X']  
        wavName=dataIN['Nw']
        recName=dataIN['Nr']
        ftrTimeBorders=dataIN['ftrTB']        
        Nsamples=np.shape(featMatrix)[0]   
        if Nsamples>0:
            try:
                probs = model.predict(featMatrix,batch_size=16)
            except:
                featMatrix=tf.expand_dims(featMatrix,axis = -1)
                probs = model.predict(featMatrix,batch_size=16)
            probs=np.reshape(probs,(Nsamples,Nclasses))
            sawPR=probs[:,0]  
            positiveArgs=np.argwhere(sawPR>probThresh)
            positiveArgs=positiveArgs.flatten()
            positiveSegments=ftrTimeBorders[positiveArgs,:]
            positiveSawPRs=sawPR[positiveArgs]
       
            finalList=[]
            tempList=[0,]
            q=0
            m=1             
            while m < np.shape(positiveSegments)[0]:
                q+=1
                if positiveSegments[m,0]<=positiveSegments[m-1,1]:
                    positiveSegments[m-1,1]=positiveSegments[m,1]
                    positiveSegments=np.delete(positiveSegments,m,axis=0)
                    tempList.append(q)                         
                else:
                    finalList.append(tempList)
                    tempList=[q,]
                    m+=1 
                    
            finalList.append(tempList)
                
            Nps=len(positiveSegments)
            for u in range(Nps):                 
                sIN, sr = librosa.load(wavName, sr=Fs, offset=positiveSegments[u,0],duration=positiveSegments[u,1]-positiveSegments[u,0])   
                sIN=sIN-np.mean(sIN)
                hours,minutes,seconds=create_timestamp(positiveSegments[u,0])   
                prob='%.2f' %np.mean(positiveSawPRs[finalList[u]])
                instanceName=('instance' + '_' + hours + 'h' + minutes + 'm' + seconds + 's' + '_prob' + prob + '.wav')
                # instanceName=('prob' + prob + '_instance' + '_' + hours + 'h' + minutes + 'm' + seconds + 's.wav')                
                sf.write(os.path.join(pathOUT,(recName + '_' + instanceName)),sIN,Fs)  
                print('Utterance classified as chainsaw with mean probability ' + str(np.mean(positiveSawPRs[finalList[u]])))
                totalDetectedDuration+=positiveSegments[u,1]-positiveSegments[u,0]
            source_folder = directory+"\\audio_file\\Extracted_segments\\"
            destination_folder = directory+"\\audio_file\\Detected_Chainsaw_Audios\\"
            for file_name in os.listdir(source_folder):
                # construct full file path
                source = source_folder + file_name
                destination = destination_folder + file_name
                # move only files
                if os.path.isfile(source):
                    shutil.move(source, destination)
                    deletefolder = directory+'\\audio_file\\Features'
                    for deletefilename in os.listdir(deletefolder):
                        file_path = os.path.join(deletefolder, deletefilename)
                        try:
                            if os.path.isfile(file_path) or os.path.islink(file_path):
                                os.unlink(file_path)
                            elif os.path.isdir(file_path):
                                shutil.rmtree(file_path)
                        except Exception as e:
                            print('Failed to delete %s. Reason: %s' % (file_path, e))
                    # os.remove("D:\\Audio Detection\\new - Copy\\audio_file\\Features\\*.obj")
                    print('Moved:', file_name)
        print('Total detected duration of chainsaw in seconds is ' + '%.2f' %totalDetectedDuration) 
        
        if totalDetectedDuration == 0:
            print('CHAINSAW NOT DETECTED ')
            return 0
        else:
            print('CHAINSAW DETECTED ')
            return 1
        
    return totalDetectedDuration 
    # time.sleep(4)         
    
