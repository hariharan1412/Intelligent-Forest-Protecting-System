import argparse
import os , shutil
import numpy as np
import librosa
import glob
import multiprocessing
import sys
import pyaudio
import wave

import smtplib
import imghdr
from email.message import EmailMessage

from functions.extract_pcen_feature import extract_pcen_feature as extract_features
from functions.classify_features import classify_features

class chainsaw_detect:
    def __init__(self):
        
        self.mail_from = 'teamstartup.ht143@gmail.com'
        self.mail_password = 'godsense143'

        self.mgs = EmailMessage()
        self.mgs['subject'] = ' Cutting Down Tree has been Detected! '
        self.mgs['From'] = self.mail_from
        self.mgs['To'] = 'teamstartuptesting@gmail.com'
        self.mgs.set_content('Illegal actions  has been detected at Microphone 1 location ')

    def main(self):

        print("recording Audio")

        audio = pyaudio.PyAudio()

        recordSeconds = 10
        sampleRate = 8000
        bufferFrames = 1024

        stream = audio.open(format=pyaudio.paInt16, channels=1, rate=sampleRate, input=True, frames_per_buffer=bufferFrames)

        frames = []

        for i in range(0, int(sampleRate / bufferFrames*recordSeconds)):
            data = stream.read(bufferFrames)
            frames.append(data)
            if i%8==0:
                print("Listening...")
        stream.stop_stream()
        stream.close()
        audio.terminate()
    #importing the os module

    #to get the current working directory
        directory = os.getcwd()

        sound_file = wave.open(directory+"\\audio_file\\Recorded Audio.wav","wb")
        sound_file.setnchannels(1)
        sound_file.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        sound_file.setframerate(8000)
        sound_file.writeframes(b''.join(frames))
        sound_file.close()
        
        parser = argparse.ArgumentParser(description='batch_processor', formatter_class=argparse.ArgumentDefaultsHelpFormatter)  
        parser.add_argument('-u', '--nopREQ', type=int, default=6, help='number of processing units employed')    
        parser.add_argument('-t', '--VADthresh', type=float, default=0.078, help='Treshold for VAD detection, default=0.078') 
        parser.add_argument('-p', '--probThresh', type=float, default=0.75, help='Treshold for RNN classifier, default=0.75')     
        args = parser.parse_args()

        #%% Parameters
        maxDur=400 #in seconds
        nop=multiprocessing.cpu_count()
        nopREC=np.max([1,nop-1])
        if args.nopREQ<nopREC:
            nopUSE=args.nopREQ
        else:
            nopUSE=nopREC
        if args.VADthresh>=0.0779:
            VADthresh=args.VADthresh
        else:
            VADthresh=0.078   
            
        if args.probThresh>1:
            pass

        probThresh=args.probThresh
        
        inputWavPath = directory+"\\audio_file"
        outputDataPath = directory+"\\audio_file"
        wavFileNames=[]
        
        modelfileName='teamStartup_chainsaw_model.hdf5'
        Nclasses=2 
        # pcen_rnn4_cl2_RMED_allARUs_run0
        
    # %%

        if os.path.exists(inputWavPath + '/' + 'Features') == True:
            shutil.rmtree((inputWavPath + '/' + 'Features'))     
            
        if os.path.exists(inputWavPath + '/' + 'Extracted_segments') == True:
           shutil.rmtree((inputWavPath + '/' + 'Extracted_segments'))    

        if os.path.exists(inputWavPath + '/' + 'Features') == False:
            os.mkdir((inputWavPath + '/' + 'Features'))      
        
        if os.path.exists(inputWavPath + '/' + 'Extracted_segments') == False:
            os.mkdir((inputWavPath + '/' + 'Extracted_segments'))    

    #%% do the a=ob  
        folder_with_recordings=(inputWavPath + '/*.wav')

        for wavName in glob.glob(folder_with_recordings):
            pool=multiprocessing.Pool(nopUSE)
            fileDuration=librosa.get_duration(filename=wavName)
            sr=librosa.get_samplerate(wavName)
            if sr==8000:
                wavFileNames.append(wavName)                     
                if fileDuration<maxDur:
                    timeBorders=np.array((0,fileDuration))   
                else:
                    timeBorders=np.arange(0,fileDuration,maxDur)
                    timeBorders=np.delete(timeBorders,-1,axis=None)
                    timeBorders=np.append(timeBorders,fileDuration)
                    
                Nsegm=timeBorders.size   
                
                for segmIdx in range(Nsegm-1): #range(0,Nsegm-1):       
                    pool.apply_async(extract_features, args=(wavName,outputDataPath,timeBorders,segmIdx,VADthresh))

                pool.close()
                pool.join() 
            else:
                print(wavName.split(os.sep)[-1] + ' has a sampling rate different than 8000  Hz, will not process this audio file ')
    #%% Run classifier and extract positive .wav segments       
        if classify_features(outputDataPath,f"Models/{modelfileName}",Nclasses,probThresh):                    

            with smtplib.SMTP_SSL('smtp.gmail.com',465) as smtp:
                smtp.login(self.mail_from,self.mail_password)
                smtp.send_message(self.mgs)
                    

    def detect(self):
        while True:
            try:
                self.main()
                self.sleep(1)
            except:
                pass