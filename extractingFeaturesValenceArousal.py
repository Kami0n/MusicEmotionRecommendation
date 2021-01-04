import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import pyAudioAnalysis

from os import listdir
from os.path import isfile, join
from commonFunctions import musicFeatureExtraction
from commonFunctions import displayVAgraph

dirname = os.path.dirname(__file__)

# function to parse CSV file of valence and arousal
def parseDEAM(pathEmotions):
    emotions = pd.read_csv(pathEmotions, index_col=0, sep=',')
    return emotions

subfolder = 'DatasetComb/'
emotions = parseDEAM(subfolder+'annotations/annotations_all.csv')
allFeatures = []

#filesTemp = [1000]
#for fileName in filesTemp:
for fileName, row in emotions.iterrows():
    print(fileName)
    fullFilePath = subfolder+'audio/'+str(fileName)+'.mp3'
    #print(fullFilePath)
    data = musicFeatureExtraction(fullFilePath)
    allFeatures.append([fileName, data, emotions.loc[fileName,'valence_mean'], emotions.loc[fileName,'arousal_mean']])

featuresdf = pd.DataFrame(allFeatures, columns=['id','features','valence', 'arousal'])
featuresdf = featuresdf.set_index(['id'])
print(featuresdf)

featuresdf.to_pickle(subfolder+'pickle/features_valence_arousal.pkl')

