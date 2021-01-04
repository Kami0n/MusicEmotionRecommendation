import numpy as np
import pandas as pd
import librosa
import warnings
warnings.filterwarnings('ignore')

subfolderName = 'napovej/'
showResults = True

def displayFeature(data, name):
    fig, ax = plt.subplots()
    img = librosa.display.specshow(data, y_axis=name, x_axis='time', ax=ax)
    fig.colorbar(img, ax=ax)
    ax.set(title=name)
    plt.show()

# 193 audio features from librosa
def musicFeatureExtraction(filePath):
    
    y, sr = librosa.load(filePath, res_type='kaiser_fast')
    
    chroma = librosa.feature.chroma_stft(y=y, sr=sr) # returns: Normalized energy for each chroma bin at each frame.
    chroma_processed = np.mean(chroma.T,axis=0)
    
    melspectrogram = librosa.feature.melspectrogram(y=y, sr=sr) # returns: Mel spectrogram
    melspectrogram_processed = np.mean(melspectrogram.T,axis=0)
    
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40) # returns: MFCC sequence
    mfccs_processed = np.mean(mfcc.T,axis=0)
    
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)  # returns: each row of spectral contrast values corresponds to a given octave-based frequency
    spectral_contrast_processed = np.mean(spectral_contrast.T,axis=0)
    
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)  # returns: p’th-order spectral bandwidth.
    spectral_bandwidth_processed = np.mean(spectral_bandwidth.T,axis=0)
    
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)  # returns: roll-off frequency.
    spectral_rolloff_processed = np.mean(spectral_rolloff.T,axis=0)
    
    #poly_features = librosa.feature.poly_features(y=y, sr=sr) # returns: coefficients of fitting an nth-order polynomial to the columns of a spectrogram.
    #poly_features_processed = np.mean(poly_features.T,axis=0)
    
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr) # returns: Tonal centroid features for each frame.
    tonnetz_processed = np.mean(tonnetz.T,axis=0)
    
    zcr = librosa.feature.zero_crossing_rate(y=y)  # returns: zero-crossing rate of an audio time series.
    zcr_processed = [np.mean(zcr)]
    
    bpm = librosa.beat.tempo(y=y, sr=sr)  # returns: bpm
    
    fetureNp = mfccs_processed
    fetureNp = np.concatenate((fetureNp,chroma_processed))
    fetureNp = np.concatenate((fetureNp,melspectrogram_processed))
    fetureNp = np.concatenate((fetureNp,spectral_contrast_processed))
    fetureNp = np.concatenate((fetureNp,tonnetz_processed))
    
    # dodane potem
    #fetureNp = np.concatenate((fetureNp,poly_features_processed))
    #fetureNp = np.concatenate((fetureNp,[0,0]))
    fetureNp = np.concatenate((fetureNp,spectral_rolloff_processed))
    fetureNp = np.concatenate((fetureNp,spectral_bandwidth_processed))
    fetureNp = np.concatenate((fetureNp,zcr_processed))
    fetureNp = np.concatenate((fetureNp,bpm))
    
    #print('size:', fetureNp.size)
    
    return fetureNp

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

def drawTriangle(ax, pts, color):
    p = Polygon(pts, closed=False, facecolor = color, alpha=.7)
    ax = plt.gca()
    ax.add_patch(p)

def displayVAgraph(valence, arousal, names, min, max):
    fig = plt.figure()
    fig.subplots_adjust(top=0.8)
    ax = fig.add_subplot()
    ax.set_ylabel('Arousal')
    ax.set_xlabel('Valence')
    ax.set_xlim([min,max])
    ax.set_ylim([min,max])
    center = (max+min)/2
    drawTriangle(ax, np.array([[max,max], [center,max], [center,center]]), 'orange')
    drawTriangle(ax, np.array([[max,max], [max,center], [center,center]]), 'yellow')
    drawTriangle(ax, np.array([[max,min], [max,center], [center,center]]), 'lawngreen')
    drawTriangle(ax, np.array([[max,min], [center,min], [center,center]]), 'lime')
    drawTriangle(ax, np.array([[min,min], [center,min], [center,center]]), 'dodgerblue')
    drawTriangle(ax, np.array([[min,min], [min,center], [center,center]]), 'cyan')
    drawTriangle(ax, np.array([[min,max], [min,center], [center,center]]), 'violet')
    drawTriangle(ax, np.array([[min,max], [center,max], [center,center]]), 'red')
    ax.axvline(x=0, linewidth=1, color='k')
    ax.axhline(y=0, linewidth=1, color='k')
    
    ax.plot(valence, arousal, 'ko')
    
    if names:
        for i,name in enumerate(names):
            plt.annotate(name, # this is the text
                    (valence[i],arousal[i]), # this is the point to label
                    textcoords="offset points", # how to position the text
                    xytext=(0,5), # distance from text to points (x,y)
                    ha='center') # horizontal alignment can be left, right or center
    plt.show()

def normalizacija(array, faktor, inMin, inMax):
    return 2.*(array - inMin)/inMax-faktor


# function to parse CSV file of valence and arousal
def parseDEAM(pathEmotions):
    #pathToEmotions = os.path.join(dirname,pathEmotions)
    emotions = pd.read_csv(pathEmotions, index_col=0, sep=',')
    return emotions

def rezultatiTestData(predictions_valence, predictions_arousal):
    from commonFunctions import parseDEAM
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import explained_variance_score
    from sklearn.metrics import r2_score
    from sklearn.metrics import max_error
    
    emotions = parseDEAM('dataset2/15_pravilne.csv')
    Y_valence = np.array(emotions['valence'].tolist())
    Y_arousal = np.array(emotions['arousal'].tolist())
    
    print('MSE:',round(mean_squared_error(Y_valence, np.array(predictions_valence)),4), round(mean_squared_error(Y_arousal, np.array(predictions_arousal)),4))
    print('MAE:',round(mean_absolute_error(Y_valence, np.array(predictions_valence)),4), round(mean_absolute_error(Y_arousal, np.array(predictions_arousal)),4))
    print('R2 :',round(r2_score(Y_valence, np.array(predictions_valence)),4), round(r2_score(Y_arousal, np.array(predictions_arousal)),4))
    print('EVS:',round(explained_variance_score(Y_valence, np.array(predictions_valence)),4), round(explained_variance_score(Y_arousal, np.array(predictions_arousal)),4))
    print('MXE:',round(max_error(Y_valence, np.array(predictions_valence)),4), round(max_error(Y_arousal, np.array(predictions_arousal)),4))