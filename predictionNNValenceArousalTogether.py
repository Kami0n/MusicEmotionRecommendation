import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("\n"*100)
clear = lambda: os.system('cls')
clear()

import numpy as np
import pandas as pd
import keras
import sklearn.preprocessing as skPre

from pickle import load
from os import listdir
from os.path import isfile, join
from commonFunctions import musicFeatureExtraction
from commonFunctions import displayVAgraph

#spremenljivke
from commonFunctions import subfolderName
from commonFunctions import showResults

subfolder = subfolderName

onlyfiles = [f for f in listdir(subfolder) if isfile(join(subfolder, f)) and ".mp3" in f]
#onlyfiles = ['Johnny Cash - Hurt.mp3','Ritchie Blackmore - Snowman.mp3']

allFeatures = []
for fileName in onlyfiles:
    print(fileName)
    data = musicFeatureExtraction(subfolder+str(fileName))
    allFeatures.append([fileName, data])

featuresdf = pd.DataFrame(allFeatures, columns=['id','features'])
featuresdf = featuresdf.set_index(['id'])
#print(featuresdf)

X_pred = np.array(featuresdf['features'].tolist()) # input
#print(X_pred)
#scaler = load(open('model_NN_valence_arousal_normalized_scaler.pkl', 'rb'))
#X_pred_scaled = scaler.transform(X_pred)
#print(X_pred_scaled)

model = keras.models.load_model('model_NN_10fold_together')

predictions = model.predict(X_pred)


# od tukaj do printa je samo za prikaz !

names = [x.replace('.mp3', '') for x in onlyfiles]
dataValenceArousal = pd.DataFrame(data=predictions, index=onlyfiles, columns=['valence', 'arousal']) 
print(dataValenceArousal.sort_index())

# displayVAgraph(valence, arousal, names, min, max)
#displayVAgraph( predictions.T[0], predictions.T[1], names, -1, 1 )

#if showResults:
#    from commonFunctions import rezultatiTestData
#    rezultatiTestData(predictions.T[0], predictions.T[1])

# save data


dataValenceArousal.to_pickle(subfolder+'pickle/predicted_valence_arousal.pkl')