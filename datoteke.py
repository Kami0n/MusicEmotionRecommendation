
from os import listdir
from os.path import isfile, join
import os
import csv
import pandas as pd
import numpy as np

from commonFunctions import displayVAgraph
from commonFunctions import normalizacija


dirname = os.path.dirname(__file__)
mypath = os.path.join(dirname, 'DatasetComb/audio')

onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
onlyfiles = {x.replace('.mp3', '') for x in onlyfiles}
#print(onlyfiles)

inputPath = os.path.join(dirname, 'DatasetComb/annotations/filtered_annotations_vse.csv')
included_cols = [0, 1, 3]
valenceArousal_1 = []
with open(inputPath, newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        #if row[0] in onlyfiles:
        content = list(row[i] for i in included_cols)
        valenceArousal_1.append(content)

valenceArousal_1.pop(0) 

ids_1 = np.array([int(row[0]) for row in valenceArousal_1])
valence_1 = np.array([float(row[1]) for row in valenceArousal_1])
arousal_1 = np.array([float(row[2]) for row in valenceArousal_1])

#displayVAgraph(valence_1, arousal_1, None, 1, 9)

valence_norm_1 = normalizacija(valence_1, 1, 1, 8)
arousal_norm_1 = normalizacija(arousal_1, 1, 1, 8)

#displayVAgraph(valence_norm_1, arousal_norm_1, None, -1, 1)



valenceArousal_2 = []
inputPath = os.path.join(dirname, 'DatasetComb/annotations/static_annotations.csv')
with open(inputPath, newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        valenceArousal_2.append(row)

valenceArousal_2.pop(0) 

ids_2 = np.array([int(row[0]) for row in valenceArousal_2])
valence_2 = np.array([float(row[1]) for row in valenceArousal_2])
arousal_2 = np.array([float(row[2]) for row in valenceArousal_2])

#displayVAgraph(valence_2, arousal_2, None, 0, 1)

valence_norm_2 = normalizacija(valence_2, 1, 0, 1)
arousal_norm_2 = normalizacija(arousal_2, 1, 0, 1)

#displayVAgraph(valence_norm_2, arousal_norm_2, None, -1, 1)

ids_2 = ids_2 + 3000



ids = np.concatenate((ids_1, ids_2))
valence = np.concatenate((valence_norm_1, valence_norm_2))
arousal = np.concatenate((arousal_norm_1, arousal_norm_2))

#displayVAgraph(valence, arousal, None, -1, 1)

d = { 'song_id': ids, 'valence_mean': valence, 'arousal_mean': arousal }
df = pd.DataFrame(data=d)
print(df)
df.to_csv('DatasetComb/annotations/annotations_all.csv',index=False,  float_format='%.4f')


onlyfiles = [int(i) for i in onlyfiles] 

listRazlicni = list(set(onlyfiles) - set(ids))
print( listRazlicni )
print( len(listRazlicni) )

# audio
# 1802
# -- 2596
# 794


# annotations
# 1802
# -- 2569
# 767

# manjkajoce v drugem
# [3459, 3205, 3206, 3464, 3596, 3085, 3095, 3352, 3226, 3610, 3866, 3101, 3359, 3110, 3371, 3118, 3384, 3064, 3264, 3392, 3011, 3404, 3278, 3291, 3687, 3311, 3320]
# 27