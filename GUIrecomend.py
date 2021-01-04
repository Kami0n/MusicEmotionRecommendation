import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

clear = lambda: os.system('cls')
clear()

import queue
import threading
import time

import numpy as np
import pandas as pd
import keras
import sklearn.preprocessing as skPre
from pickle import load
from os import listdir
from os.path import isfile, join

from commonFunctions import musicFeatureExtraction

import PySimpleGUI as sg


dataValenceArousalDf = pd.read_pickle('pickle/predicted_valence_arousal_GUI.pkl')

def zaznajEmocije(folderPath, gui_queue):
    
    onlyfiles = [f for f in listdir(folderPath) if isfile(join(folderPath, f)) and ".mp3" in f]
    allFeatures = []
    for st, fileName in enumerate(onlyfiles):
        gui_queue.put('Parsing music features: ' + fileName)
        data = musicFeatureExtraction(folderPath+'/'+str(fileName))
        allFeatures.append([fileName, data])
        
    featuresdf = pd.DataFrame(allFeatures, columns=['id','features'])
    featuresdf = featuresdf.set_index(['id'])
    X_pred = np.array(featuresdf['features'].tolist()) # input
    
    gui_queue.put('Loading model')
    
    model = keras.models.load_model('model_NN_10fold_together')
    
    gui_queue.put('Recognizig emotions')
    predictions = model.predict(X_pred)
    
    names = [x.replace('.mp3', '') for x in onlyfiles]
    dataValenceArousalDf = pd.DataFrame(data=predictions, index=onlyfiles, columns=['valence', 'arousal']) 
    dataValenceArousalDf.attrs['folderPath'] = folderPath
    dataValenceArousalDf.to_pickle('pickle/predicted_valence_arousal_GUI.pkl')
    gui_queue.put('** Done **')

def recomend(rslt_df, window, event):
    window.Element('text1').Update(value = '' )
    window.Element('text2').Update(value = '' )
    window.Element('text3').Update(value = '' )
    window.Element('text4').Update(value = '' )
    window.Element('text5').Update(value = '' )
    
    if(rslt_df.empty):
        txterr = 'No music in dataset fits selected emotion.'
        print(txterr)
        window.Element('textEmotion').Update(value='Selected emotion: '+event)
        window.Element('text1').Update(value=txterr)
    else:
        # priporoci 5 pesmi (random)
        #print(rslt_df )
        
        window.Element('textEmotion').Update(value='Selected emotion: '+event + ' Songs from folder: '+ dataValenceArousalDf.attrs['folderPath'] )
        
        if(rslt_df.shape[0] > 5):
            all_samples = rslt_df.sample(5)
            #print(all_samples)
        else:
            all_samples = rslt_df
        
        try:
            window.Element('text1').Update(value = all_samples.index[0] )
        except IndexError:
            None
        try:
            window.Element('text2').Update(value = all_samples.index[1] )
        except IndexError:
            None
        try:
            window.Element('text3').Update(value = all_samples.index[2] )
        except IndexError:
            None
        try:
            window.Element('text4').Update(value = all_samples.index[3] )
        except IndexError:
            None
        try:
            window.Element('text5').Update(value = all_samples.index[4] )
        except IndexError:
            None
        
        
######   ##     ## ####
##    ##  ##     ##  ##
##        ##     ##  ##
##   #### ##     ##  ##
##    ##  ##     ##  ##
##    ##  ##     ##  ##
######    #######  ####

def the_gui():
    """
    Starts and executes the GUI
    Reads data from a Queue and displays the data to the window
    Returns when the user exits / closes the window
    """

    gui_queue = queue.Queue()  # queue used to communicate between the gui and the threads

    textsize = 10
    btnsize = 10
    textWidth = 100
    layout = [
                [sg.Text('Select folder:'), sg.Input(), sg.FolderBrowse(), sg.Button('Recognize emotions', key='OK')],
                [
                    sg.Button('Excited',    button_color=('black', 'orange'),     size=(btnsize, 1), font=("Helvetica", textsize), key='Excited'),
                    sg.Button('Happy',      button_color=('black', 'yellow'),     size=(btnsize, 1), font=("Helvetica", textsize), key='Happy'),
                    sg.Button('Pleased',    button_color=('black', 'lawngreen'),  size=(btnsize, 1), font=("Helvetica", textsize), key='Pleased'),
                    sg.Button('Calm',       button_color=('black', 'lime'),       size=(btnsize, 1), font=("Helvetica", textsize), key='Calm'),
                    sg.Button('Bored',      button_color=('black', 'dodgerblue'), size=(btnsize, 1), font=("Helvetica", textsize), key='Bored'),
                    sg.Button('Sad',        button_color=('black', 'cyan'),       size=(btnsize, 1), font=("Helvetica", textsize), key='Sad'),
                    sg.Button('Frustrated', button_color=('black', 'violet'),     size=(btnsize, 1), font=("Helvetica", textsize), key='Frustrated'),
                    sg.Button('Angry',      button_color=('black', 'red'),        size=(btnsize, 1), font=("Helvetica", textsize), key='Angry')
                ],
                [sg.Text('', key="textEmotion", size=(textWidth,1))],
                [sg.Text('', key="text1", size=(textWidth,1))],
                [sg.Text('', key="text2", size=(textWidth,1))],
                [sg.Text('', key="text3", size=(textWidth,1))],
                [sg.Text('', key="text4", size=(textWidth,1))],
                [sg.Text('', key="text5", size=(textWidth,1))],
                [sg.Output(size=(textWidth, 15))]
            ]

    window = sg.Window('MER - Music Emotion Recognition').Layout(layout)

    # --------------------- EVENT LOOP ---------------------
    while True:
        event, values = window.Read(timeout=100)       # wait for up to 100 ms for a GUI event
        
        if event is None or event == 'Exit':
            break
        elif(event == 'OK'):
            try:
                window.Element('textEmotion').Update(value='Selected folder: '+values[0])
                window.Element('text1').Update(value='Wait, this might take a long time! (depends on number of songs in folder)')
                window.Refresh()
                #print('Selected folder: '+values[0] +'\n Wait, this might take a long time!')
                threading.Thread(target=zaznajEmocije, args=( values[0], gui_queue, ), daemon=True).start()
            except Exception as e:
                print('Error starting work thread. Did you input a valid # of seconds? You entered: %s' % values['_SECONDS_'])
                
        elif(event == 'Excited'):
            # valence > 0 | arousal > 0 | valence < arousal
            rslt_df = dataValenceArousalDf[ (dataValenceArousalDf['valence'] > 0) &
            (dataValenceArousalDf['arousal'] > 0) & (dataValenceArousalDf['valence'] < dataValenceArousalDf['arousal']) ]
            recomend(rslt_df, window, event)
        elif(event == 'Happy'):
            # valence > 0 | arousal > 0 | valence > arousal
            rslt_df = dataValenceArousalDf[ (dataValenceArousalDf['valence'] > 0) &
            (dataValenceArousalDf['arousal'] > 0) & (dataValenceArousalDf['valence'] > dataValenceArousalDf['arousal']) ]
            recomend(rslt_df, window, event)
        elif(event == 'Pleased'):
            # valence > 0 | arousal < 0 | valence > -arousal
            rslt_df = dataValenceArousalDf[ (dataValenceArousalDf['valence'] > 0) &
            (dataValenceArousalDf['arousal'] < 0) & (dataValenceArousalDf['valence'] > -dataValenceArousalDf['arousal']) ]
            recomend(rslt_df, window, event)
        elif(event == 'Calm'):
            # valence > 0 | arousal < 0 | valence < -arousal
            rslt_df = dataValenceArousalDf[ (dataValenceArousalDf['valence'] > 0) &
            (dataValenceArousalDf['arousal'] < 0) & (dataValenceArousalDf['valence'] < -dataValenceArousalDf['arousal']) ]
            recomend(rslt_df, window, event)
        elif(event == 'Bored'):
            # valence < 0 | arousal < 0 | valence > arousal
            rslt_df = dataValenceArousalDf[ (dataValenceArousalDf['valence'] < 0) &
            (dataValenceArousalDf['arousal'] < 0) & (dataValenceArousalDf['valence'] > dataValenceArousalDf['arousal']) ]
            recomend(rslt_df, window, event)
        elif(event == 'Sad'):
            # valence < 0 | arousal < 0 | valence < arousal
            rslt_df = dataValenceArousalDf[ (dataValenceArousalDf['valence'] < 0) &
            (dataValenceArousalDf['arousal'] < 0) & (dataValenceArousalDf['valence'] < dataValenceArousalDf['arousal']) ]
            recomend(rslt_df, window, event)
        elif(event == 'Frustrated'):
            # valence < 0 | arousal > 0 | valence < -arousal
            rslt_df = dataValenceArousalDf[ (dataValenceArousalDf['valence'] < 0) &
            (dataValenceArousalDf['arousal'] > 0) & (dataValenceArousalDf['valence'] < -dataValenceArousalDf['arousal']) ]
            recomend(rslt_df, window, event)
        elif(event == 'Angry'):
            # valence < 0 | arousal > 0 | valence > -arousal
            rslt_df = dataValenceArousalDf[ (dataValenceArousalDf['valence'] < 0) &
            (dataValenceArousalDf['arousal'] > 0) & (dataValenceArousalDf['valence'] > -dataValenceArousalDf['arousal']) ]
            recomend(rslt_df, window, event)
        
        # --------------- Check for incoming messages from threads  ---------------
        try:
            message = gui_queue.get_nowait()
        except queue.Empty:             # get_nowait() will get exception when Queue is empty
            message = None              # break from the loop if no more messages are queued up
        
        
        # if message received from queue, display the message in the Window
        if message:
            #print('Got a message back from the thread: ', message)
            window.Element('text3').Update(value = message )
            window.Refresh()
            
    # if user exits the window, then close the window and exit the GUI func
    window.Close()


##     ##    ###    #### ##    ##
###   ###   ## ##    ##  ###   ##
#### ####  ##   ##   ##  ####  ##
## ### ## ##     ##  ##  ## ## ##
##     ## #########  ##  ##  ####
##     ## ##     ##  ##  ##   ###
##     ## ##     ## #### ##    ##

if __name__ == '__main__':
    the_gui()
    print('Exiting Program')