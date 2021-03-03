import os
import joblib
import tensorflow as tf
import librosa
import numpy as np


#verifier path fichier, path scaler, path_model

PATH = './'

#choisir un son d'entre la liste, extraire le type de machine et son label (real_class)
#machine = 'pump'  

def load_audio(audio_path):
    audio, fe = librosa.load(audio_path, sr=None, duration=10)  # limiter l'import d'audio a 10sec
    return audio, fe 



def get_features(file_name, machine,              
                 n_mels=128, 
                 n_fft=1024,
                 hop_length=512,
                 power=2.0):
    
    # load audio
    audio, fe = load_audio(file_name)
    
    # generate melspectrogram using librosa
    mel_spectrogram = librosa.feature.melspectrogram(y=audio,
                                                     sr=fe,
                                                     n_fft=n_fft,
                                                     hop_length=hop_length,
                                                     n_mels=n_mels,
                                                     power=power)
    # MFECs
    log_mel_energy = librosa.core.power_to_db(mel_spectrogram)
    
    vector_array = log_mel_energy.T    # transpose for the time axis to be in axis=0  dim(313, 128)
    
    #load scaler
    scaler_name = 'scaler_'+ machine +'.gz'
    scaler = joblib.load(PATH + 'scalers/' + scaler_name)
    
    vector_array = scaler.transform(vector_array)

    return vector_array



# label predicted
def predicted_class(file_path, machine):
    # load model from file
    path_model = os.path.join(PATH, 'modelsAE', machine)
    model = tf.saved_model.load(path_model+'/')


    # Prediction du label d'un son
    test_pred = 0
    # get audio features and normalise
    vector_array = get_features(file_path, machine)
    length, _ = vector_array.shape
    dim = 32
    step = 3
    idex = np.arange(length-dim+step, step=step)
    for idx in range(len(idex)):
        start = min(idex[idx], length - dim)
        vector = vector_array[start:start+dim,:]
        vector = vector.reshape((1, vector.shape[0], vector.shape[1]))
        if idx==0:
            batch = vector
        else:
            batch = np.concatenate((batch, vector))
    # add channels dimension
    data = batch.reshape((batch.shape[0], batch.shape[1], batch.shape[2], 1))
    # calculate prediction
    errors = np.mean(np.square(data - model(data)), axis=-1)
    test_pred = np.mean(errors)
        
    seuils = {'slider':0.35406, 'fan':0.99742, 'pump':0.432354, 'valve':0.31110253, 'ToyCar':0.415157, 'ToyConveyor': 0.3927323 }
    seuil = seuils[machine]
        
    
    return 'anomaly' if test_pred>seuil else 'normal'
        
    

# example prediction of label:
# predicted_class('./pump_anomaly_id_00_00000004.wav', 'pump')   
# returns 'anomaly'