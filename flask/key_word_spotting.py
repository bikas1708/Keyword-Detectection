
import tensorflow.keras as keras
import numpy as np
import librosa


MODEL_PATH = "model.h5"
NUM_SAMPLES_TO_CONSIDER = 22050 #1 sec worth of sound


class _Keyword_Spotting_Service:

    model = None
    _mappings = [
        "down",
        "go",
        "left",
        "off",
        "on",
        "right",
        "stop",
        "up",
        "yes"
    ]
    _instance = None

    def predict(self, file_path):

        #extract the MFCCs
        MFCCs = self.preprocess(file_path)  #Shape = (#segments, #co-eff)


        #convert 2D MFCCs array into 4D array  -> (#samples, #segments, #co-eff, #channels))
        MFCCs =  MFCCs[np.newaxis, ..., np.newaxis]


        #make prediction
        predictions = self.model.predict(MFCCs)  #[ [0.1, 0.6, 0.1, ...]   ]
        predicted_index = np.argmax(predictions)
        predicted_keyword = self._mappings[predicted_index]

        return predicted_keyword


    def preprocess(self, file_path, n_mfcc = 13,  n_fft = 2048, hop_length = 512):

        #load the audio file
        signal, sr = librosa.load(file_path)

        #ensure consistency in the audio file length
        if len(signal) > NUM_SAMPLES_TO_CONSIDER:
            signal = signal[:NUM_SAMPLES_TO_CONSIDER]

        #extract the MFCCs
        MFCCs = librosa.feature.mfcc(signal, n_mfcc = n_mfcc, n_fft = n_fft, hop_length = hop_length)

        return MFCCs.T


def Keyword_Spotting_Service():

    #ensure that we only have 1 instance of KSS

    if _Keyword_Spotting_Service._instance is None:
        _Keyword_Spotting_Service._instance = _Keyword_Spotting_Service()
        _Keyword_Spotting_Service.model = keras.models.load_model(MODEL_PATH)
    return _Keyword_Spotting_Service._instance


if __name__ == "__main__":

    kss = Keyword_Spotting_Service()

    keyword1 =  kss.predict("test/down.wav")
    keyword2 = kss.predict("test/awawd.wav")

    print(f"Predicted Keywords :{keyword1}, {keyword2}")