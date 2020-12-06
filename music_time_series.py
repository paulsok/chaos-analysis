# Simple module to convert music (*.wav) to NumPy array and plot the time series
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np


def musdat(file='Toccata.wav'):
    """
    Write a wav file into numpy float array and plot signal.
    _____
    Input (file : string {path to *.wav file})
    Output (data : array {Data read from wav file})
    """
    samplerate, data = wavfile.read(file)
    print(f"number of channels = {data.shape[1]}")

    length = data.shape[0] / samplerate
    print(f"length = {length}s")
    time = np.linspace(0., length, data.shape[0])
    
    plt.plot(time, data[:, 0], label="Left channel")
    plt.plot(time, data[:, 1], label="Right channel")
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.show()

    return data[:, 0]
