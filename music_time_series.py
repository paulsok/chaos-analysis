from scipy.io import wavfile


def musdat(file='Toccata.wav'):

    samplerate, data = wavfile.read(file)

    print(f"Shape of the record: {data.shape}")
    print(f"Samplerate: {samplerate}")
    print(f"Lenght of the one channel: {len(data[:,0])}")

    return data[:, 0]
