import numpy as np

def GetDFTWithShiftsAndFrequencyScaling(dt_signal):
    response = np.fft.fft(dt_signal)
    response = np.fft.fftshift(response)
    frequency = np.fft.fftfreq(dt_signal.shape[-1])
    frequency = np.fft.fftshift(frequency)
    w = 2 * np.pi * frequency
    return (w, response)
