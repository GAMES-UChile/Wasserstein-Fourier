import numpy as np


def PSD2Signal(frequencies, times, psd):
    signal = np.zeros(len(times))
    for f, p in zip(frequencies, psd): 
        signal += np.sqrt(p)*np.cos(f*times)
    return signal

def rect(x):
    res = np.zeros(x.shape)
    res[np.abs(x) <= 0.5] = 1
    return res

def fourierTransform(frequencies,times,signal):  
    four = np.zeros(len(frequencies))*1j
    for t, s in zip(times, signal):
        four += s*np.exp(-t*frequencies*1j*2*np.pi)        
    return four


def Signal2NPSD(frequencies, times, signal):   
    npsd = np.zeros(len(frequencies))
    npsd = np.abs(fourierTransform(frequencies,times,signal))
    npsd /= np.sum(npsd)    
    return npsd


def PF_to_signal(frequencies, times, inst_amplitude, inst_phase):
    signal = np.zeros(len(times))*1j
    for f, ia, ip in zip(frequencies, inst_amplitude, inst_phase):
        signal += ia * np.exp(1j*(ip + f*times*2*np.pi))        
    return signal
