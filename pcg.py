


import scipy.signal as signal  
import matplotlib.pyplot as plt  
import numpy as np  
from scipy.io import wavfile

 
sr,y = wavfile.read(r"C:\Users\\Downloads\Eko Recording Sound Export 09_11_2025 3_40 PM.wav")


start_sec, end_sec = 8, 10  
start_sample = int(start_sec * sr)  
end_sample = int(end_sec * sr)  
segment = y[start_sample:end_sample]  
time = np.linspace(start_sec, end_sec, len(segment))  
def bandpass_filter(sig, lowcut, highcut, sr, order=4):  
    nyquist = 0.5 * sr  
    low = lowcut / nyquist  
    high = highcut / nyquist  
    b, a = signal.butter(order, [low, high], btype='band')  
    return signal.filtfilt(b, a, sig)  
filtered_segment = bandpass_filter(segment, 20, 200, sr)  
s1_band = bandpass_filter(filtered_segment, 20, 60, sr)  
s2_band = bandpass_filter(filtered_segment, 60, 150, sr)  
s1_env = np.abs(signal.hilbert(s1_band))  
s2_env = np.abs(signal.hilbert(s2_band))  
th = 0.78 * np.max(s1_env)  
pad = int(0.06 * sr)  
mask_s1 = s1_env >= th  
mask_s1 = np.convolve(mask_s1.astype(int), np.ones(2 * pad + 1, dtype=int), mode='same') > 0  
s1_env = s1_env * mask_s1  
mask_s2 = s2_env >= th  
mask_s2 = np.convolve(mask_s2.astype(int), np.ones(2 * pad + 1, dtype=int), mode='same') > 0  
s2_env = s2_env * mask_s2  
plt.figure(figsize=(12, 8))  
plt.plot(time, filtered_segment, label="Filtered PCG", color='olive')  
plt.plot(time, s1_env, label="S1 Envelope", color='blue')  
plt.plot(time, s2_env, label="S2 Envelope", color='red')  
plt.title("Frequency-Band–Based S1/S2 Detection")  
plt.xlabel("Time (s)")  
plt.ylabel("Amplitude / Normalized Energy")  
plt.legend()  
plt.grid(True)  
plt.tight_layout()  
plt.show()  
plt.figure(figsize=(12, 8))  
plt.subplot(3, 1, 1)  
plt.plot(time, filtered_segment, color='black')  
plt.title("Filtered PCG Signal")  
plt.xlabel("Time (s)")  
plt.ylabel("Amplitude")  
plt.grid(True)  
s1 = mask_s1 * filtered_segment  
s2 = mask_s2 * filtered_segment  
plt.subplot(3, 1, 2)  
plt.plot(time, s1, color='blue')  
plt.title("S1 Segments")  
plt.xlabel("Time (s)")  
plt.ylabel("Amplitude")  
plt.grid(True)  
plt.subplot(3, 1, 3)  
plt.plot(time, s2, color='red')  
plt.title("S2 Segments")  
plt.xlabel("Time (s)")  
plt.ylabel("Amplitude")  
plt.grid(True)  
plt.tight_layout()  
plt.show()  
m = np.max(s1)  
t = m * 0.6  
X = np.zeros(len(time))  
Y = np.zeros(len(time))  
for i in range(1, len(time) - 1):  
    if s1_env[i] >= t:  
            if s1_env[i] > s1_env[i + 1] and s1_env[i] > s1_env[i - 1]:  
                    X[i] = i  
                    Y[i] = s1_env[i]  
X = [i for i in X if i != 0]  
Y = [i for i in Y if i != 0]  
interval = np.diff(X)    

x = int(np.mean(interval))  
x = x / sr  
hr = 60 / x  
print(f"\n Estimated Heart Rate: {hr:.2f} BPM")  
  

 
