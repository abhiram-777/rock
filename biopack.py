# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 19:51:31 2024

@author: gokul
"""

from scipy import signal as signal
import numpy as np
import matplotlib.pyplot as plt


ecg = np.loadtxt('C:\studymaterial   me\5th semester\LAB/ecg data.txt')
x = ecg[1:800]  # Extracting a portion of the PPG signal for analysis
n = len(x)
nx = np.arange(0, n, 1)  

plt.figure(figsize=(10, 8))
plt.subplot(4, 1, 1)
plt.plot(nx, x, label="ecg Signal", color="green")
plt.title("Segment of ecg Signal")
plt.xlabel("Sample Number")
plt.ylabel("Amplitude")
plt.grid(True)

plt.subplot(4,1,2)
fs=10000
f=2000
t=np.arange(0,.0799,1/fs)
y=np.sin(2*np.pi*f*t)
plt.plot(t,y)


plt.subplot(4,1,3)
sum=x+y
plt.plot(nx,sum)

plt.subplot(4,1,4)
b, a = signal.butter(5, 50, 'low', analog=False,fs=500)
#w, h = signal.freqs(b, a)
Y = signal.lfilter(b, a, sum)
plt.plot(nx,Y,color="red")
# plt.semilogx(w, 20 * np.log10(abs(h)))
plt.title('Low Pass')
plt.xlabel('Frequency')
plt.grid(which='both', axis='both')
plt.plot()