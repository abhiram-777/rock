import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, sosfiltfilt, welch, detrend

# --- Load data and settings ---
eeg = np.loadtxt(r"C:\Users\gokul\Downloads\eeg.txt")


fs = 256  # sampling frequency (Hz)
t = np.arange(len(eeg)) / fs

# --- Basic cleaning ---
# remove DC offset
eeg = eeg - np.mean(eeg)
# remove linear trend (optional but useful)
eeg = detrend(eeg)
# gentle high-pass filter (0.2 Hz) for slow drift removal
sos = butter(4, 0.2, btype='high', fs=fs, output='sos')
eeg = sosfiltfilt(sos, eeg)

# --- Frequency bands to extract ---
bands = {
    'Delta (0.5–4 Hz)': (0.5, 4),
    'Theta (4–8 Hz)': (4, 8),
    'Alpha (8–13 Hz)': (8, 13),
    'Beta (13–30 Hz)': (13, 30),
    'Gamma (30–45 Hz)': (30, 45)
}

# --- Bandpass filter function (uses filtfilt for zero-phase) ---
def bandpass_filtfilt(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

# --- Filter each band ---
filtered = {}
for band_name, (low, high) in bands.items():
    filtered[band_name] = bandpass_filtfilt(eeg, low, high, fs, order=4)

# --- Plot time-domain signals ---
plt.figure(figsize=(12, 10))
plt.subplot(len(bands) + 1, 1, 1)
plt.plot(t, eeg, color='black')
plt.title('Raw EEG Signal (DC removed)')
plt.ylabel('Amplitude (μV)')

for i, (band_name, signal) in enumerate(filtered.items(), start=2):
    plt.subplot(len(bands) + 1, 1, i)
    plt.plot(t, signal)
    plt.title(band_name)
    plt.ylabel('μV')

plt.xlabel('Time (s)')
plt.tight_layout()
plt.show()

# --- Power Spectral Density (PSD) by band ---

# psd
plt.figure(figsize=(10, 6))
for band, signal in filtered.items():
    f, Pxx = welch(signal, fs, nperseg=1024)
    plt.semilogy(f, Pxx, label=band)
plt.title('Power Spectral Density by Band')
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD (V²/Hz)')
plt.legend()
plt.tight_layout()
plt.show()

