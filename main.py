import numpy as np
import matplotlib.pyplot as plt

# =====================
# Parametry 5G-like OFDM
# =====================
N = 64              # FFT size (jak mały PRB grid)
CP = 16             # cyclic prefix
mod_bits = 6        # 64-QAM

# =====================
# Wiadomość
# =====================
text = "Hello 5G OFDM baseband!"
bits_tx = np.unpackbits(np.frombuffer(text.encode('utf-8'), dtype=np.uint8))

num_bits = len(bits_tx)
pad_len = (N * mod_bits - (num_bits % (N * mod_bits))) % (N * mod_bits)
bits_tx = np.hstack([bits_tx, np.zeros(pad_len, dtype=np.uint8)])

num_symbols = len(bits_tx) // (N * mod_bits)

# =====================
# 64-QAM
# =====================
def qam64_mod(bits):
    bits = bits.reshape((-1, 6))

    def map3(b):
        return (4*b[0] + 2*b[1] + b[2])

    I = np.array([map3(b[:3]) for b in bits])
    Q = np.array([map3(b[3:]) for b in bits])

    I = 2*I - 7
    Q = 2*Q - 7

    return (I + 1j*Q) / np.sqrt(42)

symbols = qam64_mod(bits_tx)

# =====================
# RESOURCE GRID (jak w 5G)
# =====================
grid = symbols.reshape((num_symbols, N))

# (opcjonalnie: piloty jak DM-RS)
grid[:, ::16] = 1+1j  # proste "pilots"

# =====================
# OFDM - CP-OFDM (5G)
# =====================
# normalizacja IFFT jak w standardach OFDM
ofdm_time = np.fft.ifft(grid, axis=1) * np.sqrt(N)

# CP insertion
cp = ofdm_time[:, -CP:]
tx_signal = np.hstack([cp, ofdm_time])

tx_serial = tx_signal.flatten()

# =====================
# Widmo baseband vs RF
# =====================
fs = 1000
fc = 200

bb = tx_signal[0]
t = np.arange(len(bb)) / fs

rf = np.real(bb)*np.cos(2*np.pi*fc*t) - np.imag(bb)*np.sin(2*np.pi*fc*t)

def spectrum(x):
    S = np.fft.fftshift(np.fft.fft(x))
    f = np.fft.fftshift(np.fft.fftfreq(len(x), d=1/fs))
    return f, np.abs(S)

f_bb, S_bb = spectrum(bb)
f_rf, S_rf = spectrum(rf)

plt.figure(figsize=(10,5))
plt.plot(f_bb, S_bb, label="Baseband (5G CP-OFDM)")
plt.plot(f_rf, S_rf, label="RF shifted")
plt.grid()
plt.legend()
plt.title("Baseband → RF (5G idea)")
plt.show()

# =====================
# AWGN channel
# =====================
def awgn(x, snr_db):
    p = np.mean(np.abs(x)**2)
    snr = 10**(snr_db/10)
    npow = p/snr
    noise = np.sqrt(npow/2)*(np.random.randn(*x.shape)+1j*np.random.randn(*x.shape))
    return x + noise

rx = awgn(tx_serial, 30)

# =====================
# Receiver
# =====================
rx_mat = rx.reshape(tx_signal.shape)
rx_no_cp = rx_mat[:, CP:]
rx_fft = np.fft.fft(rx_no_cp, axis=1) / np.sqrt(N)

# usunięcie pilotów
rx_fft[:, ::16] = 0

plt.scatter(rx_fft.real, rx_fft.imag, alpha=0.5)
plt.title("Konstelacja po kanale (5G-like)")
plt.grid()
plt.show()

# =====================
# Demod 64-QAM
# =====================
def qam64_demod(x):
    x = x * np.sqrt(42)
    I, Q = np.real(x), np.imag(x)

    def demap(v):
        v = np.clip(np.round((v + 7)/2), 0, 7).astype(int)
        return np.stack([(v>>2)&1, (v>>1)&1, v&1], axis=1)

    return np.hstack([demap(I), demap(Q)]).reshape(-1)

rx_bits = qam64_demod(rx_fft.flatten())
rx_bits = rx_bits[:num_bits]

# =====================
# BER
# =====================
ber = np.mean(bits_tx[:num_bits] != rx_bits)
print("BER:", ber)

# =====================
# Text
# =====================
rx_bytes = np.packbits(rx_bits)
print("Odebrano:", rx_bytes.tobytes().decode('utf-8', errors='ignore'))
