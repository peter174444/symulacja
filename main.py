import numpy as np
import matplotlib.pyplot as plt

# =====================
# Parametry 5G-like OFDM
# =====================
N = 64
CP = 16
mod_bits = 6  # 64-QAM

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
# RESOURCE GRID
# =====================
grid = symbols.reshape((num_symbols, N))
grid[:, ::16] = 1+1j  # piloty

# =====================
# OFDM (IFFT + CP)
# =====================
ofdm_time = np.fft.ifft(grid, axis=1) * np.sqrt(N)

cp = ofdm_time[:, -CP:]
tx_signal = np.hstack([cp, ofdm_time])
tx_serial = tx_signal.flatten()

# =====================
# Nieliniowość PA
# =====================
def nonlinear_pa(x, alpha=1.0, beta=0.005):
    return alpha * x - beta * (np.abs(x)**2) * x

tx_nl = nonlinear_pa(tx_serial)

# =====================
# Widmo
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
f_nl, S_nl = spectrum(tx_nl[:len(bb)])

plt.figure(figsize=(10,5))
plt.plot(f_bb, S_bb, label="Baseband")
plt.plot(f_rf, S_rf, label="RF")
plt.plot(f_nl, S_nl, label="Po nieliniowości")
plt.grid()
plt.legend()
plt.title("Widmo sygnału")
plt.show()

# =====================
# Kanał AWGN
# =====================
def awgn(x, snr_db):
    p = np.mean(np.abs(x)**2)
    snr = 10**(snr_db/10)
    npow = p/snr
    noise = np.sqrt(npow/2)*(np.random.randn(*x.shape)+1j*np.random.randn(*x.shape))
    return x + noise

rx = awgn(tx_nl, 20)

# =====================
# Receiver
# =====================
rx_mat = rx.reshape(tx_signal.shape)
rx_no_cp = rx_mat[:, CP:]
rx_fft = np.fft.fft(rx_no_cp, axis=1) / np.sqrt(N)

rx_fft[:, ::16] = 0  # usuń piloty

# =====================
# Konstelacje
# =====================

# TX idealny
tx_fft = np.fft.fft(ofdm_time, axis=1) / np.sqrt(N)

# po nieliniowości (bez szumu)
tx_nl_mat = tx_nl.reshape(tx_signal.shape)
tx_nl_no_cp = tx_nl_mat[:, CP:]
tx_nl_fft = np.fft.fft(tx_nl_no_cp, axis=1) / np.sqrt(N)

plt.figure(figsize=(15,4))

plt.subplot(1,3,1)
plt.scatter(tx_fft.real, tx_fft.imag, alpha=0.4)
plt.title("Ideal TX")
plt.grid()

plt.subplot(1,3,2)
plt.scatter(tx_nl_fft.real, tx_nl_fft.imag, alpha=0.4)
plt.title("Po nieliniowości (PA)")
plt.grid()

plt.subplot(1,3,3)
plt.scatter(rx_fft.real, rx_fft.imag, alpha=0.4)
plt.title("Po kanale (AWGN + PA)")
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