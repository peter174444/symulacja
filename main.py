import numpy as np
import matplotlib.pyplot as plt

# Parametry OFDM
N = 64
CP = 16
mod_bits = 6  # 64-QAM

# Wiadomość
text = "Hello 5G world!"

# Tekst -> bity
bits_tx = np.unpackbits(np.frombuffer(text.encode('utf-8'), dtype=np.uint8))

# Padding
num_bits = len(bits_tx)
pad_len = (N * mod_bits - (num_bits % (N * mod_bits))) % (N * mod_bits)
bits_tx = np.hstack([bits_tx, np.zeros(pad_len, dtype=np.uint8)])

num_symbols = len(bits_tx) // (N * mod_bits)

# =====================
# 64-QAM modulacja
# =====================
def qam64_mod(bits):
    bits = bits.reshape((-1, 6))

    def map_3bits(b):
        return (2*b[0] + b[1]) * 2 + b[2]

    I = np.array([map_3bits(b[:3]) for b in bits])
    Q = np.array([map_3bits(b[3:]) for b in bits])

    I = 2*I - 7
    Q = 2*Q - 7

    return (I + 1j*Q) / np.sqrt(42)

symbols = qam64_mod(bits_tx)
symbols = symbols.reshape((num_symbols, N))

# Konstelacja przed kanałem
plt.scatter(np.real(symbols.flatten()), np.imag(symbols.flatten()), alpha=0.5)
plt.title("64-QAM przed kanałem")
plt.grid()
plt.show()

# OFDM
ofdm_time = np.fft.ifft(symbols, axis=1)
cp = ofdm_time[:, -CP:]
ofdm_with_cp = np.hstack([cp, ofdm_time])

# Kanał AWGN
def awgn(signal, snr_db):
    power = np.mean(np.abs(signal)**2)
    snr = 10**(snr_db/10)
    noise_power = power / snr
    noise = np.sqrt(noise_power/2) * (
        np.random.randn(*signal.shape) + 1j*np.random.randn(*signal.shape)
    )
    return signal + noise

snr_db = 20
rx_signal = awgn(ofdm_with_cp, snr_db)

# Odbiór
rx_no_cp = rx_signal[:, CP:]
rx_symbols = np.fft.fft(rx_no_cp, axis=1)

# Konstelacja po kanale
plt.scatter(np.real(rx_symbols.flatten()), np.imag(rx_symbols.flatten()), alpha=0.5)
plt.title("64-QAM po kanale")
plt.grid()
plt.show()

# Demodulacja
def qam64_demod(symbols):
    symbols = symbols * np.sqrt(42)
    I = np.real(symbols)
    Q = np.imag(symbols)

    def demap(val):
        val = np.clip(np.round((val + 7)/2), 0, 7).astype(int)
        b2 = val & 1
        b1 = (val >> 1) & 1
        b0 = (val >> 2) & 1
        return np.vstack([b0, b1, b2]).T

    bits_I = demap(I)
    bits_Q = demap(Q)

    return np.hstack([bits_I, bits_Q]).reshape(-1)

rx_bits = qam64_demod(rx_symbols.flatten())

# Usunięcie paddingu
rx_bits = rx_bits[:num_bits]

# Bity -> tekst
rx_bytes = np.packbits(rx_bits)
rx_text = rx_bytes.tobytes().decode('utf-8', errors='ignore')

print("Odebrana wiadomość:", rx_text)

# BER
ber = np.mean(bits_tx[:num_bits] != rx_bits)
print("BER:", ber)
