import numpy as np
import matplotlib.pyplot as plt

# =====================
# Parametry OFDM
# =====================
N = 64              # liczba podnośnych
CP = 16             # cyclic prefix
mod_bits = 6        # 64-QAM

fs = 1000           # sampling
fc = 200            # nośna RF

# =====================
# Wiadomość
# =====================
text = "Cz ęść I Obrachunek S ł owo wst ę pne Adolfa Hitlera 9 pa ź dziernika I921 roku, w cztery lata od jej powstania, Narodowosocjalistyczna Niemiecka Partia Robotnicza zosta ł a rozwi ą zana, a jej dzia ł alno ść zakazana w ca ł ej Rzeszy. I kwietnia I924 roku wyrokiem S ą du Ludowego w Monachium zosta ł em skazany i osadzony w twierdzy Landsberg nad Lechem. To da ł o mi po latach nieprzerwanej pracy mo ż liwo ść przyst ą pienia do dzie ł a, którego wielu si ę domaga ł o, a które ja uwa ż a ł em za po ż yteczne dla ruchu. Tak wi ę c postanowi ł em wyja ś ni ć w tej ksi ąż ce cele naszego ruchu, a tak ż e przedstawi ć obraz jego rozwoju. Z niej b ę dzie si ę mo ż na wi ę cej nauczy ć ni ż z jakiejkolwiek czysto doktrynerskiej rozprawy naukowej. Da ł o mi to sposobno ść przedstawienia swojej osobowo ś ci na tyle, na ile jest to potrzebne do zrozumienia idei tej ksi ąż ki i rozwiania sfabrykowanej przez ż ydowsk ą pras ę legendy mojej osoby. T ą prac ą zwracam si ę nie do obcych, ale do tych stronników ruchu, którzy nale żą do niego sercem i pragn ą jego zrozumienia. Wiem, ż e ludzi ł atwiej mo ż na pozyska ć s ł owem mówionym ni ż pisanym i ż e ka ż dy wielki ruch na tej ziemi ro ś nie w si łę dzi ę ki mówcom, a nie wielkim pisarzom. Jednak ż e w celu stworzenia podstaw jakiej ś doktryny i jej ujednolicenia wewn ę trzne zasady musz ą zosta ć spisane. Mo ż e wi ę c ta ksi ąż ka stanie si ę kamieniem w ę gielnym naszego ruchu, do którego i ja wnios ę swój wk ł ad. Au"
#bits_tx = np.random.randint(0, 2, 10000)
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
# GRID (OFDM resource grid)
# =====================
grid = symbols.reshape((num_symbols, N))

# piloty (opcjonalnie)
#grid[:, ::16] = 1+1j


grid[0, :] = 1
# =====================
# OFDM modulacja
# =====================
ofdm_time = np.fft.ifft(grid, axis=1) * np.sqrt(N)

cp = ofdm_time[:, -CP:]
tx_signal = np.hstack([cp, ofdm_time])

# sygnał szeregowy (WAŻNE!)
tx_serial = tx_signal.flatten()

# =====================
# Upconversion (baseband → RF)
# =====================
t = np.arange(len(tx_serial)) / fs

rf = (np.real(tx_serial) * np.cos(2*np.pi*fc*t)
     -np.imag(tx_serial) * np.sin(2*np.pi*fc*t))

# =====================
# Widmo (duże FFT!)
# =====================
def spectrum(x):
    Nfft = 4096
    S = np.fft.fftshift(np.fft.fft(x, Nfft))
    f = np.fft.fftshift(np.fft.fftfreq(Nfft, d=1/fs))
    return f, np.abs(S)

f_bb, S_bb = spectrum(tx_serial)
f_rf, S_rf = spectrum(rf)

plt.figure(figsize=(10,5))
plt.plot(f_bb, S_bb, label="Baseband OFDM")
plt.plot(f_rf, S_rf, label="RF shifted")
plt.legend()
plt.grid()
plt.title("OFDM Spectrum (wiele podnośnych widoczne)")
plt.show()

# =====================
# SPECTROGRAM (🔥 najlepsze!)
# =====================
plt.figure(figsize=(10,5))
plt.specgram(tx_serial, NFFT=128, Fs=fs, noverlap=64)
plt.title("Spectrogram OFDM (widać podnośne)")
plt.xlabel("Czas")
plt.ylabel("Częstotliwość")
plt.colorbar()
plt.show()

# =====================
# AWGN
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

# usuwamy piloty
rx_fft[:, ::16] = 0

# konstelacja
# konstelacja wielu podnośnych (kolory)
for k in range(4):  # np. 8 pierwszych podnośnych
    plt.scatter(rx_fft[:, k+1].real,
                rx_fft[:, k+1].imag,
                label=f"podnośna {k+1}",
                alpha=0.6)

plt.legend()
plt.grid()
plt.title("Różne podnośne (kolorami)")
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
# Tekst
# =====================
rx_bytes = np.packbits(rx_bits)
print("Odebrano:", rx_bytes.tobytes().decode('utf-8', errors='ignore'))
print("liczba podśnych :", N)
