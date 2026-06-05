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
bits_tx = np.hstack([bits_tx, np.zeros(pad_len, dtype=np.int8)])

num_symbols = len(bits_tx) // (N * mod_bits)

# =====================
# 64-QAM
# =====================
def qam64_mod(bits):
    bits = bits.reshape((-1, 6))

    def map_I(b):
        return (1 - 2*b[0]) * (4 - (1 - 2*b[2]) * (2 - (1 - 2*b[4])))
    
    def map_Q(b):
        return (1 - 2*b[1]) * (4 - (1 - 2*b[3]) * (2 - (1 - 2*b[5])))

    I = np.array([map_I(b) for b in bits])
    Q = np.array([map_Q(b) for b in bits])

    return (I + 1j*Q) / np.sqrt(42)

symbols = qam64_mod(bits_tx)

# ===================================
# GRID (OFDM resource grid) + pilots
# ===================================
pilot_carriers = np.arange(0, N, 16)
data_carriers = np.setdiff1d(np.arange(N), pilot_carriers)

data_per_ofdm = len(data_carriers)
num_ofdm = len(symbols) // data_per_ofdm
ofdm_data = symbols[:num_ofdm*data_per_ofdm]

grid = np.zeros((num_ofdm, N), dtype=complex)
grid[:, pilot_carriers] = 1 + 1j
grid[:, data_carriers] = ofdm_data.reshape((-1, data_per_ofdm))

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
rx_data = rx_fft[:, data_carriers]

# konstelacja
# konstelacja wielu podnośnych (kolory)
for k in range(4):  # np. 8 pierwszych podnośnych
    plt.scatter(rx_data[:, k+1].real,
                rx_data[:, k+1].imag,
                label=f"podnośna {k+1}",
                alpha=0.6)

plt.legend()
plt.grid()
plt.title("Różne podnośne (kolorami)")
plt.show()

# =====================
# Demod 64-QAM
# =====================
def qam64_demod(symbols):
    symbols = symbols * np.sqrt(42)

    I = np.real(symbols)
    Q = np.imag(symbols)

    levels = np.array([-7, -5, -3, -1, 1, 3, 5, 7])

    # tabela odwrotna do modulatora
    level_to_bits = {
        -7: np.array([1,1,1]),
        -5: np.array([1,1,0]),
        -3: np.array([1,0,0]),
        -1: np.array([1,0,1]),
        1: np.array([0,0,1]),
        3: np.array([0,0,0]),
        5: np.array([0,1,0]),
        7: np.array([0,1,1]),
    }

    bits = []
    for i_val, q_val in zip(I, Q):
        # najbliższy poziom
        i_level = levels[np.argmin(np.abs(levels - i_val))]
        q_level = levels[np.argmin(np.abs(levels - q_val))]

        bI = level_to_bits[i_level]
        bQ = level_to_bits[q_level]

        bits.append(np.array([
            bI[0], bQ[0],
            bI[1], bQ[1],
            bI[2], bQ[2]
        ]))

    return np.array(bits).reshape(-1)

rx_bits = qam64_demod(rx_data.flatten())
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