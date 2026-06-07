import numpy as np
import matplotlib.pyplot as plt

can_save = False
can_show = True

# =====================
# Parametry OFDM / RF
# =====================
N        = 64          # FFT size
delta_f  = 15e3        # 15 kHz
fs       = N * delta_f
fc       = 240e3       # nośna RF (tylko do wykresów)
CP       = 5
mod_bits = 6           # 64-QAM

# 0 = brak EQ, 1 = ZF, 2 = MMSE
EQ_MODE = 2

# =====================
# Wiadomość → bity
# =====================
text = "Cz ęść I Obrachunek S ł owo wst ę pne Adolfa Hitlera 9 pa ź dziernika I921 roku, w cztery lata od jej powstania, Narodowosocjalistyczna Niemiecka Partia Robotnicza zosta ł a rozwi ą zana, a jej dzia ł alno ść zakazana w ca ł ej Rzeszy. I kwietnia I924 roku wyrokiem S ą du Ludowego w Monachium zosta ł em skazany i osadzony w twierdzy Landsberg nad Lechem. To da ł o mi po latach nieprzerwanej pracy mo ż liwo ść przyst ą pienia do dzie ł a, którego wielu si ę domaga ł o, a które ja uwa ż a ł em za po ż yteczne dla ruchu. Tak wi ę c postanowi ł em wyja ś ni ć w tej ksi ąż ce cele naszego ruchu, a tak ż e przedstawi ć obraz jego rozwoju. Z niej b ę dzie si ę mo ż na wi ę cej nauczy ć ni ż z jakiejkolwiek czysto doktrynerskiej rozprawy naukowej. Da ł o mi to sposobno ść przedstawienia swojej osobowo ś ci na tyle, na ile jest to potrzebne do zrozumienia idei tej ksi ąż ki i rozwiania sfabrykowanej przez ż ydowsk ą pras ę legendy mojej osoby. T ą prac ą zwracam si ę nie do obcych, ale do tych stronników ruchu, którzy nale żą do niego sercem i pragn ą jego zrozumienia. Wiem, ż e ludzi ł atwiej mo ż na pozyska ć s ł owem mówionym ni ż pisanym i ż e ka ż dy wielki ruch na tej ziemi ro ś nie w si łę dzi ę ki mówcom, a nie wielkim pisarzom. Jednak ż e w celu stworzenia podstaw jakiej ś doktryny i jej ujednolicenia wewn ę trzne zasady musz ą zosta ć spisane. Mo ż e wi ę c ta ksi ąż ka stanie si ę kamieniem w ę gielnym naszego ruchu, do którego i ja wnios ę swój wk ł ad. Au"

bits_tx = np.unpackbits(np.frombuffer(text.encode('utf-8'), dtype=np.uint8))
num_bits_orig = len(bits_tx)

# =====================
# 64-QAM mod/demod
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

def qam64_demod(symbols):
    symbols = symbols.reshape(-1) * np.sqrt(42)

    I = np.real(symbols)
    Q = np.imag(symbols)

    levels = np.array([-7, -5, -3, -1, 1, 3, 5, 7])
    level_to_bits = {
        -7: np.array([1,1,1]),
        -5: np.array([1,1,0]),
        -3: np.array([1,0,0]),
        -1: np.array([1,0,1]),
        1:  np.array([0,0,1]),
        3:  np.array([0,0,0]),
        5:  np.array([0,1,0]),
        7:  np.array([0,1,1]),
    }

    bits = []
    for i_val, q_val in zip(I, Q):
        i_level = levels[np.argmin(np.abs(levels - i_val))]
        q_level = levels[np.argmin(np.abs(levels - q_val))]
        bI = level_to_bits[i_level]
        bQ = level_to_bits[q_level]
        bits.append([bI[0], bQ[0], bI[1], bQ[1], bI[2], bQ[2]])

    return np.array(bits).reshape(-1)

# =====================
# GRID (5 RB, piloty)
# =====================
active_subcarriers = np.arange(0, 60)
pilot_carriers = np.concatenate([
    rb*12 + np.arange(0, 12, 2) for rb in range(5)
])
data_carriers = np.setdiff1d(active_subcarriers, pilot_carriers)

data_per_ofdm = len(data_carriers)
bits_per_ofdm = data_per_ofdm * mod_bits

pad_len = (bits_per_ofdm - (len(bits_tx) % bits_per_ofdm)) % bits_per_ofdm
bits_tx_pad = np.hstack([bits_tx, np.zeros(pad_len, dtype=np.int8)])

symbols = qam64_mod(bits_tx_pad)
num_ofdm = len(symbols) // data_per_ofdm
symbols = symbols[:num_ofdm * data_per_ofdm]

grid = np.zeros((num_ofdm, N), dtype=complex)
grid[:, pilot_carriers] = 1 + 1j
grid[:, data_carriers] = symbols.reshape(num_ofdm, data_per_ofdm)

# =====================
# OFDM modulacja (baseband)
# =====================
ofdm_time = np.fft.ifft(grid, axis=1) * np.sqrt(N)
cp = ofdm_time[:, -CP:]
tx_signal = np.hstack([cp, ofdm_time])
tx_serial = tx_signal.flatten()

# =====================
# RF tylko do wykresów
# =====================
t = np.arange(len(tx_serial)) / fs
rf = (np.real(tx_serial) * np.cos(2*np.pi*fc*t)
     -np.imag(tx_serial) * np.sin(2*np.pi*fc*t))

def spectrum(x):
    Nfft = 4096
    S = np.fft.fftshift(np.fft.fft(x, Nfft))
    f = np.fft.fftshift(np.fft.fftfreq(Nfft, d=1/fs))
    return f, np.abs(S)

f_bb, S_bb = spectrum(tx_serial)
f_rf, S_rf = spectrum(rf)

if can_save or can_show:
    plt.figure(figsize=(10,5))
    plt.plot(f_bb, S_bb, label="Baseband OFDM")
    plt.plot(f_rf, S_rf, label="RF shifted")
    plt.legend()
    plt.grid()
    plt.title("OFDM Spectrum")
    if can_save:
        plt.savefig("plots/OFDM Spectrum.png")
    if can_show:
        plt.show()
    plt.close()

def psd(x):
    Nfft = 4096
    X = np.fft.fft(x, Nfft)
    X = np.fft.fftshift(X)
    Pxx = (np.abs(X)**2) / (Nfft * fs)
    f = np.fft.fftshift(np.fft.fftfreq(Nfft, d=1/fs))
    return f, Pxx

f, Pxx = psd(tx_serial)
Pxx_dB = 10 * np.log10(Pxx + 1e-20)

if can_save or can_show:
    plt.plot(f, Pxx_dB)
    plt.xlabel("Częstotliwość [Hz]")
    plt.ylabel("PSD [dB/Hz]")
    plt.grid()
    if can_save:
        plt.savefig("plots/PSD.png")
    if can_show:
        plt.show()
    plt.close()

if can_save or can_show:
    plt.figure(figsize=(10,5))
    plt.specgram(tx_serial, NFFT=128, Fs=fs, noverlap=64)
    plt.title("Spectrogram OFDM")
    plt.xlabel("Czas")
    plt.ylabel("Częstotliwość")
    plt.colorbar()
    if can_save:
        plt.savefig("plots/Spectrogram OFDM.png")
    if can_show:
        plt.show()
    plt.close()

# =====================
# Kanał w dziedzinie częstotliwości
# =====================
h = np.array([0.9+0j, 0.4-0.3j, 0.2+0.1j])
H = np.fft.fft(h, N)

X_f = np.fft.fft(ofdm_time, axis=1) / np.sqrt(N)
Y_f = X_f * H
y_time = np.fft.ifft(Y_f, axis=1) * np.sqrt(N)

y_cp = np.hstack([y_time[:, -CP:], y_time])
rx_serial = y_cp.reshape(-1)

# =====================
# AWGN (baseband)
# =====================
def awgn(x, snr_db):
    p = np.mean(np.abs(x)**2)
    snr = 10**(snr_db/10)
    npow = p/snr
    noise = np.sqrt(npow/2)*(np.random.randn(*x.shape)+1j*np.random.randn(*x.shape))
    return x + noise

rx_serial = awgn(rx_serial, 30)

# =====================
# Receiver: CP → FFT
# =====================
rx_mat = rx_serial.reshape(num_ofdm, N+CP)
rx_no_cp = rx_mat[:, CP:]
Y_rx = np.fft.fft(rx_no_cp, axis=1) / np.sqrt(N)

# =====================
# LS estymacja z pilotów
# =====================
Y_pilots = Y_rx[:, pilot_carriers]
H_pilots = Y_pilots / (1 + 1j)

H_est = np.zeros_like(Y_rx, dtype=complex)
for n in range(num_ofdm):
    H_real = np.interp(active_subcarriers, pilot_carriers, H_pilots[n].real)
    H_imag = np.interp(active_subcarriers, pilot_carriers, H_pilots[n].imag)
    H_est[n, active_subcarriers] = H_real + 1j * H_imag

# =====================
# Equalizacja: brak / ZF / MMSE
# =====================
Y_eq = np.zeros_like(Y_rx)

if EQ_MODE == 0:
    print("Equalizer: BRAK")
    Y_eq = Y_rx.copy()

elif EQ_MODE == 1:
    print("Equalizer: ZF")
    Y_eq[:, active_subcarriers] = Y_rx[:, active_subcarriers] / H_est[:, active_subcarriers]

elif EQ_MODE == 2:
    print("Equalizer: MMSE")
    noise_var = 10**(-30/10)  # SNR=30 dB jak w awgn()
    Hloc = H_est[:, active_subcarriers]
    Yloc = Y_rx[:, active_subcarriers]
    Y_eq[:, active_subcarriers] = (np.conj(Hloc) / (np.abs(Hloc)**2 + noise_var)) * Yloc

else:
    raise ValueError("EQ_MODE must be 0, 1, or 2")

# =====================
# Wykres konstelacji wielu podnośnych
# =====================
if can_save or can_show:
    plt.figure(figsize=(10, 6))

    max_sub = min(30, len(data_carriers))  # np. 30 pierwszych podnośnych

    for idx, sc in enumerate(data_carriers[:max_sub]):
        plt.scatter(
            Y_eq[:, sc].real,
            Y_eq[:, sc].imag,
            s=8,
            alpha=0.6,
            label=f"podnośna {sc}"
        )

    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.grid()
    plt.title("Różne podnośne (kolorami)")
    plt.xlabel("Re")
    plt.ylabel("Im")

    if can_save:
        plt.savefig("plots/Subcarriers_Constellation.png", bbox_inches="tight")
    if can_show:
        plt.show()

    plt.close()
    
# =====================
# Demod 64-QAM
# =====================
rx_data = Y_eq[:, data_carriers].reshape(-1)
rx_bits = qam64_demod(rx_data)
rx_bits = rx_bits[:num_bits_orig]

# =====================
# BER + tekst
# =====================
ber = np.mean(bits_tx[:num_bits_orig] != rx_bits)
print("EQ_MODE:", EQ_MODE)
print("BER:", ber)

rx_bytes = np.packbits(rx_bits)
print("Odebrano:")
print(rx_bytes.tobytes().decode('utf-8', errors='ignore'))
print("liczba podnośnych:", N)
