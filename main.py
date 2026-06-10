import numpy as np                      # biblioteka do obliczeń numerycznych (wektory, FFT, itp.)
import matplotlib.pyplot as plt         # biblioteka do rysowania wykresów
import random                           # generator losowych wartości
import string                           # zestaw znaków ASCII

# flagi sterujące zapisem i wyświetlaniem wykresów
can_save = False
can_show = True

# =====================
# Parametry OFDM / RF
# =====================
N        = 64          # liczba punktów FFT (liczba podnośnych OFDM)
delta_f  = 15e3        # odstęp między podnośnymi 15 kHz
fs       = N * delta_f # częstotliwość próbkowania (spełnia warunek OFDM)
fc       = 240e3       # częstotliwość nośna RF (do wizualizacji sygnału)
CP       = 5           # długość prefiksu cyklicznego
mod_bits = 6           # liczba bitów na symbol (64-QAM → 2^6 = 64)

# 0 = brak equalizacji, 1 = ZF, 2 = MMSE
EQ_MODE = 2

# =====================
# Wiadomość → bity
# =====================
text = "".join(random.choices(string.printable, k=1000))
# generuje losowy tekst (1000 znaków)

bits_tx = np.unpackbits(
    np.frombuffer(text.encode('utf-8'), dtype=np.uint8)
)
# zamiana tekstu → bajty → bity (ciąg 0/1)

num_bits_orig = len(bits_tx)  # zapamiętanie długości oryginalnych bitów

# =====================
# 64-QAM mod/demod
# =====================
def qam64_mod(bits):
    bits = bits.reshape((-1, 6))  # grupowanie bitów po 6 (1 symbol QAM)

    # mapowanie 3 bitów na część rzeczywistą (I)
    def map_I(b):
        return (1 - 2*b[0]) * (4 - (1 - 2*b[2]) * (2 - (1 - 2*b[4])))

    # mapowanie 3 bitów na część urojoną (Q)
    def map_Q(b):
        return (1 - 2*b[1]) * (4 - (1 - 2*b[3]) * (2 - (1 - 2*b[5])))

    I = np.array([map_I(b) for b in bits])  # generowanie składowej I
    Q = np.array([map_Q(b) for b in bits])  # generowanie składowej Q

    return (I + 1j*Q) / np.sqrt(42)  # normalizacja energii symbolu


def qam64_demod(symbols):
    symbols = symbols.reshape(-1) * np.sqrt(42)  # cofnięcie normalizacji

    I = np.real(symbols)  # część rzeczywista
    Q = np.imag(symbols)   # część urojona

    levels = np.array([-7, -5, -3, -1, 1, 3, 5, 7])  # poziomy 64-QAM

    # mapa poziom → 3 bity (Gray-like mapping)
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

    bits = []  # lista na zdekodowane bity

    for i_val, q_val in zip(I, Q):
        # wybór najbliższego poziomu dla I i Q
        i_level = levels[np.argmin(np.abs(levels - i_val))]
        q_level = levels[np.argmin(np.abs(levels - q_val))]

        # konwersja poziomów na bity
        bI = level_to_bits[i_level]
        bQ = level_to_bits[q_level]

        # składanie 6 bitów symbolu
        bits.append([bI[0], bQ[0], bI[1], bQ[1], bI[2], bQ[2]])

    return np.array(bits).reshape(-1)  # spłaszczenie do wektora bitów


# =====================
# GRID (5 RB, piloty)
# =====================
active_subcarriers = np.arange(0, 60)  # aktywne podnośne OFDM

# generowanie pilotów (co 2 subnośne w blokach RB)
pilot_carriers = np.concatenate([
    rb*12 + np.arange(0, 12, 2) for rb in range(5)
])

# podnośne danych = aktywne - piloty
data_carriers = np.setdiff1d(active_subcarriers, pilot_carriers)

data_per_ofdm = len(data_carriers)     # ile symboli na OFDM
bits_per_ofdm = data_per_ofdm * mod_bits  # ile bitów na OFDM

# dopasowanie długości bitów do całych ramek OFDM
pad_len = (bits_per_ofdm - (len(bits_tx) % bits_per_ofdm)) % bits_per_ofdm
bits_tx_pad = np.hstack([bits_tx, np.zeros(pad_len, dtype=np.int8)])

# modulacja QAM
symbols = qam64_mod(bits_tx_pad)

num_ofdm = len(symbols) // data_per_ofdm  # liczba symboli OFDM

# obcięcie do pełnych ramek
symbols = symbols[:num_ofdm * data_per_ofdm]

# siatka OFDM (czas × podnośne)
grid = np.zeros((num_ofdm, N), dtype=complex)

# wstawienie pilotów
grid[:, pilot_carriers] = 1 + 1j

# wstawienie danych
grid[:, data_carriers] = symbols.reshape(num_ofdm, data_per_ofdm)

# =====================
# OFDM modulacja (IFFT)
# =====================
ofdm_time = np.fft.ifft(grid, axis=1) * np.sqrt(N)

# =====================
# NIELINIOWOŚĆ ANTENY (PA)
# =====================
def antenna_nonlinearity(x, a1=1.0, a3=0.01):
    # model nieliniowy (AM/AM distortion)
    return x + a3 * x * np.abs(x)**2 + a1 * x

ofdm_time = antenna_nonlinearity(ofdm_time)

# dodanie prefixu cyklicznego
cp = ofdm_time[:, -CP:]

tx_signal = np.hstack([cp, ofdm_time])  # CP + sygnał
tx_serial = tx_signal.flatten()         # serializacja (1D)

# =====================
# RF tylko do wykresów
# =====================
t = np.arange(len(tx_serial)) / fs  # oś czasu

# modulacja na nośną RF (IQ → RF)
rf = (np.real(tx_serial) * np.cos(2*np.pi*fc*t)
     -np.imag(tx_serial) * np.sin(2*np.pi*fc*t))

def spectrum(x):
    Nfft = 4096  # liczba punktów FFT
    S = np.fft.fftshift(np.fft.fft(x, Nfft))  # widmo
    f = np.fft.fftshift(np.fft.fftfreq(Nfft, d=1/fs))  # oś częstotliwości
    return f, np.abs(S)

f_bb, S_bb = spectrum(tx_serial)  # widmo baseband
f_rf, S_rf = spectrum(rf)         # widmo RF

# wykres widma
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

# =====================
# PSD (gęstość widmowa mocy)
# =====================
def psd(x):
    Nfft = 4096
    X = np.fft.fft(x, Nfft)
    X = np.fft.fftshift(X)
    Pxx = (np.abs(X)**2) / (Nfft * fs)
    f = np.fft.fftshift(np.fft.fftfreq(Nfft, d=1/fs))
    return f, Pxx

f, Pxx = psd(tx_serial)
Pxx_dB = 10 * np.log10(Pxx + 1e-20)  # skala dB

# wykres PSD
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

# spektrogram
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
# Kanał (multipath w freq domain)
# =====================
h = np.array([0.9+0j, 0.4-0.3j, 0.2+0.1j])  # odpowiedź impulsowa kanału
H = np.fft.fft(h, N)  # odpowiedź częstotliwościowa

X_f = np.fft.fft(ofdm_time, axis=1) / np.sqrt(N)  # FFT sygnału
Y_f = X_f * H  # kanał (mnożenie w freq)
y_time = np.fft.ifft(Y_f, axis=1) * np.sqrt(N)

# dodanie CP po kanale
y_cp = np.hstack([y_time[:, -CP:], y_time])
rx_serial = y_cp.reshape(-1)

# =====================
# AWGN (szum)
# =====================
def awgn(x, snr_db):
    p = np.mean(np.abs(x)**2)  # moc sygnału
    snr = 10**(snr_db/10)      # SNR liniowe
    npow = p/snr               # moc szumu
    noise = np.sqrt(npow/2)*(np.random.randn(*x.shape)+1j*np.random.randn(*x.shape))
    return x + noise

rx_serial = awgn(rx_serial, 30)  # dodanie szumu 30 dB

# =====================
# Receiver (CP removal + FFT)
# =====================
rx_mat = rx_serial.reshape(num_ofdm, N+CP)
rx_no_cp = rx_mat[:, CP:]              # usunięcie CP
Y_rx = np.fft.fft(rx_no_cp, axis=1) / np.sqrt(N)

# =====================
# Estymacja kanału (LS z pilotów)
# =====================
Y_pilots = Y_rx[:, pilot_carriers]     # sygnał na pilotach
H_pilots = Y_pilots / (1 + 1j)         # znane piloty

H_est = np.zeros_like(Y_rx, dtype=complex)

# interpolacja kanału
for n in range(num_ofdm):
    H_real = np.interp(active_subcarriers, pilot_carriers, H_pilots[n].real)
    H_imag = np.interp(active_subcarriers, pilot_carriers, H_pilots[n].imag)
    H_est[n, active_subcarriers] = H_real + 1j * H_imag

# =====================
# Equalizacja
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
    noise_var = 10**(-30/10)
    Hloc = H_est[:, active_subcarriers]
    Yloc = Y_rx[:, active_subcarriers]

    # MMSE equalization
    Y_eq[:, active_subcarriers] = (np.conj(Hloc) / (np.abs(Hloc)**2 + noise_var)) * Yloc

else:
    raise ValueError("EQ_MODE must be 0, 1, or 2")

# =====================
# Konstelacje
# =====================
if can_save or can_show:
    plt.figure(figsize=(10, 6))

    max_sub = min(30, len(data_carriers))

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
# Demodulacja
# =====================
rx_data = Y_eq[:, data_carriers].reshape(-1)  # zbiór symboli danych
rx_bits = qam64_demod(rx_data)                 # demodulacja QAM
rx_bits = rx_bits[:num_bits_orig]              # obcięcie paddingu


# =====================
# SER (Symbol Error Rate)
# =====================

# symbole nadane
tx_symbols = symbols[:len(rx_data)]

# twarda decyzja detektora
rx_symbols_decided = qam64_mod(qam64_demod(rx_data))

# liczba błędnych symboli
symbol_errors = np.sum(tx_symbols != rx_symbols_decided)

# SER
ser = symbol_errors / len(tx_symbols)

# =====================
# BER + SER + odtworzenie tekstu
# =====================
ber = np.mean(bits_tx[:num_bits_orig] != rx_bits)

print("EQ_MODE:", EQ_MODE)
print("SER:", ser)
print("BER:", ber)

rx_bytes = np.packbits(rx_bits)  # bity → bajty
print("Odebrano:")
print(rx_bytes.tobytes().decode('utf-8', errors='ignore'))
print("liczba podnośnych:", N)
