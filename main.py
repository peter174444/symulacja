from ofdm3 import qam64_mod, qam64_demod, get_text, awgn_real, bb_to_rf, rf_to_bb, antenna_nonlinearity
from dmrs import make_global_grid, visualize_grid
import numpy as np
from scipy.signal import resample
from matplotlib import pyplot as plt
import random                           
import string                           

# flagi sterujące zapisem i wyświetlaniem wykresów
can_save = False
can_show = False
eq_type = "mmse"   

# =====================
# Parametry OFDM / BB
# =====================
num_qam_syms  = 60
data_per_ofdm = 60
delta_f       = 15e3                 # 15 kHz
fs_bb         = num_qam_syms * delta_f   # 900 kHz (baseband sampling)
CP            = 5
mod_bits      = 6                    # 64-QAM

# =====================
# Parametry RF
# =====================
os_factor = 4                        # oversampling RF
fs_rf    = fs_bb * os_factor         # 3.6 MHz
fc       = fs_rf / 8                 # 450 kHz
bw_bb    = fs_bb / 2                 # 450 kHz
snr_db = 40

# =====================
# Wiadomość → bity
# =====================
text = get_text()
# text = "".join(random.choices(string.printable, k=1000))
bits_tx = np.unpackbits(np.frombuffer(text.encode('utf-8'), dtype=np.uint8))
num_bits_orig = len(bits_tx)

# =====================
# GRID (5 RB, DMRS w czasie jak w 5G NR)
# =====================

num_rb = 5
num_data_symbols = num_rb * 12 * 12      # 5 RB × 12 subcarriers × 12 data symbols
num_data_bits = num_data_symbols * mod_bits

# przycinamy/padujemy wiadomość do pojemności grida
if len(bits_tx) >= num_data_bits:
    bits_tx_use = bits_tx[:num_data_bits]
else:
    bits_tx_use = np.hstack([bits_tx, np.zeros(num_data_bits - len(bits_tx), dtype=np.int8)])

symbols = qam64_mod(bits_tx_use)        # dokładnie 720 symboli QAM
data_symbols = symbols                  # wszystkie są danymi

# 1) pojedynczy RB (pilotowe symbole 2 i 11)
# rb = make_rb(data_symbols[:12*12], pilot_symbols=[2, 11])

# 2) globalna siatka 5 RB
grid = make_global_grid(data_symbols, num_rb=5)

# 3) wizualizacja
# visualize_rb(rb)
visualize_grid(grid)

# =====================
# OFDM modulacja (baseband)
# =====================
ofdm_time = np.fft.ifft(grid, axis=1) * np.sqrt(num_qam_syms)

# dodanie prefixu cyklicznego
cp = ofdm_time[:, -CP:]
tx_signal = np.hstack([cp, ofdm_time])
tx_serial = tx_signal.flatten()

# =====================
# NIELINIOWOŚĆ ANTENY (PA)
# =====================
tx_serial = antenna_nonlinearity(tx_serial, a3=0.01)

# =====================
# Kanał wielodrogowy (3-tap)
# =====================
h = np.array([0.9+0j, 0.4-0.3j, 0.2+0.1j, 0, 0])
h = h / np.linalg.norm(h)

tx_serial = np.convolve(tx_serial, h)[:len(tx_serial)]

# =====================
# Oversampling BB → RF
# =====================
L = len(tx_serial) * os_factor
tx_bb_os = resample(tx_serial, L)

# modulacja I/Q → RF
rf = bb_to_rf(tx_bb_os, fs_rf, fc)

# opcjonalny szum RF:
rf = awgn_real(rf, snr_db)

# =====================
# Widmo (FFT)
# =====================
def spectrum(x, fs):
    Nfft = 2048
    S = np.fft.fftshift(np.fft.fft(x, Nfft))
    f = np.fft.fftshift(np.fft.fftfreq(Nfft, d=1/fs))
    return f, np.abs(S)

# Baseband i RF
f_bb, S_bb = spectrum(tx_serial, fs_bb)
f_rf, S_rf = spectrum(rf, fs_rf)

if can_save or can_show:
    plt.figure(figsize=(10,5))
    plt.plot(f_bb, S_bb, label="Baseband OFDM")
    plt.plot(f_rf, S_rf, label="RF signal")
    plt.legend()
    plt.grid()
    plt.title("OFDM Spectrum (BB & RF)")
    if can_save:
        plt.savefig("plots/OFDM_Spectrum.png")
    if can_show:
        plt.show()
    plt.close()


# =====================
# PSD (Baseband)
# =====================
def psd(x, fs):
    Nfft = 4096
    X = np.fft.fftshift(np.fft.fft(x, Nfft))
    Pxx = (np.abs(X)**2) / (Nfft * fs)
    f = np.fft.fftshift(np.fft.fftfreq(Nfft, d=1/fs))
    return f, Pxx

f_psd_bb, Pxx_bb = psd(tx_serial, fs_bb)
Pxx_bb_dB = 10 * np.log10(Pxx_bb + 1e-20)

if can_save or can_show:
    plt.figure(figsize=(10,5))
    plt.plot(f_psd_bb, Pxx_bb_dB)
    plt.xlabel("Częstotliwość [Hz]")
    plt.ylabel("PSD [dB/Hz]")
    plt.grid()
    plt.title("PSD (Baseband)")
    if can_save:
        plt.savefig("plots/PSD_BB.png")
    if can_show:
        plt.show()
    plt.close()


# =====================
# PSD (RF)
# =====================
f_psd_rf, Pxx_rf = psd(rf, fs_rf)
Pxx_rf_dB = 10 * np.log10(Pxx_rf + 1e-20)

if can_save or can_show:
    plt.figure(figsize=(10,5))
    plt.plot(f_psd_rf, Pxx_rf_dB)
    plt.xlabel("Częstotliwość [Hz]")
    plt.ylabel("PSD [dB/Hz]")
    plt.grid()
    plt.title("PSD (RF)")
    if can_save:
        plt.savefig("plots/PSD_RF.png")
    if can_show:
        plt.show()
    plt.close()


# =====================
# Spektrogram Baseband
# =====================
if can_save or can_show:
    plt.figure(figsize=(10,5))
    plt.specgram(tx_serial, NFFT=256, Fs=fs_bb, noverlap=128)
    plt.title("Spectrogram OFDM (Baseband)")
    plt.xlabel("Czas [s]")
    plt.ylabel("Częstotliwość [Hz]")
    plt.colorbar()
    if can_save:
        plt.savefig("plots/Spectrogram_BB.png")
    if can_show:
        plt.show()
    plt.close()


# =====================
# Spektrogram RF
# =====================
if can_save or can_show:
    plt.figure(figsize=(10,5))
    plt.specgram(rf, NFFT=256, Fs=fs_rf, noverlap=128)
    plt.title("Spectrogram OFDM (RF)")
    plt.xlabel("Czas [s]")
    plt.ylabel("Częstotliwość [Hz]")
    plt.colorbar()
    if can_save:
        plt.savefig("plots/Spectrogram_RF.png")
    if can_show:
        plt.show()
    plt.close()


# =====================
# RF → BB
# =====================
bb_os_rec = rf_to_bb(rf, fs_rf, fc, bw_bb)

# downsampling do fs_bb
rx_serial = resample(bb_os_rec, len(tx_serial))

# =====================
# Receiver: CP → FFT
# =====================
samples_per_ofdm = data_per_ofdm + CP
num_ofdm_rx = len(rx_serial) // samples_per_ofdm

rx_mat = rx_serial[:num_ofdm_rx * samples_per_ofdm].reshape(num_ofdm_rx, samples_per_ofdm)
rx_no_cp = rx_mat[:, CP:]
Y_rx = np.fft.fft(rx_no_cp, axis=1) / np.sqrt(num_qam_syms)

# ===========================
# Estymacja kanału z DMRS (symbol 2 i 11)
# ===========================
pilot_symbols = [2, 11]
pilot_val = 1.0 + 1j

# estymacja na podnośnych pilotowych (co 2)
H_dmrs_2  = Y_rx[2,  ::2] / pilot_val
H_dmrs_11 = Y_rx[11, ::2] / pilot_val

# ===========================
# Interpolacja 2D: czas + częstotliwość
# ===========================

num_symbols = Y_rx.shape[0]   # 14
num_subcarriers = Y_rx.shape[1]  # 60

H_est = np.zeros((num_symbols, num_subcarriers), dtype=complex)

# 1) Interpolacja po czasie (dla każdej podnośnej pilotowej)
for i, sc in enumerate(range(0, num_subcarriers, 2)):
    H_est[:, sc] = np.interp(
        np.arange(num_symbols),      # 0..13
        pilot_symbols,               # [2, 11]
        [H_dmrs_2[i], H_dmrs_11[i]]  # wartości na DMRS
    )

# 2) Interpolacja po częstotliwości (dla każdego symbolu)
pilot_sc = np.arange(0, num_subcarriers, 2)

for s in range(num_symbols):
    H_real = np.interp(
        np.arange(num_subcarriers),
        pilot_sc,
        H_est[s, pilot_sc].real
    )
    H_imag = np.interp(
        np.arange(num_subcarriers),
        pilot_sc,
        H_est[s, pilot_sc].imag
    )
    H_est[s, :] = H_real + 1j * H_imag

# ===========================
# Equalizacja ZF / MMSE
# ===========================

if eq_type == "zf":
    Y_eq = Y_rx * np.conj(H_est) / (np.abs(H_est)**2 + 1e-12)

elif eq_type == "mmse":
    snr_lin = 10**(snr_db / 10)
    sigma2 = 1 / snr_lin
    Y_eq = Y_rx * np.conj(H_est) / (np.abs(H_est)**2 + sigma2)

else:
    raise ValueError("eq_type must be 'zf' or 'mmse'")


# =====================
# Mask RE danych (bez pilotów)
# =====================
pilot_symbols = [2, 11]
mask = np.ones_like(grid, dtype=bool)
for s in pilot_symbols:
    mask[s, :] = False   # cały symbol pilotowy = brak danych

# RX: bierzemy tylko RE-dane
rx_data = Y_eq[mask].reshape(-1)

# =====================
# Mask RE danych (bez pilotów)
# =====================
pilot_symbols = [2, 11]
mask = np.ones_like(grid, dtype=bool)
for s in pilot_symbols:
    mask[s, :] = False   # cały symbol pilotowy = brak danych

# RX: bierzemy tylko RE-dane
rx_data = Y_eq[mask].reshape(-1)


# =====================
# Konstelacje (tylko RE danych, bez DMRS)
# =====================

pilot_symbols = [2, 11]
pilot_subcarriers = np.arange(0, num_qam_syms, 2)   # 0,2,4,...,58
data_subcarriers = np.setdiff1d(np.arange(num_qam_syms), pilot_subcarriers)

if can_save or can_show:
    plt.figure(figsize=(10, 6))

    # ile podnośnych pokazać
    max_sub = min(30, len(data_subcarriers))

    # symbole danych (bez DMRS)
    data_symbols = [s for s in range(Y_eq.shape[0]) if s not in pilot_symbols]

    for idx, sc in enumerate(data_subcarriers[:max_sub]):
        pts = Y_eq[data_symbols, sc]   # punkty konstelacji dla tej podnośnej

        plt.scatter(
            pts.real,
            pts.imag,
            s=8,
            alpha=0.6,
            label=f"podnośna {sc}"
        )

    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.grid()
    plt.title("Konstelacje różnych podnośnych (tylko dane)")
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
rx_bits = qam64_demod(rx_data)
rx_bits = rx_bits[:num_bits_orig]

# =====================
# SER + BER
# =====================
tx_symbols = symbols[:len(rx_data)]
rx_symbols_decided = qam64_mod(qam64_demod(rx_data))
symbol_errors = np.sum(tx_symbols != rx_symbols_decided)
ser = symbol_errors / len(tx_symbols)
ber = np.mean(bits_tx_use != rx_bits)

print("SER:", ser)
print("BER:", ber)

rx_bytes = np.packbits(rx_bits)
print("Odebrano:")
print(rx_bytes.tobytes().decode('utf-8', errors='ignore'))
