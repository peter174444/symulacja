import numpy as np

def get_text():
    return "Lecz jeśli zechcesz mnie posłuchać, zaraz zaspokoję twoją ciekawość. Historia nie jest zbyt długa. Ty może osobiście znałeś Wanniusza, króla Swebów, który, wypędzony z kraju, długi czas przesiadywał tu w Rzymie, a nawet wsławił się szczęśliwą grą w kości i dobrym powożeniem. Cezar Drusus wprowadził go znów na tron. Wanniusz, który był w istocie rzeczy tęgim człowiekiem, rządził z początku dobrze i prowadził szczęśliwe wojny, później jednak począł nadto łupić ze skóry nie tylko sąsiadów, ale i własnych Swebów. Wówczas Wangio i Sido, dwaj jego siostrzeńcy, a synowie Wibiliusza, króla Hermandurów, postanowili zmusić go, by znów pojechał do Rzymu… próbować szczęścia w kości."

def qam64_mod(bits):
    bits = bits.astype(np.int8)
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

def awgn(x, snr_db):
    p = np.mean(np.abs(x)**2)
    snr = 10**(snr_db/10)
    npow = p/snr
    noise = np.sqrt(npow/2)*(np.random.randn(*x.shape)+1j*np.random.randn(*x.shape))
    return x + noise

def awgn_real(x, snr_db):
    p = np.mean(x**2)
    snr = 10**(snr_db/10)
    npow = p/snr
    noise = np.sqrt(npow) * np.random.randn(*x.shape)
    return x + noise


# =====================
# Funkcje RF
# =====================
def bb_to_rf(bb, fs, fc):
    t = np.arange(len(bb)) / fs
    I = np.real(bb)
    Q = np.imag(bb)
    return I * np.cos(2*np.pi*fc*t) - Q * np.sin(2*np.pi*fc*t)

def lowpass_fft(x, fs, bw):
    X = np.fft.fft(x)
    f = np.fft.fftfreq(len(x), 1/fs)
    H = np.abs(f) <= bw
    return np.fft.ifft(X * H)

def rf_to_bb(rf, fs, fc, bw):
    t = np.arange(len(rf)) / fs
    yI = rf * np.cos(2*np.pi*fc*t)
    yQ = -rf * np.sin(2*np.pi*fc*t)
    I = lowpass_fft(yI, fs, bw)
    Q = lowpass_fft(yQ, fs, bw)
    return 2*(I + 1j*Q)


def antenna_nonlinearity(x, a1=1.0, a3=0.01):
    # model nieliniowy (AM/AM distortion)
    return x + a3 * x * np.abs(x)**2 + a1 * x




