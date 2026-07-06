import numpy as np
import h5py

# =====================================================
# Parameters
# =====================================================

NUM_SAMPLES = 1024
SAMPLES_PER_SYMBOL = 8
NUM_SYMBOLS = NUM_SAMPLES // SAMPLES_PER_SYMBOL

NUM_EXAMPLES = 500  # Per modulation per SNR

SNR_VALUES = np.arange(-20, 21, 2)

MODULATIONS = [
    "OOK",
    "ASK",
    "BPSK",
    "QPSK",
    "8PSK",
    "16PSK",
    "BFSK",
    "4FSK",
    "8FSK"
]

LABEL_MAP = {
    "OOK": 0,
    "ASK": 1,
    "BPSK": 2,
    "QPSK": 3,
    "8PSK": 4,
    "16PSK": 5,
    "BFSK": 6,
    "4FSK": 7,
    "8FSK": 8
}

# =====================================================
# AWGN
# =====================================================

def add_awgn(signal, snr_db):

    signal_power = np.mean(np.abs(signal) ** 2)

    snr_linear = 10 ** (snr_db / 10)

    noise_power = signal_power / snr_linear

    noise = np.sqrt(noise_power / 2) * (
        np.random.randn(*signal.shape)
        + 1j * np.random.randn(*signal.shape)
    )

    return signal + noise


# =====================================================
# Modulation Functions
# =====================================================

def ook():

    bits = np.random.randint(0, 2, NUM_SYMBOLS)

    symbols = bits.astype(float)

    return np.repeat(symbols, SAMPLES_PER_SYMBOL)


def ask():

    bits = np.random.randint(0, 2, NUM_SYMBOLS)

    amplitudes = 2 * bits + 1

    return np.repeat(amplitudes, SAMPLES_PER_SYMBOL)


def bpsk():

    bits = np.random.randint(0, 2, NUM_SYMBOLS)

    symbols = 2 * bits - 1

    return np.repeat(symbols, SAMPLES_PER_SYMBOL)


def qpsk():

    symbols = np.random.randint(0, 4, NUM_SYMBOLS)

    phase = np.pi / 4 + symbols * np.pi / 2

    signal = np.exp(1j * phase)

    return np.repeat(signal, SAMPLES_PER_SYMBOL)


def psk8():

    symbols = np.random.randint(0, 8, NUM_SYMBOLS)

    phase = symbols * 2 * np.pi / 8

    signal = np.exp(1j * phase)

    return np.repeat(signal, SAMPLES_PER_SYMBOL)


def psk16():

    symbols = np.random.randint(0, 16, NUM_SYMBOLS)

    phase = symbols * 2 * np.pi / 16

    signal = np.exp(1j * phase)

    return np.repeat(signal, SAMPLES_PER_SYMBOL)


def bfsk():

    bits = np.random.randint(0, 2, NUM_SYMBOLS)

    t = np.arange(SAMPLES_PER_SYMBOL)

    signal = []

    for bit in bits:

        f = 1 if bit == 0 else 3

        tone = np.exp(1j * 2 * np.pi * f * t / SAMPLES_PER_SYMBOL)

        signal.extend(tone)

    return np.array(signal)


def fsk4():

    symbols = np.random.randint(0, 4, NUM_SYMBOLS)

    frequencies = [1, 2, 3, 4]

    t = np.arange(SAMPLES_PER_SYMBOL)

    signal = []

    for s in symbols:

        tone = np.exp(
            1j * 2 * np.pi * frequencies[s] * t / SAMPLES_PER_SYMBOL
        )

        signal.extend(tone)

    return np.array(signal)


def fsk8():

    symbols = np.random.randint(0, 8, NUM_SYMBOLS)

    frequencies = np.arange(1, 9)

    t = np.arange(SAMPLES_PER_SYMBOL)

    signal = []

    for s in symbols:

        tone = np.exp(
            1j * 2 * np.pi * frequencies[s] * t / SAMPLES_PER_SYMBOL
        )

        signal.extend(tone)

    return np.array(signal)


# =====================================================
# Function Dictionary
# =====================================================

GENERATORS = {
    "OOK": ook,
    "ASK": ask,
    "BPSK": bpsk,
    "QPSK": qpsk,
    "8PSK": psk8,
    "16PSK": psk16,
    "BFSK": bfsk,
    "4FSK": fsk4,
    "8FSK": fsk8
}

# =====================================================
# Dataset Generation
# =====================================================

X = []
Y = []
SNR = []

print("Generating Dataset...\n")

for modulation in MODULATIONS:

    print(f"Generating {modulation}...")

    generator = GENERATORS[modulation]

    label = LABEL_MAP[modulation]

    for snr in SNR_VALUES:

        for _ in range(NUM_EXAMPLES):

            signal = generator()

            signal = add_awgn(signal, snr)

            iq = np.vstack((signal.real, signal.imag))

            X.append(iq.astype(np.float32))

            Y.append(label)

            SNR.append(snr)

# =====================================================
# Convert to Arrays
# =====================================================

X = np.array(X, dtype=np.float32)

Y = np.array(Y, dtype=np.int64)

SNR = np.array(SNR, dtype=np.int32)

print("\nDataset Created Successfully")

print("X Shape :", X.shape)
print("Y Shape :", Y.shape)
print("SNR Shape :", SNR.shape)

# Expected:
# X   -> (94500, 2, 1024)
# Y   -> (94500,)
# SNR -> (94500,)

# =====================================================
# Save HDF5 Dataset
# =====================================================

with h5py.File("modulation_dataset.h5", "w") as f:

    f.create_dataset("X", data=X)

    f.create_dataset("Y", data=Y)

    f.create_dataset("SNR", data=SNR)

print("\nDataset saved as modulation_dataset.h5")
print("Done.")
