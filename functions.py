import logging
from fractions import Fraction
import re
import jax
jax.config.update("jax_enable_x64", True)

from jax import numpy as jnp
import numpy as np
from scipy.constants import c, h, pi

from jax.debug import callback as jax_callback
from commpy.filters import rrcosfilter
from commpy import PSKModem, QAMModem
import time
from matplotlib import pyplot as plt

log = logging.getLogger(__name__)
modulation_re = re.compile(r'(\d+)-?([A-Z]{3,4})')


class Timer:
    def __init__(self, name='Timer'):
        self.name = name

    def __enter__(self):
        self.start = time.perf_counter()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop = time.perf_counter()
        print(f"{self.name} took {self.stop - self.start:.3f}sec")


def graycode(n):
    j = np.arange(n, dtype=np.uint8)
    return j ^ (j >> 1)

def get_modulation(name: str):
    """ Quick and dirty solution, name being 8PSK/16QAM/etc. I'd recommend using something else, like qampy or opticommpy """
    match = modulation_re.match(name.upper())
    if match is None:
        raise ValueError('Unknown modulation')
    size = int(match.groups()[0])
    name = match.groups()[1]
    n = int(np.log2(size))
    assert 2**n == size
    if name == 'QAM':
        bits = np.unpackbits(graycode(size)[:, None], axis=1)[:, -n:]
        c = QAMModem(size).modulate(bits.flatten())
        return c / np.sqrt(np.mean(np.abs(c) ** 2))
    if name == 'PSK':
        bits = np.unpackbits(graycode(size)[:, None], axis=1)[:, -n:]
        c = PSKModem(size).modulate(bits.flatten())
        return c / np.sqrt(jnp.mean(np.abs(c) ** 2))
    raise ValueError('Unknown modulation')

def ssfm_local_error(sig, fs, distance, delta_g, beta2, beta3, alpha, gamma, callback=None):
    ww = 2. * pi * jnp.fft.fftfreq(sig.shape[-1], d=1 / fs)
    d = 1j * beta2 / 2 * ww ** 2
    d += 1j * beta3 / 6 * ww ** 3
    dz0 = distance / 100
    dz_factor = 2 ** (1 / 3)

    def _step(Ef, D, nl_factor, dza):
        Et = jnp.fft.ifft(Ef * D)
        power = jnp.sum(jnp.abs(Et) ** 2, axis=0)
        Et *= jnp.exp(8./9. * nl_factor * power - dza)
        return jnp.fft.fft(Et) * D

    def _loop(args):
        Ef, z, dz = args

        dz = jnp.minimum(dz, distance - z)  # Do not overshoot distance

        # Coarse step
        nl_factor_c = 1j * dz * gamma
        D_c = jnp.exp(dz / 2 * d)
        dza_c = dz * alpha / 2
        u_c = _step(Ef, D_c, nl_factor_c, dza_c)

        # Fine step
        D_f = jnp.exp(dz / 4 * d)
        u_f = _step(Ef, D_f, nl_factor_c / 2, dza_c / 2)
        u_f = _step(u_f, D_f, nl_factor_c / 2, dza_c / 2)

        # Relative Local Error
        u_c = jnp.fft.ifft(u_c)
        u_f = jnp.fft.ifft(u_f)
        local_error = jnp.sum(jnp.abs(u_f - u_c) ** 2) / jnp.sum(jnp.abs(u_f) ** 2)
        Ef1 = jnp.fft.fft(4 / 3 * u_f - 1 / 3 * u_c)  # next Ef

        # Next step size
        dz1 = jax.lax.select(local_error < (0.5 * delta_g), dz * dz_factor, dz)
        dz1 = jax.lax.select(local_error > delta_g, dz / dz_factor, dz1)
        dz1 = jax.lax.select(local_error > (2 * delta_g), dz / 2, dz1)

        # Discard check
        Ef = jax.lax.select(local_error > 2 * delta_g, Ef, Ef1)
        dz2 = jax.lax.select(local_error > 2 * delta_g, 0., dz)
        z += dz2
        if callable(callback):
            jax_callback(lambda x: callback(x), z)
        return Ef, z, dz1

    sig, dist, _ = jax.lax.while_loop(
        lambda arg: arg[1] < distance,
        _loop,
        (jnp.fft.fft(sig), 0.0, dz0)
    )
    return jnp.fft.ifft(sig)


def rrc_fourier_func(n, fs, fb, roll_off):
    """ Create RRC filter function performed using Fourier method """
    hh = jnp.fft.fft(rrcosfilter(n, alpha=roll_off, Ts=1 / fb, Fs=fs)[1])
    return lambda signal: jnp.fft.ifft(jnp.fft.fft(signal) * hh)

def add_awgn(signal, key, snr):
    sigma = 10 ** (-snr / 20)
    noise = jnp.sqrt(jnp.mean(jnp.abs(signal) ** 2)) * jax.random.normal(key, signal.shape, dtype=signal.dtype) * sigma
    return signal + noise

def freq_shift_fourier(signal: jax.Array, offset: float, fs: float, axis=-1) -> jax.Array:
    """ Shift signal in frequency domain by offset using Fourier method """
    e_f = jnp.fft.fft(signal)
    e_f = jnp.roll(e_f, jnp.array(offset / (fs / signal.shape[-1])).astype(int), axis=axis)
    return jnp.fft.ifft(e_f)

def power_meter(signal):
    """ Return signal power in dBm """
    power = (jnp.abs(signal) ** 2).mean() * signal.shape[0]
    return 10 * jnp.log10(power) + 30

def set_power(signal, power):
    """ Set signal power in dBm """
    g = 10 ** ((power - power_meter(signal)) / 20)
    g = jnp.nan_to_num(g, copy=False, nan=0.0)
    return signal * g

def make_resampler(nx: int, up: int, down: int):
    """ This is pretty much a clone from scipy resample that works on JAX """
    samples = nx * up // down
    if nx == samples:
        def _resampler(x):
            return x

        return _resampler
    n = np.minimum(nx, samples)
    nyq = n // 2 + 1

    def _resampler(signal):
        x = jnp.fft.fft(signal)
        y = jnp.zeros((2, samples), dtype='complex')
        y = y.at[:, :nyq].set(x[:, :nyq])
        y = y.at[:, nyq - n:].set(x[:, nyq - n:])

        if n % 2 == 0:
            if samples < nx:  # downsampling
                sl = slice(-n // 2, -n // 2 + 1)
                y = y.at[:, sl].set(y[:, sl] + x[:, sl])

            if samples > nx:  # upsampling
                sl = slice(n // 2, n // 2 + 1)
                temp = y[:, sl] * 0.5
                y = y.at[:, sl].set(temp)
                y = y.at[:, samples - n // 2:samples - n // 2 + 1].set(temp)

        yt = jnp.fft.ifft(y)
        yt *= (samples / nx)
        return yt

    return _resampler


def generate_signal(ch_spacing, symbol_rate, sample_rate, num_channels, seq_length, modulation, prng_key,
                    roll_off, ch_power_dBm, return_downsampler=False):
    """ This function generates a WDM signal """
    ns = sample_rate / symbol_rate # Samples per symbol
    npol = 2  # always 2 polarisations
    # Final signal sample rate
    # if num_channels == 1:
    #     fs = sample_rate
    # else:
    fs = 2 ** np.ceil(np.log2(ch_spacing / symbol_rate * num_channels * ns)) * symbol_rate
    frac = Fraction(float(sample_rate / fs)).limit_denominator(int(1e5))
    ch_seq_size = int(ns * seq_length)
    sch_seq_size = ch_seq_size * frac.denominator // frac.numerator

    f_offsets = (jnp.arange(num_channels) - (num_channels - 1) / 2) * ch_spacing

    if not isinstance(modulation, str):
        a = modulation
        tx_seq = jax.random.randint(prng_key, (num_channels, npol, seq_length), minval=0, maxval=a.size)
        tx = jnp.take(a, tx_seq)
    elif modulation.lower() in {"gaus", "gauss", "gaussian"}:
        print(f"Generating gaussian symbols")
        tx_seq = None
        tx = jax.random.normal(prng_key, (num_channels, npol, seq_length), dtype=jnp.complex64)
    else:
        a = get_modulation(modulation)
        print(f"Generating {modulation.upper()} [{len(a)} points] symbols")
        tx_seq = jax.random.randint(prng_key, (num_channels, npol, seq_length), minval=0, maxval=a.size)
        tx = jnp.take(a, tx_seq)

    kron_vec = jnp.array([1] + [0] * int(ns - 1))
    superch = jnp.zeros((npol, sch_seq_size), dtype='complex')

    rrc = rrc_fourier_func(ch_seq_size, sample_rate, symbol_rate, roll_off)
    upsampler = make_resampler(ch_seq_size, frac.denominator, frac.numerator)

    def _make_ch(ch, signal, callback=None):
        ch_sig = jnp.kron(tx[ch], kron_vec)
        ch_sig = rrc(ch_sig)
        ch_sig = upsampler(ch_sig)
        ch_sig = freq_shift_fourier(ch_sig, f_offsets[ch], fs, axis=-1)
        if isinstance(ch_power_dBm, (float, int)):
            ch_sig = set_power(ch_sig, ch_power_dBm)
        else:
            ch_sig = set_power(ch_sig, ch_power_dBm[ch])
        signal += ch_sig
        if callable(callback):
            jax_callback(lambda x: callback(x), ch)
        return signal

    superch = jax.lax.fori_loop(0, num_channels, _make_ch, superch)
    if return_downsampler:
        downsampler = make_resampler(sch_seq_size, frac.numerator, frac.denominator)
        return fs, tx, tx_seq, superch, downsampler
    return fs, tx, tx_seq, superch

def cdc_rrc(signal, fs, fb, distance, beta2, beta3, roll_off):
    """ Chromatic dispersion combination + RRC filter in one frequency step """
    ww = 2. * pi * jnp.fft.fftfreq(signal.shape[-1], d=1 / fs)
    d = 1j * beta2 / 2 * ww ** 2
    d += 1j * beta3 / 6 * ww ** 3

    _, hh = rrcosfilter(signal.shape[-1], alpha=roll_off, Ts=1 / fb, Fs=fs)
    return jnp.fft.ifft(jnp.fft.fft(signal) * jnp.exp(distance * -d) * jnp.fft.fft(hh))

def equaliser(tx_symbols, rx_signal, taps, ns):
    # If sequence size is more than 2**13, don't to lstsq on the whole sequence,
    # take only every nth tx_symbol (kinda like doing equaliser on pilots)
    n = max(rx_signal.shape[1] // 2 ** 13, 1)
    toeplitz = jax.lax.conv_general_dilated_patches(
        rx_signal[None, :, :], (taps,), (ns,), 'SAME')[0]
    h, _, _, _ = jnp.linalg.lstsq(toeplitz[:, ::n].T, tx_symbols[:, ::n].T)
    return h.T @ toeplitz, h

def ase_power(gain, nf, fs, wavelengths):
    n = (10 ** (nf / 10) * gain - 1) / (2 * (gain - 1))
    return 2 * n * (gain - 1) * h * (c / wavelengths) * fs

def amplifier(signal, gain, nf, fs, wavelengths, key):
    signal *= jnp.sqrt(gain)
    sigma = jnp.sqrt(0.5 / signal.shape[0] * ase_power(gain, nf, fs, wavelengths))
    noise = jax.random.normal(key, signal.shape, dtype=signal.dtype) * jnp.sqrt(2) * sigma
    signal += noise
    return

def compute_snr(tx, rx, padding=1000):
    tx, rx = tx[:, padding:-padding], rx[:, padding:-padding]
    snr = jnp.sum(jnp.abs(tx) ** 2, axis=-1) / jnp.sum(jnp.abs(rx - tx) ** 2, axis=-1)
    return 10 * jnp.log10(snr)

def show_spectrum(signal, fs, samples=8000):
    bins = signal.shape[-1]
    ff = np.fft.fftshift(np.fft.fftfreq(bins, d=1 / fs))
    ef = np.fft.fftn(signal, (signal.shape[0], bins))
    ef = np.fft.fftshift(np.sum(np.abs(np.sqrt(1./bins) * ef) ** 2, axis=-2))
    ef[ef == 0] = np.min(ef[np.nonzero(ef)]) * 1e-2  # Replace 0 with lowest value -20dB

    units = 'GHz'
    unit_scale = 1e9
    if (ff / unit_scale).max() >= 1e3:
        units = 'THz'
        unit_scale = 1e12
    splices = int(np.ceil(bins / samples))
    fig, ax = plt.subplots()
    a = 10 * np.log10(ef[::splices])
    ax.plot(ff[::splices] / unit_scale, a)
    ax.set_xlabel(f'Frequency [{units}]')
    ax.set_ylabel('Optical power [dBm]')
    ax.grid(True)
    plt.show()

def show_constellations(rx, tx=None, bins=128):
    if tx is not None:
        constellation = np.unique(tx)
    titles = ['X-pol', 'Y-pol']
    npol = 2
    lim = 1.7
    fig, ax = plt.subplots(1, npol)

    for i in range(npol):
        hist, _, _ = np.histogram2d(rx[i].real, rx[i].imag, bins, range=[[-lim, lim], [-lim, lim]])
        ax[i].imshow(hist.T, cmap='viridis', aspect='equal', origin='lower',
                     interpolation='bicubic', extent=[-lim, lim, -lim, lim])
        ax[i].set_xticks(np.arange(-1, 2, 1))
        ax[i].set_yticks(np.arange(-1, 2, 1))

        ax[i].set_title(titles[i])
        ax[i].grid(color='m', linestyle='-', linewidth=.4)
        if tx is not None:
            ax[i].plot(np.real(constellation), np.imag(constellation), '.w')

    fig.tight_layout()
    plt.show()