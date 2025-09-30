"""
     ###  #######  ##   ##   #####              ####
     ###  ####     # # ###  ### ###           ##  ###
  ######  ######   #######  ### ###              ###
 ### ###  ####     #######  ### ###             ###
 ### ###  ####     ### ###  #######            ###
  ### ##  #######  ### ###   #####            #######

This demo shows fast split step!
"""
from functions import *
from scipy.constants import c, pi
import sibase
sibase.set_special_unit('dB', lambda x, _: x, lambda x: x)
v = lambda x: float(sibase.Value(x))

D = v('0 ps/nm/km')
S = v('0.057 ps/nm^2/km')
att = v('0.33 dB/km')
distance = v('80 km')
gamma = v('2 W/km')
lambda_ref = v('1310nm')

def my_function(key):
    fb = 48e9
    ns = 2   # samples per symbol, nyquist
    fs_ch = fb * ns
    roll_off = 0.001  # rrc roll off
    n_ch = 3

    fs, tx, tx_seq, signal, downsampler = generate_signal(96e9, fb, fs_ch, n_ch, 2 ** 13, '16QAM', key, roll_off, ch_power_dBm=10, return_downsampler=True)
    print(f'FS={fs*1e-9:.0f}GHz')
    beta2 = -D * lambda_ref ** 2 / (2 * pi * c)
    beta3 = S * lambda_ref ** 4 / (4 * pi**2 * c ** 2) - beta2 * lambda_ref / (pi * c)
    alpha = att / (np.log(10) / 10)
    signal = add_awgn(signal, key, 15)
    signal = ssfm_local_error(signal, fs, distance, 1e-8, beta2, beta3, alpha, gamma)

    signal = downsampler(signal)
    # signal = rrc(signal)
    signal = cdc_rrc(signal, fs_ch, fb, distance, beta2, beta3, roll_off)

    # We will only do central channel
    rx_symbols, taps = equaliser(tx[n_ch//2], signal, 15, ns)
    return fs, signal, tx[n_ch//2], rx_symbols, taps
    return tx, rx_symbols


def main():
    key = jax.random.PRNGKey(0)

    with Timer("Running my_function #1"):
        my_function_jit = jax.jit(my_function)
        fs, signal, tx, rx, taps = my_function_jit(key)
        snr = compute_snr(tx, rx)
        print(f"SNR={snr[0]:.3f}/{snr[1]:.3f} dB")
    show_spectrum(signal, fs)
    show_constellations(rx / jnp.sqrt(jnp.mean(jnp.abs(rx) ** 2)), tx)
    pass


if __name__ == '__main__':
    print(f"Device: {jax.devices()[0].device_kind}")
    main()
