"""
'########::'########:'##::::'##::'#######::::::'#######::
 ##.... ##: ##.....:: ###::'###:'##.... ##::::'##.... ##:
 ##:::: ##: ##::::::: ####'####: ##:::: ##::::..::::: ##:
 ##:::: ##: ######::: ## ### ##: ##:::: ##:::::'#######::
 ##:::: ##: ##...:::: ##. #: ##: ##:::: ##:::::...... ##:
 ##:::: ##: ##::::::: ##:.:: ##: ##:::: ##::::'##:::: ##:
 ########:: ########: ##:::: ##:. #######:::::. #######::
........:::........::..:::::..:::.......:::::::.......:::

GN Models and Optimisers!
"""

from functions import *
from demo3 import isrs_gn_jax, isrs_gn

from scipy.optimize import minimize
import optax       # https://optax.readthedocs.io/en/latest/api/optimizers.html
import jaxopt      # https://jaxopt.github.io/stable/index.html
import optimistix  # https://docs.kidger.site/optimistix/
import diffrax     # https://docs.kidger.site/diffrax/

n = 1  # number of spans
Bch = 40.004e9  # WDM channel bandwidth
channels = 251  # number of channels
spacing = 40.005e9  # WDM channel spacing

P = {
    'fi': np.repeat(np.reshape(
        (np.arange(channels) - (channels - 1) / 2) * spacing
        , [-1, 1]), n, axis=1),  # center frequencies of WDM channels (relative to reference frequency)
    'n': n,  # number of spans
    'Bch': np.tile(40.004e9, [channels, n]),  # channel bandwith
    'RefLambda': 1550e-9,  # reference wavelength
    'D': 17 * 1e-12 / 1e-9 / 1e3 * np.ones(n),  # dispersion coefficient      (same) for each span
    'S': 0.067 * 1e-12 / 1e-9 / 1e3 / 1e-9 * np.ones(n),  # dispersion slope            (same) for each span
    'Att': 0.2 / 4.343 / 1e3 * np.ones([channels, n]),
    # attenuation coefficient     (same) for each channel and span
    'Cr': 0.028 / 1e3 / 1e12 * np.ones([channels, n]),
    # Raman gain spectrum slope   (same) for each channel and span
    'gamma': 1.2 / 1e3 * np.ones(n),  # nonlinearity coefficient    (same) for each span
    'Length': 100 * 1e3 * np.ones(n),  # fiber length                (same) for each span
    'coherent': 1  # NLI is added coherently across multiple spansP_tot
}

P['Att_bar'] = P['Att']

def compute_snr(x):
    Pch = 10 ** (x / 10) * 0.001 * np.ones([channels, n])

    nli, eta = jax.jit(isrs_gn_jax.ISRSGNmodel)(
        Att=P['Att'],
        Att_bar=P['Att_bar'],
        Cr=P['Cr'],
        Pch=Pch,
        fi=P['fi'],
        Bch=P['Bch'],
        Length=P['Length'],
        D=P['D'],
        S=P['S'],
        gamma=P['gamma'],
        RefLambda=P['RefLambda']
    )
    ase = ase_power(10 ** (0.2e-3 * P['Length'] / 10), 4.5, Bch, c / (c / P['RefLambda'] + P['fi'][:, 0]))
    return Pch, nli, ase

def main():

    Pch, nli, ase = compute_snr(0.0)  # 0 dBm/ch
    p0 = plt.plot(P['fi'][:, 0] / 1e12, 10 * jnp.log10(Pch[:, 0] / nli), label='NLI 0dBm/ch')
    p1 = plt.plot(P['fi'][:, 0] / 1e12, 10 * jnp.log10(Pch[:, 0] / ase), label='ASE')
    p2 = plt.plot(P['fi'][:, 0] / 1e12, 10 * jnp.log10(Pch[:, 0] / (nli+ase)), label='Total 0dBm/ch')

    def _loss(x):
        Pch, nli, ase = compute_snr(x)
        return -(Pch[:, 0] / (nli + ase)).sum()

    with Timer('Optimisation'):
        sol = minimize(jax.jit(_loss), jnp.zeros((1, )), method='BFGS')
    Pch, nli, ase = compute_snr(sol.x)
    plt.plot(P['fi'][:, 0] / 1e12, 10 * jnp.log10(Pch[:, 0] / nli), '--', c=p0[0].get_color(), label='NLI optimum')
    plt.plot(P['fi'][:, 0] / 1e12, 10 * jnp.log10(Pch[:, 0] / (ase+nli)), '--', c=p2[0].get_color(),  label='Total optimum')
    plt.xlabel('Frequency [THz]')
    plt.ylabel('SNR [dB]')
    plt.legend()
    plt.grid()
    plt.show()



if __name__ == "__main__":
    main()