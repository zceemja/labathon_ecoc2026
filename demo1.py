"""
       ....
   .xH888888Hx.                                                          oe
 .H8888888888888:                  ..    .     :           u.          .@88
 888*""'?""*88888X        .u     .888: x888  x888.   ...ue888b     ==*88888
'f     d8x.   ^%88k    ud8888.  ~`8888~'888X`?888f`  888R Y888r       88888
'>    <88888X   '?8  :888'8888.   X888  888X '888>   888R I888>       88888
 `:..:`888888>    8> d888 '88%"   X888  888X '888>   888R I888>       88888
        `"*88     X  8888.+"      X888  888X '888>   888R I888>       88888
   .xHHhx.."      !  8888L        X888  888X '888>  u8888cJ888        88888
  X88888888hx. ..!   '8888c. .+  "*88%""*88" '888!`  "*888*P"         88888
 !   "*888888888"     "88888%      `~    "    `"`      'Y"            88888
        ^"***"`         "YP'                                       '**%%%%%%**

This demo shows performance improvements with jit
"""

from functions import *

def main():
    key = jax.random.PRNGKey(0)

    with Timer("Running generate_signal #1"):
        fs, tx, tx_seq, superch = generate_signal(100e9, 48e9, 96e9, 10, 2 ** 13, 'gaussian', key, 0.01, ch_power_dBm=0)

    key1, key2, key3 = jax.random.split(key, 3)
    with Timer(f"Running generate_signal jit #1"):
        generate_signal_jit = jax.jit(lambda x: generate_signal(100e9, 48e9, 96e9, 10, 2 ** 13, 'gaussian', x, 0.01, ch_power_dBm=0))
        fs, tx, tx_seq, superch = generate_signal_jit(key1)
        jax.block_until_ready(superch)
        # print(power_meter(superch))

    with Timer(f"Running generate_signal jit #2"):
        fs, tx, tx_seq, superch = generate_signal_jit(key2)
        jax.block_until_ready(superch)
        # print(power_meter(superch))

    with Timer(f"Running generate_signal jit #3"):
        fs, tx, tx_seq, superch = generate_signal_jit(key3)
        jax.block_until_ready(superch)
        # print(power_meter(superch))
    show_spectrum(superch, fs)

if __name__ == '__main__':
    print(f"Device: {jax.devices()[0].device_kind}")
    main()
