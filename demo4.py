"""
88888888ba,                                                         ,d8    
88      `"8b                                                      ,d888    
88        `8b                                                   ,d8" 88    
88         88   ,adPPYba,  88,dPYba,,adPYba,    ,adPPYba,     ,d8"   88    
88         88  a8P_____88  88P'   "88"    "8a  a8"     "8a  ,d8"     88    
88         8P  8PP"""""""  88      88      88  8b       d8  8888888888888  
88      .a8P   "8b,   ,aa  88      88      88  "8a,   ,a8"           88    
88888888Y"'     `"Ybbd8"'  88      88      88   `"YbbdP"'            88    
                       
Pretty visuals
"""

import taichi as ti
import taichi.math as tm

ti.init(arch=ti.gpu, random_seed=42)

WIDTH = 400
HEIGHT = 400
SYMBOLS = 10000

window = ti.ui.Window("Real-time constellations", res=(WIDTH * 2, HEIGHT * 2), vsync=False)
canvas = window.get_canvas()
gui = window.get_gui()

tx_symbols = ti.Vector.field(2, dtype=ti.f32, shape=SYMBOLS)
rx_symbols = ti.Vector.field(2, dtype=ti.f32, shape=SYMBOLS)
image = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))
histogram = ti.field(dtype=ti.f32, shape=(WIDTH, HEIGHT))

@ti.kernel
def fade_histogram(_retention: ti.f32):
    for x, y in histogram:
        histogram[x, y] *= _retention
@ti.func
def gray_4pam(bit1: ti.u8, bit2: ti.u8):
    sign = 2.0 * ti.cast(bit1, ti.f32) - 1.0
    magnitude = 3.0 - 2.0 * ti.cast(bit2, ti.f32)
    return sign * magnitude

@ti.kernel
def generate_frame(_noise_sigma: ti.f32, _std: ti.f32):
    scale = .2

    for k in range(tx_symbols.shape[0]):
        random_bits = ti.random(ti.u32)  # we waste most of this
        i = .36 * gray_4pam(random_bits & 1, (random_bits >> 1) & 1)
        q = .36 * gray_4pam((random_bits >> 2) & 1, (random_bits >> 3) & 1)
        tx_symbols[k][0] = i
        tx_symbols[k][1] = q

        # Its x * exp(j*phase), but there is no complex numbers in taichi
        phase = ti.randn(ti.f32) * _std
        cos_n = ti.cos(phase)
        sin_n = ti.sin(phase)
        i_n = i * cos_n - q * sin_n
        q_n = i * sin_n + q * cos_n

        i_n += _noise_sigma * ti.randn(ti.f32)
        q_n += _noise_sigma * ti.randn(ti.f32)

        rx_symbols[k][0] = i_n
        rx_symbols[k][1] = q_n

        # This maps symbols to screen space and adds to histogram
        px = (0.5 + scale * i_n) * (WIDTH - 1)
        py = (0.5 + scale * q_n) * (HEIGHT - 1)
        x = ti.cast(ti.floor(px), ti.i32)
        y = ti.cast(ti.floor(py), ti.i32)
        if 0 <= x < WIDTH and 0 <= y < HEIGHT:
            # must be atomic to ensure no two threads do addition at the same time
            ti.atomic_add(histogram[x, y], 1.0)

@ti.kernel
def render_histogram(_exposure: ti.f32):
    for x, y in histogram:
        intensity = 1.0 - ti.exp(-_exposure * histogram[x, y])

        colour = ti.Vector([
            intensity * intensity,
            intensity,
            ti.sqrt(intensity),
        ])

        # Add axes
        if x == WIDTH // 2 or y == HEIGHT // 2:
            colour = ti.max(
                colour,
                ti.Vector([0.08, 1.00, 0.11]),
            )

        image[x, y] = colour

@ti.kernel
def compute_snr() -> ti.f32:
    signal_pow = 0.0
    noise_pow = 0.0

    for k in range(tx_symbols.shape[0]):
        error_i = rx_symbols[k][0] - tx_symbols[k][0]
        error_q = rx_symbols[k][1] - tx_symbols[k][1]
        signal_pow += tx_symbols[k][0] * tx_symbols[k][0] + tx_symbols[k][1] * tx_symbols[k][1]
        noise_pow += error_i * error_i + error_q * error_q

    return 10.0 * ti.log(signal_pow / noise_pow) / ti.log(10.0)

noise_sigma = -10
retention = .98
exposure = 0.2
linewidth = 50
fs = 25

while window.running:
    noise_sigma = gui.slider_float('ASE Noise (dB)', noise_sigma, -30, 10)
    retention = gui.slider_float('Retention', retention, 0.90, 1.0)
    exposure = gui.slider_float('Exposure', exposure, 0.005, 0.40)
    linewidth = gui.slider_float('Linewidth (kHz)', linewidth, 1, 2000)
    fs = gui.slider_float('Sample Rate (GHz)', fs, 1, 500)

    fade_histogram(retention)
    generate_frame(ti.sqrt(20 ** (noise_sigma / 10)), ti.sqrt(2 * tm.pi * 1 / (fs * 1e9) * linewidth * 1e6))  # <- there is something wrong with phase noise scale.
    render_histogram(exposure)

    gui.text(f'SNR: {compute_snr():.3f}dB')

    canvas.set_image(image)
    window.show()
