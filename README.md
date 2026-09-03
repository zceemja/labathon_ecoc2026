# ECOC ~~2025~~ 2026 Hack your research: Using JAX to crunch numbers fast

Have you ever wanted to run a simulation, but it will take 40 days to finish and ECOC deadline is in the week? **Worry not!** This hack demo will show the nice goodies that python JAX framework ecosystem has to offer and how your slow code can be turned into a research paper.

# Labathon?
Lab + Hackathon. I've been proposing this name since ECOC2022 but organisers keep refusing me.
 
# How do I run this demo?!

Here is a quick python hack: use [uv](https://docs.astral.sh/uv/). It makes a lot less headaches than pip+venv or conda (if you can still use it in your university).
I highly recommend checking it out because you can do cool thinks for instance have [headacheless matlab - python setup](pybridge.m).

Anyway, to run it you need uv
## Install uv
(need to do this only once)
```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
```
or on windows:
```shell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
This will install uv locally on your account (in ~/.local/bin or %APPDATA%)

## Run

As simple as:
```shell
uv run demo1.py
```

# Cool stuff

* [Patrick Kidger](https://github.com/patrick-kidger)
* [Awesome JAX](https://github.com/n2cholas/awesome-jax) [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
* [Neural Networks](https://flax.readthedocs.io/en/stable/)
* [Reinforcement Learning](https://chrislu.page/blog/meta-disco/)

### Optimisation:
* [optax](https://optax.readthedocs.io/en/latest/api/optimizers.html)
* [jaxopt](https://jaxopt.github.io/stable/index.html)
* [optimistix](https://docs.kidger.site/optimistix/)
* [diffrax](https://docs.kidger.site/diffrax/)
