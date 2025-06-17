<p align="center">
<img width="450" src="./logo.png"/>
</p>

[![codecov](https://codecov.io/gh/probcomp/genjax/branch/main/graph/badge.svg?token=AGORyvyVeu)](https://codecov.io/gh/probcomp/genjax)

> This is the research version of GenJAX, a [(more stable) community version can be found here](https://github.com/genjax-community/genjax).

(**Probabilistic programming language**) GenJAX is a probabilistic programming language (PPL): a system which provides automation for writing programs which perform computations on probability distributions, including sampling, variational approximation, gradient estimation for expected values, and more.

(**With programmable inference**) The design of GenJAX is centered on _programmable inference_: automation which allows users to express and customize Bayesian inference algorithms (algorithms for computing with posterior distributions: "_x_ affects _y_, and I observe _y_, what are my new beliefs about _x_?"). Programmable inference includes advanced forms of Monte Carlo and variational inference methods.

GenJAX's automation is based on two key concepts: _generative functions_ (GenJAX's version of probabilistic programs) and _traces_ (samples from probabilistic programs). GenJAX provides:

- Modeling language automation for constructing complex probability distributions from pieces
- Inference automation for constructing Monte Carlo samplers using convenient idioms (programs expressed by creating and editing traces), and [variational inference automation](https://dl.acm.org/doi/10.1145/3656463) using [new extensions to automatic differentation for expected values](https://dl.acm.org/doi/10.1145/3571198).

(**Fully vectorized & compatible with JAX**) All of GenJAX's automation is fully compatible with JAX, implying that any program written in GenJAX can be `vmap`'d and `jit` compiled.
