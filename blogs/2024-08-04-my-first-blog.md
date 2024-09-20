---
layout: page
permalink: /blogs/2024-08-04-my-first-blog/index.html
title: My First Blog
---

## Test formula

$$
\begin{align}
\int_{-\infty}^{\infty} e^{-x^2} dx = \sqrt{\pi}
\label{eq:gaussian_unnormalized}
\end{align}
$$

## Test tag for formula

$$
\begin{equation}
\begin{split}
\int_{-\infty}^{\infty} \exp\left\{-\frac{x^2}2\right\} dx &= \int_{-\infty}^{\infty} \sqrt{2} \exp\left\{-\frac{x^2}2\right\} dx/\sqrt{2} \\
&= \sqrt{2\pi} \\
\end{split}
\label{eq:gaussian}
\end{equation}
$$

Gaussion integral is a very important formula. We can refer to the formula \eqref{eq:gaussian}.

$$
\begin{align}
\mathbb E(X) = \int_{-\infty}^{\infty} x f(x) dx
\label{eq:expectation}
\end{align}
$$

The expectation of a random variable is defined as \eqref{eq:expectation}.

## Test tag for code

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 100)
y = np.exp(-x**2)

plt.plot(x, y)
plt.show()
```

## Test tag for image

<!-- ![Klee](https://chia202.github.io/images/klee1.png) -->
<div style="text-align: center;"> <img src="https://chia202.github.io/images/klee1.png" alt="Klee"> <p>Klee</p> </div>

## Test Footnote

This is a test footnote[^1].

[^1]: This is a test footnote.

## Test Table

| Header 1 | Header 2 |
|----------|----------|
| Row 1    | Row 1    |
| Row 2    | Row 2    |
