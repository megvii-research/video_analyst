import numpy as np


def normalize(score):
    score = (score - np.min(score)) / (np.max(score) - np.min(score))
    return score


def normfun(x, mu, sigma):
    pdf = np.exp(-((x - mu)**2) / (2 * sigma**2))
    return pdf


def generate_xy_attention(center, size):

    a = np.linspace(-size // 2 + 1, size // 2, size)
    x = -normfun(a, center[1], 10).reshape((size, 1)) + 2
    y = -normfun(a, center[0], 10).reshape((1, size)) + 2
    z = normalize(1. / np.dot(np.abs(x), np.abs(y)))
    return z


if __name__ == '__main__':
    generate_xy_attention([0, 0], 31)
