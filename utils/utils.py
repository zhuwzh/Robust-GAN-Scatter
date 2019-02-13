import numpy as np
from scipy.stats import kendalltau, t, norm

# def scale_kendall(X, subsample=None):
#     # scaling factor
#     medX = np.median(X, axis=0)
#     X = X - medX
#     s = np.median(X**2, axis=0)**(1/2)
#     # kendall's \tau correlation with sub-sampling
#     if subsample is not None:
#         assert subsample < len(X)
#         indices = np.random.choice(len(X), size=subsample, replace=False)
#         X = X[indices]
#     n, p = X.shape
#     kenmat = np.zeros((p, p))
#     for i in range(n):
#         for j in range(i):
#             v = X[i] - X[j]
#             kenmat += np.outer(v, v)/np.inner(v, v)
#     kenmat = kenmat * 2 / (n*(n-1))
#     # scaling
#     cov = s.reshape(p, 1) * kenmat * s.reshape(1, p)
#     return cov


def kendall(Y, config, subsample=None):
    # X is torch.tensor N by p
    X = Y.numpy()
    # scaling factor
    medX = np.median(X, axis=0)
    X = X - medX
    # median absolute deviation
    s = np.median(np.abs(X), axis=0)
    # std = k * MAD with k = 1/F^{-1}(3/4), where F is dist of real
    if config.real == 'Gaussian':
        k = 1/norm.ppf(3/4)
    elif config.real == 'Student':
        k = 1/t.ppf(3/4, df=config.r_df)
    s = k * s
    # sub-sampling
    if subsample is not None:
        assert subsample <= len(X)
        indices = np.random.choice(len(X), size=subsample, replace=False)
        X = X[indices]
    _, p = X.shape
    corr = np.zeros((p, p))
    for i in range(p):
        for j in range(i + 1):
            corr[i, j] = np.sin(np.pi / 2 * kendalltau(Y[:, i], Y[:, j])[0])
            corr[j, i] = corr[i, j]
    cov = s.reshape(p, 1) * corr * s.reshape(1, p)
    return cov
