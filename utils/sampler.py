import numpy as np
import torch


def Sampler(dist, n, loc=None, cov=None, df=None, delta=None, a=None, b=None, p=None):

    assert dist in ['Gaussian', 'Student', 'Uniform', 'Delta']
    if dist == 'Gaussian':
        x = np.random.multivariate_normal(np.array(loc), np.array(cov), size=(n,))
    elif dist == 'Student':
        assert df is not None
        z = np.random.multivariate_normal(np.zeros(len(cov)), np.array(cov), size=(n,))
        y = np.random.chisquare(df, n) / df
        x = np.array(loc) + z / np.sqrt(y).reshape(-1, 1)
    elif dist == 'Uniform':
        assert (a is not None) and (b is not None) and (p is not None)
        x = np.random.uniform(a, b, size=(n, p))
    elif dist == 'Delta':
        assert (delta is not None) and (p is not None)
        x = delta * np.ones((n, p))

    return torch.from_numpy(x).float()


def HuberSampler(**kwargs):

    c_n = int(kwargs['ns'] * kwargs['eps'])
    r_n = kwargs['ns'] - c_n
    realX = Sampler(dist=kwargs['real'], n=r_n, p=kwargs['dim'],
                    loc=kwargs['r_loc'], cov=kwargs['r_cov'], df=kwargs['r_df'],
                    delta=None, a=None, b=None)
    contX = Sampler(dist=kwargs['cont'], n=c_n, p=kwargs['dim'],
                    loc=kwargs['c_loc'], cov=kwargs['c_cov'], df=kwargs['c_df'],
                    delta=kwargs['c_delta'], a=kwargs['c_a'], b=kwargs['c_b'])

    return torch.cat([realX, contX], dim=0)