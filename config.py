import os
import logging
import argparse
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('--real', default='Gaussian', type=str, choices=['Gaussian', 'Student', 'Cauchy'])
parser.add_argument('--cont', default='Gaussian', type=str,
                    choices=['Gaussian', 'Student', 'Cauchy', 'Uniform', 'Delta'])
parser.add_argument('--gnrt', default=None, type=str, choices=['Gaussian', 'Student'])
parser.add_argument('--dim', type=int, default=100)
parser.add_argument('--eps', type=float, default=0.2)
parser.add_argument('--ns', type=int, default=50000)

parser.add_argument('--r_loc', default=None, type=float)
parser.add_argument('--c_loc', default=None, type=float)
parser.add_argument('--r_cov_type', default=None, type=str, choices=['spherical', 'ar', 'wis', 'sp'])
parser.add_argument('--c_cov_type', default=None, type=str, choices=['spherical', 'ar', 'wis', 'sp'])
parser.add_argument('--r_var', default=None, type=float)
parser.add_argument('--c_var', default=None, type=float)
# Student t distribution degree of freedom
parser.add_argument('--r_df', default=None, type=float)
parser.add_argument('--c_df', default=None, type=float)
parser.add_argument('--g_df', default=None, type=float)
# Uniform distribution parameter
parser.add_argument('--c_a', default=None, type=float)
parser.add_argument('--c_b', default=None, type=float)
# Delta distribution parameter
parser.add_argument('--c_delta', default=None, type=float)

# Use default hyper-parameter
parser.add_argument('--use_default', type=bool, default=True)

# Network structure
parser.add_argument('--d_hidden_units', nargs='+', default=None, type=int)
parser.add_argument('--d_act_1', type=str, default=None, choices=['LeakyReLU', 'LogReLU', 'Sigmoid', 'ReLU'])
parser.add_argument('--d_act_n', type=str, default=None, choices=['LeakyReLU', 'Sigmoid', 'ReLU'])
parser.add_argument('--d_init_std', type=float, default=None)
parser.add_argument('--gxi_hidden_units', default=None, nargs='+', type=int)
parser.add_argument('--gxi_act', type=str, default=None, choices=['LeakyReLU', 'Sigmoid', 'ReLU'])
parser.add_argument('--g_init', type=str, default=None, choices=['diag', 'kendall'])
parser.add_argument('--use_el', '-el', action='store_true')
parser.add_argument('--use_prob', '-pb', action='store_true')
parser.add_argument('--use_weight', type=bool, default=True)
parser.add_argument('--use_bias', type=bool, default=False)
parser.add_argument('--use_ig', type=bool, default=True)

# Optimizer Arguments
parser.add_argument('--d_lr', type=float, default=None)
parser.add_argument('--d_steps', type=int, default=None)
parser.add_argument('--d_decay', type=float, default=None)
parser.add_argument('--d_sch', type=int, default=None)
parser.add_argument('--g_lr', type=float, default=None)
parser.add_argument('--g_steps', type=int, default=None)
parser.add_argument('--g_decay', type=float, default=None)
parser.add_argument('--g_sch', type=int, default=None)
parser.add_argument('--epochs', type=int, default=None)
parser.add_argument('--avg_epochs', type=int, default=None)
parser.add_argument('--batch_size', type=int, default=None)

# Training Settings
parser.add_argument('--floss', default='js', choices=['js', 'ls', 'beta'], type=str)
parser.add_argument('--rcd_dir', type=str, default='./results/')
parser.add_argument('--rcd_name', type=str)
parser.add_argument('--val_period', default=None, type=int)
parser.add_argument('--delta', default=1e-5, type=float)
parser.add_argument('--worker', default=1, type=int)
parser.add_argument('--Hfunc', default='abs', type=str, choices=['abs', 'ramp'])
parser.add_argument('--subsample', default=5000, type=int)

def get_config():

    config = parser.parse_args()

    # Huber\'s Contamination Model Settings
    for attr in ['real', 'cont', 'gnrt']:
        if getattr(config, attr) == 'Cauchy':
            setattr(config, attr, 'Student')
            setattr(config, attr[0]+'_df', 1)

    if config.r_cov_type is None:
        config.r_cov_type = 'spherical'
        config.r_var = 1.0
    if config.r_loc is None:
        config.r_loc = np.zeros(config.dim)
    else:
        config.r_loc = config.r_loc * np.ones(config.dim)

    if config.r_cov_type == 'spherical':
        assert config.r_var is not None
        config.r_cov = config.r_var * np.diag(np.ones(config.dim))
    else:
        assert config.dim == 100
        if config.r_cov_type == 'ar':
            config.r_cov = np.load('./datasets/cov/ar_p100_k1.0_t0.5.npy')
        elif config.r_cov_type == 'wis':
            config.r_cov = np.load('./datasets/cov/wis_p100_df100_sc1.0.npy')
        elif config.r_cov_type == 'sp':
            config.r_cov = np.load('./datasets/cov/sp_p100_s0.5.npy')
        else:
            config.r_cov = None

    if config.cont in ['Gaussian', 'Student']:
        assert config.c_loc is not None
        config.c_loc = config.c_loc * np.ones(config.dim)
        if config.c_cov_type == 'spherical':
            assert config.c_var is not None
            config.c_cov = config.c_var * np.diag(np.ones(config.dim))
        else:
            assert config.dim == 100
            if config.c_cov_type == 'ar':
                config.c_cov = np.load('./datasets/cov/ar_p100_k5.0_t0.8.npy')
            elif config.c_cov_type == 'wis':
                config.c_cov = np.load('./datasets/cov/wis_p100_df100_sc5.0.npy')
            elif config.c_cov_type == 'sp':
                config.c_cov = np.load('./datasets/cov/sp_p100_s0.5.npy')
            else:
                config.c_cov = None
    else:
        config.c_cov = None

    # Network Structure Settings
    if not config.use_el:
        if config.gnrt == None:
            # unless specified particularly, we use the real distribution for the choice of generator.
            config.gnrt = config.real
            config.g_df = config.r_df
        else:
            if config.gnrt == 'Student':
                assert config.g_df is not None

    # Training Settings
    assert config.rcd_name is not None
    config.rcd_dir = os.path.join(config.rcd_dir, config.rcd_name)
    if not os.path.exists(config.rcd_dir):
        os.mkdir(config.rcd_dir)
    if config.val_period is None:
        if config.use_el:
            config.val_period = 25
        else:
            config.val_period = 10

    if config.use_default:
        if not config.use_el:
            config.g_lr = 0.1 if config.g_lr is None else config.g_lr
            config.d_lr = 0.025 if config.d_lr is None else config.d_lr
            config.d_decay = 1.0 if config.d_decay is None else config.d_decay
            config.g_decay = 0.2 if config.g_decay is None else config.g_decay
            config.g_sch = 200 if config.g_sch is None else config.g_sch
            config.g_steps = 3 if config.g_steps is None else config.g_steps
            config.d_steps = 12 if config.d_steps is None else config.d_steps
            config.d_init_std = .0025 if config.d_init_std is None else config.d_init_std
            config.g_init = 'kendall' if config.g_init is None else config.g_init
            config.epochs = 450 if config.epochs is None else config.epochs
            config.avg_epochs = 25 if config.avg_epochs is None else config.avg_epochs
            config.batch_size = 500 if config.batch_size is None else config.batch_size
            config.d_act_1 = 'LeakyReLU' if config.d_act_1 is None else config.d_act_1
            config.d_act_n = 'Sigmoid' if config.d_act_n is None else config.d_act_n
            config.d_hidden_units = [200, 25] if config.d_hidden_units is None else config.d_hidden_units
        else:
            config.g_lr = 0.025 if config.g_lr is None else config.g_lr
            config.d_lr = 0.05 if config.d_lr is None else config.d_lr
            config.d_decay = 1.0 if config.d_decay is None else config.d_decay
            config.g_decay = 0.2 if config.g_decay is None else config.g_decay
            config.g_sch = 200 if config.g_sch is None else config.g_sch
            config.g_steps = 1 if config.g_steps is None else config.g_steps
            config.d_steps = 5 if config.d_steps is None else config.d_steps
            config.d_init_std = .025 if config.d_init_std is None else config.d_init_std
            config.g_init = 'diag' if config.g_init is None else config.g_init
            config.epochs = 450 if config.epochs is None else config.epochs
            config.avg_epochs = 25 if config.avg_epochs is None else config.avg_epochs
            config.batch_size = 500 if config.batch_size is None else config.batch_size
            config.d_act_1 = 'LeakyReLU' if config.d_act_1 is None else config.d_act_1
            config.d_act_n = 'Sigmoid' if config.d_act_n is None else config.d_act_n
            config.d_hidden_units = [200, 25] if config.d_hidden_units is None else config.d_hidden_units
            config.gxi_hidden_units = [48, 32, 24, 12] if config.gxi_hidden_units is None else config.gxi_hidden_units
            config.gxi_act = 'LeakyReLU' if config.gxi_act is None else config.gxi_act

    # Logger
    logger = logging.getLogger('InfoLog')
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(os.path.join(config.rcd_dir, f'log.txt'))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)

    logger.addHandler(handler)
    logger.addHandler(console)

    return config
