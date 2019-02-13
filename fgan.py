import os
import logging
import numpy as np
from scipy.optimize import bisect

import torch
from torch import nn,optim
import torch.nn.functional as F
import torch.utils.data as data
from torch.distributions import Normal, Chi2, StudentT

from models.network import *
from utils.sampler import HuberSampler
from utils.metrices import AveMeter, Timer
from utils.utils import kendall
from tensorboardX import SummaryWriter

logger = logging.getLogger('InfoLog')
device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')

class fgan(object):
    def __init__(self, config):
        self.config = config
        self.timer = Timer()
        self.writer = SummaryWriter(log_dir=self.config.rcd_dir)
        self._dist_init()
        self._network_init()
        self._optim_init()

    def _dist_init(self):
        self.real = self.config.real
        self.cont = self.config.cont
        if self.config.gnrt is not None:
            self.gnrt = self.config.gnrt
            if self.gnrt == 'Student':
                self.g_df = self.config.g_df
                self.chi2 = Chi2(df=self.g_df)
        self.dim = self.config.dim
        self.eps = self.config.eps

        self.HuberX = HuberSampler(**self.config.__dict__)
        self.HuberDataset = data.TensorDataset(self.HuberX)
        self.HuberLoader = data.DataLoader(self.HuberDataset,
                                           batch_size=self.config.batch_size,
                                           shuffle=False,
                                           num_workers=self.config.worker)
        logger.info('----------------------------------------------------')
        logger.info('Initialize Dataset under Huber\'s Contamination Mode')

    def _network_init(self):
        self.use_weight = self.config.use_weight
        self.use_bias = self.config.use_bias
        self.use_el = self.config.use_el
        self.use_prob = self.config.use_prob

        if self.use_el:
            self.netGXi = GeneratorXi(activation=self.config.gxi_act,
                                      hidden_units=self.config.gxi_hidden_units)
            self.netGXi.to(device)
        self.netG = Generator(dim=self.dim,
                              use_weight=self.use_weight,
                              use_bias=self.use_bias,
                              use_el=self.use_el)
        self.netG.to(device)
        self.netD = Discriminator(dim=self.dim,
                                  hidden_units=self.config.d_hidden_units,
                                  activation_1=self.config.d_act_1,
                                  activation='LeakyReLU',
                                  activation_n=self.config.d_act_n,
                                  use_prob=self.use_prob)
        self.netD.to(device)
        logger.info('Initialize Network')

        # initialization
        self._init_G()
        self._init_D(std=self.config.d_init_std)

    def _train_one_epoch(self, ep):
        self.netG.train()
        if self.use_el:
            self.netGXi.train()
        self.netD.train()
        if self.config.d_decay < 1.0:
            self.d_scheduler.step()
        if self.config.g_decay < 1.0:
            self.g_scheduler.step()

        lossD = AveMeter()
        lossG = AveMeter()
        d_iter = 1

        for ind, (xb,) in enumerate(self.HuberLoader):

            # update Discriminator
            r_x = xb.to(device)
            _, r_sc = self.netD(r_x)
            if (self.floss == 'js') or (self.floss == 'ls'):
                r_lossD = self.criterion(r_sc, torch.ones(len(xb)).to(device))
            elif self.floss == 'beta':
                r_coef = - ((r_sc+self.config.delta)**(self.alpha0-1)) * ((1-r_sc)**self.beta0)
                r_sc.backward(r_coef/len(xb))

            zb = torch.randn(len(xb), self.dim).to(device)
            if not self.use_el:
                if self.gnrt == 'Student':
                    zb.data.div_(
                        torch.sqrt(self.chi2.sample((len(xb),1)).to(device)/self.g_df) + self.config.delta
                    )
                f_x = self.netG(zb).detach()
            else:
                zb.div_(zb.norm(2, dim=1).view(-1,1) + self.config.delta)
                if self.config.use_ig:
                    ub1 = torch.randn(len(xb), self.netGXi.input_dim//2).to(device)
                    ub2 = torch.randn(len(xb), self.netGXi.input_dim - self.netGXi.input_dim//2).to(device)
                    ub2.data.div_(torch.abs(ub2.data) + self.config.delta)
                    ub = torch.cat([ub1, ub2], dim=1)
                else:
                    ub = torch.randn(len(xb), self.netGXi.input_dim).to(device)
                xib = self.netGXi(ub)
                f_x = self.netG(zb, xib).detach()
            _, f_sc = self.netD(f_x)
            if (self.floss == 'js') or (self.floss == 'ls'):
                f_lossD = self.criterion(f_sc, torch.zeros(len(xb)).to(device))
            elif self.floss == 'beta':
                f_coef = (f_sc ** self.alpha0) * ((1-f_sc+self.config.delta) ** (self.beta0-1))
                f_sc.backward(f_coef/len(xb))
                lossD.update(-1., len(xb))
            loss = r_lossD + f_lossD
            lossD.update(loss.cpu().item(), len(xb))

            self.netD.zero_grad()
            loss.backward()
            self.optD.step()
            if d_iter < self.d_steps:
                d_iter += 1
                continue
            else:
                d_iter = 1

            # update Generator
            for _ in range(self.g_steps):
                zb = torch.randn(len(xb), self.dim).to(device)
                if not self.use_el:
                    if self.gnrt == 'Student':
                        zb.data.div_(
                            torch.sqrt(self.chi2.sample((len(xb), 1)).to(device) / self.g_df) + self.config.delta
                        )
                    f_x = self.netG(zb)
                else:
                    zb.div_(zb.norm(2, dim=1).view(-1, 1) + self.config.delta)
                    if self.config.use_ig:
                        ub1 = torch.randn(len(xb), self.netGXi.input_dim // 2).to(device)
                        ub2 = torch.randn(len(xb), self.netGXi.input_dim - self.netGXi.input_dim // 2).to(device)
                        ub2.data.div_(torch.abs(ub2.data) + self.config.delta)
                        ub = torch.cat([ub1, ub2], dim=1)
                    else:
                        ub = torch.randn(len(xb), self.netGXi.input_dim).to(device)
                    xib = self.netGXi(ub)
                    f_x = self.netG(zb, xib)
                _, f_sc = self.netD(f_x)
                if (self.floss == 'js') or (self.floss == 'ls'):
                    f_lossG = - self.criterion(f_sc, torch.zeros(len(xb)).to(device))
                elif self.floss == 'beta':
                    f_coef = - (f_sc ** self.alpha0) * ((1 - f_sc + self.config.delta) ** (self.beta0 - 1))
                    f_sc.backward(f_coef / len(xb))
                    lossG.update(-1., len(xb))

                self.netG.zero_grad()
                if self.use_el:
                    self.netGXi.zero_grad()
                f_lossG.backward()
                self.optG.step()
                lossG.update(f_lossG.cpu().item(), len(xb))
        # logger.info(f'Epoch: [{ep}/{self.epochs} |'
        #             f'Time: {self.timer.timeSince()} |'
        #             f'LossD: {lossD.avg:.4f} |'
        #             f'LossG: {lossG.avg:.4f} |')
        self.writer.add_scalar('lossD', lossD.avg, ep)
        self.writer.add_scalar('lossG', lossG.avg, ep)

    def train(self):
        self.epochs = self.config.epochs
        self.floss = self.config.floss
        if self.floss == 'js':
            self.criterion = nn.BCEWithLogitsLoss()
        elif self.floss == 'ls':
            self.criterion = nn.MSELoss()
        elif self.floss == 'beta':
            self.alpha0 = self.config.alpha0
            self.beta0 = self.config.beta0

        if self.use_weight:
            cov_est_record = []
        if self.use_bias:
            loc_est_record = []

        for ep in range(1, self.config.epochs+1):

            self._train_one_epoch(ep)

            if ep % self.config.val_period == 0:
                logger.info(f'Epoch: [{ep}/{self.config.epochs}]')
                logger.info(f'Validation Starts: ')
                if self.use_weight:
                    cov_error = self._cov_error()
                    logger.info(f'Scatter Matrix Estimation Error: %.4f' % cov_error)
                    self.writer.add_scalar('ScatterError', cov_error, ep)
                if self.use_bias:
                    loc_error = self._loc_error()
                    logger.info(f'Location Estimation Error: %.4f' % loc_error)
                    self.writer.add_scalar('LocationError', loc_error, ep)

            if ep > (self.config.epochs - self.config.avg_epochs):
                if self.use_weight:
                    if self.use_el:
                        fact = self._reweight() + self.config.delta
                        cov_est_record.append(1/fact * self.netG.weight.data.clone().cpu())
                    else:
                        cov_est_record.append(self.netG.weight.data.clone().cpu())
                if self.use_bias:
                    loc_est_record.append(self.netG.bias.data.clone().cpu())

        if self.use_weight:
            avg_cov_est = sum(cov_est_record)/len(cov_est_record)
            avg_cov_err = self._cov_error(weight = avg_cov_est)
            last_cov_err = self._cov_error()
            logger.info(f'----------------------Final Result------------------------------'
                        f'Average/Final Scatter Estimation Error: {avg_cov_err:.4f} {last_cov_err:.4f}')
        if self.use_bias:
            avg_loc_est = sum(loc_est_record)/len(loc_est_record)
            avg_loc_err = self._loc_error(loc=avg_loc_est)
            last_loc_err = self._loc_error()
            logger.info(f'----------------------Final Result------------------------------'
                        f'Average/Final Location Estimation Error: {avg_loc_err:.4f} {last_loc_err:.4f}')
        logger.info(f'End within {self.timer.timeSince()}')

        self.writer.close()

    def _init_G(self):
        if self.use_weight:
            # scaled Kendall's \tau estimator
            if self.config.g_init == 'kendall':
                cov_init = kendall(self.HuberX, self.config)
            if self.config.g_init == 'diag':
                medX2 = np.median(self.HuberX.numpy()**2, axis=0)
                cov_init = np.diag(medX2)
            # cov = USU^T, W = S^(1/2)U^T
            u, s, vt = np.linalg.svd(cov_init)
            weight_init = np.matmul(np.diag(s)**(1/2), vt)
            self.netG.weight.data.copy_(torch.from_numpy(weight_init).float().to(device))
            init_cov_error = self._cov_error()
            logger.info('Initialization Scatter Matrix Error: %.4f' % init_cov_error)
            self.writer.add_scalar('ScatterError', init_cov_error, 0)
        if self.use_bias:
            Xmed = torch.median(self.HuberX, dim=0)[0]
            self.netG.bias.data.copy_(Xmed.to(device))
            init_loc_error = self._loc_error()
            logger.info('Initialization Location Error: %.4f' % init_loc_error)
            self.writer.add_scalar('LocationError', init_loc_error, 0)
        if self.use_el:
            self.netGXi.apply(weights_init_xavier)

    def _init_D(self, std):
        self.netD.apply(weights_init_xavier)
        if std is not None:
            self.netD.feature.lyr1.weight.data.normal_(0, std)

    def _cov_error(self, weight=None):
        with torch.no_grad():
            if weight is None:
                cov_est = torch.mm(self.netG.weight.transpose(1,0), self.netG.weight)
                if self.use_el:
                    fact = self._reweight() + self.config.delta
                    cov_est.data.div_(fact**2)
            else:
                cov_est = torch.mm(weight.transpose(1, 0), weight)
        cov_err = np.linalg.norm(cov_est.cpu().numpy() - self.config.r_cov, ord=2)
        return cov_err

    def _loc_error(self, loc=None):
        if loc is None:
            loc_err = (self.netG.bias.cpu() - self.t_loc).norm(2).item()
        else:
            loc_err = (loc.cpu() - self.t_loc).norm(2).item()
        return loc_err

    def _reweight(self, N=100000):
        # Expect value: \mathbb{E}_{x~X}Ramp(|x|)
        if not hasattr(self, 'epv'):
            self.Hfunc = self.config.Hfunc
            # self.Hfunc = 'ramp'
            if self.real == 'Student':
                tdist = StudentT(df=self.config.r_df)
                x = tdist.sample((5000000,))
            elif self.real == 'Gaussian':
                ndist = Normal(0, 1)
                x = ndist.sample((5000000,))
            self.epv = self._HFunc(x, mode=self.Hfunc).mean().item()

        def sov_func(a, bs=1000):
            # find a suitable factor a to match expected value.
            r = AveMeter()
            for _ in range(N//bs):
                if self.config.use_ig:
                    ub1 = torch.randn(bs, self.netGXi.input_dim//2).to(device)
                    ub2 = torch.randn(bs, self.netGXi.input_dim - self.netGXi.input_dim//2).to(device)
                    ub2.data.div_(torch.abs(ub2.data) + self.config.delta)
                    ub = torch.cat([ub1, ub2], dim=1)
                else:
                    ub = torch.randn(bs, self.netGXi.input_dim).to(device)
                with torch.no_grad():
                    xib = self.netGXi(ub)
                zb = torch.randn(bs, self.dim).to(device)
                vu = (zb[:,0].div_(zb.norm(2, dim=1)) + self.config.delta).to(device)
                r.update(self._HFunc(a * xib * vu, mode=self.Hfunc).mean().item(), bs)
            return r.avg - self.epv
        # if sov_func(1) > 0: down,up= 0,3
        # elif sov_func(3) > 0: down,up = 0,5
        # elif sov_func(10) > 0: down,up = 1,12
        # elif sov_func(25) > 0: down,up = 8,27
        # elif sov_func(75) > 0: down,up = 23,77
        if sov_func(250) > 0:
            down, up = 0, 3000
        else:
            logger.info('Factor is larger than 2500!')
            return 250
        factor = bisect(sov_func, down, up)
        print(factor)
        return factor

    def _HFunc(self, x, mode):
        if mode == 'abs':
            return torch.abs(x)
        elif mode == 'ramp':
            return F.hardtanh(torch.abs(x))

    def _optim_init(self):
        if self.use_el:
            self.optG = optim.SGD(params=list(self.netG.parameters())+list(self.netGXi.parameters()),
                                  lr=self.config.g_lr)
        else:
            self.optG = optim.SGD(params=self.netG.parameters(),
                                  lr=self.config.g_lr)
        self.optD = optim.SGD(params=self.netD.parameters(),
                              lr=self.config.d_lr)
        self.g_steps = self.config.g_steps
        self.d_steps = self.config.d_steps
        if self.config.d_decay < 1.0:
            self.d_scheduler = optim.lr_scheduler.StepLR(self.optD,
                                                         step_size=self.config.d_sch,
                                                         gamma=self.config.d_decay)
        if self.config.g_decay < 1.0:
            self.g_scheduler = optim.lr_scheduler.StepLR(self.optG,
                                                         step_size=self.config.g_sch,
                                                         gamma=self.config.g_decay)