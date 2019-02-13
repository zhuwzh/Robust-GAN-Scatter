import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_ as xavier_normal_
from collections import OrderedDict


# Some candidate activation functions
class myact_Ramp(nn.Module):

    def __init__(self):
        super(myact_Ramp, self).__init__()

    def forward(self, x):
        return .5 * (F.hardtanh(input, min_val=-0.5, max_val=0.5) + 1)


class myact_LogReLU(nn.Module):
    def __init__(self):
        super(myact_LogReLU, self).__init__()

    def forward(self, x):
        return torch.log(1 + F.relu(x))


class GeneratorXi(nn.Module):
    def __init__(self, activation=None, hidden_units=None, input_dim=None):
        # activation, 'Sigmoid'/'ReLU'/'LeakyReLU'
        super(GeneratorXi, self).__init__()
        self.arg = {'negative_slope': 0.2} if (activation == 'LeakyReLU') else {}
        self.activation = activation
        if input_dim is None:
            self.input_dim = hidden_units[0]
        else:
            self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.layers = len(self.hidden_units)
        self.map = self._make_layers()

    def _make_layers(self):

        layer_list = []
        for lyr in range(self.layers):
            if lyr == 0:
                layer_list += [('lyr%d' % (lyr + 1), nn.Linear(self.input_dim, self.hidden_units[lyr])),
                               ('act%d' % (lyr + 1), getattr(nn, self.activation)(**self.arg))]
            else:
                layer_list += [('lyr%d' % (lyr + 1), nn.Linear(self.hidden_units[lyr - 1], self.hidden_units[lyr])),
                               ('act%d' % (lyr + 1), getattr(nn, self.activation)(**self.arg))]

        layer_list += [('lyr%d' % (self.layers + 1), nn.Linear(self.hidden_units[-1], 1))]

        return nn.Sequential(OrderedDict(layer_list))

    def forward(self, z):

        xi = self.map(z.view(-1, self.input_dim))
        xi = torch.abs(xi)

        return xi


class Generator(nn.Module):
    def __init__(self, dim, use_weight=True, use_bias=True, use_el=False):

        super(Generator, self).__init__()
        self.dim = dim
        self.use_weight = use_weight
        self.use_bias = use_bias
        if self.use_weight:
            # Note covariance matrix is W^TW
            self.weight = nn.Parameter(torch.eye(self.dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(self.dim))
        self.ues_el = use_el

    def forward(self, z, xi=None):
        if self.use_weight:
            x = z.view(-1, self.dim).mm(self.weight)
        else:
            x = z.view(-1, self.dim)
        if self.ues_el:
            x = xi * x
        if self.use_bias:
            x = x + self.bias
        return x


class Discriminator(nn.Module):
    def __init__(self, dim, hidden_units,
                 activation_1, activation, activation_n, use_prob=False):
        super(Discriminator, self).__init__()
        self.dim = dim
        self.use_prob = use_prob

        assert activation in ['ReLU', 'Sigmoid', 'LeakyReLU']
        assert activation_1 in ['LogReLU', 'Sigmoid', 'ReLU', 'LeakyReLU']
        assert activation_n in ['ReLU', 'Sigmoid', 'LeakyReLU']
        self.arg = {'negative_slope': 0.2} if (activation == 'LeakyReLU') else {}
        self.activation = activation
        self.arg_1 = {'negative_slope': 0.2} if (activation_1 == 'LeakyReLU') else {}
        self.activation_1 = activation_1
        self.arg_n = {'negative_slope': 0.2} if (activation_n == 'LeakyReLU') else {}
        self.activation_n = activation_n

        self.layers = len(hidden_units)
        self.hidden_units = hidden_units

        self.feature = self._make_layers()
        self.map_last = nn.Linear(self.hidden_units[-1], 1)
        if self.use_prob:
            self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.feature(x.view(-1, self.dim))
        d = self.map_last(x).squeeze()
        if self.use_prob:
            d = self.sigmoid(d)
        return x, d

    def _make_layers(self):

        layer_list = []
        for lyr in range(self.layers):
            if lyr == 0:
                if self.activation_1 in ['Sigmoid', 'ReLU', 'LeakyReLU']:
                    layer_list += [('lyr%d' % (lyr + 1), nn.Linear(self.dim, self.hidden_units[lyr])),
                                   ('act%d' % (lyr + 1), getattr(nn, self.activation_1)(**self.arg_1))]
                elif self.activation_1 in ['LogReLU']:
                    layer_list += [('lyr%d' % (lyr + 1), nn.Linear(self.dim, self.hidden_units[lyr])),
                                   ('act%d' % (lyr + 1), eval('myact_' + self.activation_1)())]
            elif lyr == (self.layers - 1):
                layer_list += [('lyr%d' % (lyr + 1), nn.Linear(self.hidden_units[lyr - 1], self.hidden_units[lyr])),
                               ('act%d' % (lyr + 1), getattr(nn, self.activation_n)(**self.arg_n))]
            else:
                layer_list += [('lyr%d' % (lyr + 1), nn.Linear(self.hidden_units[lyr - 1], self.hidden_units[lyr])),
                               ('act%d' % (lyr + 1), getattr(nn, self.activation)(**self.arg))]

        return nn.Sequential(OrderedDict(layer_list))


def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
      xavier_normal_(m.weight)
      m.bias.data.fill_(0.0)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
      m.weight.data.normal_(0.0, 0.02)
      m.bias.data.fill_(0.0)