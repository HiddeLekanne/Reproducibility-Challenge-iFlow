from numbers import Number

import numpy as np
import torch
from torch import distributions as dist
from torch import nn
from torch.nn import functional as F

import pdb


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers, activation='none', slope=.1, device='cpu'):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.device = device
        
        if isinstance(hidden_dim, Number):
            self.hidden_dim = [hidden_dim] * (self.n_layers - 1)
        elif isinstance(hidden_dim, list):
            self.hidden_dim = hidden_dim
        else:
            raise ValueError('Wrong argument type for hidden_dim: {}'.format(hidden_dim))

        if isinstance(activation, str):
            self.activation = [activation] * (self.n_layers - 1)
        elif isinstance(activation, list):
            self.activation = activation
        else:
            raise ValueError('Wrong argument type for activation: {}'.format(activation))

        self._act_f = []
        for act in self.activation:
            if act == 'lrelu':
                self._act_f.append(lambda x: F.leaky_relu(x, negative_slope=slope))
            elif act == 'xtanh':
                self._act_f.append(lambda x: self.xtanh(x, alpha=slope))
            elif act == 'sigmoid':
                self._act_f.append(F.sigmoid)
            elif act == 'none':
                self._act_f.append(lambda x: x)
            else:
                ValueError('Incorrect activation: {}'.format(act))

        if self.n_layers == 1:
            _fc_list = [nn.Linear(self.input_dim, self.output_dim)]
        else:
            _fc_list = [nn.Linear(self.input_dim, self.hidden_dim[0])]
            
            for i in range(1, self.n_layers - 1):
                _fc_list.append(nn.Linear(self.hidden_dim[i - 1], self.hidden_dim[i]))
            
            _fc_list.append(nn.Linear(self.hidden_dim[self.n_layers - 2], self.output_dim))
        
        self.fc = nn.ModuleList(_fc_list)
        self.to(self.device)

    @staticmethod
    def xtanh(x, alpha=.1):
        """tanh function plus an additional linear term"""
        return x.tanh() + alpha * x

    def forward(self, x):
        h = x
        for c in range(self.n_layers - 1):
            h = self._act_f[c](self.fc[c](h))
        return self.fc[-1](h)


class Dist:
    """ Base class for all distributions. """
    def __init__(self):
        pass

    def sample(self, *args):
        pass

    def log_pdf(self, *args, **kwargs):
        pass


class Normal(Dist):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        self.c = 2 * np.pi * torch.ones(1).to(self.device)
        self.log_c = torch.log(self.c)
        self._dist = dist.normal.Normal(torch.zeros(1).to(self.device), torch.ones(1).to(self.device))
        self.name = 'gauss'

    def sample(self, mu, v):
        eps = self._dist.sample(mu.size()).squeeze()
        scaled = eps.mul(v.sqrt())
        return scaled.add(mu)

    def log_pdf(self, x, mu, v, reduce=True, param_shape=None):
        """compute the log-pdf of a normal distribution with diagonal covariance"""
        if param_shape is not None:
            mu, v = mu.view(param_shape), v.view(param_shape)
        lpdf = -0.5 * (self.log_c + v.log() + (x - mu).pow(2).div(v))
        if reduce:
            return lpdf.sum(dim=-1)
        else:
            return lpdf

    def log_pdf_full(self, x, mu, v):
        """
        compute the log-pdf of a normal distribution with full covariance
        v is a batch of "pseudo sqrt" of covariance matrices of shape (batch_size, d_latent, d_latent)
        mu is batch of means of shape (batch_size, d_latent)
        """
        batch_size, d = mu.size()
        cov = torch.einsum('bik,bjk->bij', v, v)  # compute batch cov from its "pseudo sqrt"
        assert cov.size() == (batch_size, d, d)
        inv_cov = torch.inverse(cov)  # works on batches
        c = d * torch.log(self.c)
        # matrix log det doesn't work on batches!
        _, logabsdets = self._batch_slogdet(cov)
        xmu = x - mu
        return -0.5 * (c + logabsdets + torch.einsum('bi,bij,bj->b', [xmu, inv_cov, xmu]))

    def _batch_slogdet(self, cov_batch: torch.Tensor):
        """
        compute the log of the absolute value of determinants for a batch of 2D matrices. Uses torch.slogdet
        this implementation is just a for loop, but that is what's suggested in torch forums
        gpu compatible
        """
        batch_size = cov_batch.size(0)
        signs = torch.empty(batch_size, requires_grad=False).to(self.device)
        logabsdets = torch.empty(batch_size, requires_grad=False).to(self.device)
        for i, cov in enumerate(cov_batch):
            signs[i], logabsdets[i] = torch.slogdet(cov)
        return signs, logabsdets


class Laplace(Dist):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        self._dist = dist.laplace.Laplace(torch.zeros(1).to(self.device), torch.ones(1).to(self.device) / np.sqrt(2))
        self.name = 'laplace'

    def sample(self, mu, b):
        eps = self._dist.sample(mu.size())
        scaled = eps.mul(b)
        return scaled.add(mu)

    def log_pdf(self, x, mu, b, reduce=True, param_shape=None):
        """compute the log-pdf of a laplace distribution with diagonal covariance"""
        if param_shape is not None:
            mu, b = mu.view(param_shape), b.view(param_shape)
        lpdf = -torch.log(2 * b) - (x - mu).abs().div(b)
        if reduce:
            return lpdf.sum(dim=-1)
        else:
            return lpdf


class iVAE(nn.Module):
    def __init__(self, latent_dim, data_dim, aux_dim, 
                 prior=None, decoder=None, encoder=None,
                 n_layers=3, hidden_dim=50, activation='lrelu', slope=.1, 
                 trainable_prior_mean=True, device='cpu'):
        
        super(iVAE, self).__init__()

        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.aux_dim = aux_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.activation = activation
        self.slope = slope
        self.device = device

        if prior is None:
            self.prior_dist = Normal(device=device)
        else:
            self.prior_dist = prior

        if decoder is None:
            self.decoder_dist = Normal(device=device)
        else:
            self.decoder_dist = decoder

        if encoder is None:
            self.encoder_dist = Normal(device=device)
        else:
            self.encoder_dist = encoder

        # Prior parameters, crucially dependant on the auxillary variable u
        if not trainable_prior_mean:
            # As done in the iFlow paper, keep the prior mean fixed at 0
            self.prior_mean = lambda _: torch.zeros(latent_dim).to(self.device)
        else:
            self.prior_mean = MLP(aux_dim, 
                                  latent_dim, 
                                  hidden_dim, 
                                  n_layers, 
                                  activation=activation, 
                                  slope=slope, 
                                  device=device)
        self.prior_log_var = MLP(aux_dim, 
                                 latent_dim, 
                                 hidden_dim, 
                                 n_layers, 
                                 activation=activation, 
                                 slope=slope, 
                                 device=device)

        # Decoder
        self.decoder = MLP(latent_dim, 
                           data_dim, 
                           hidden_dim, 
                           n_layers, 
                           activation=activation, 
                           slope=slope, 
                           device=device)
        # Fixed decoder variance
        self.decoder_var = .01 * torch.ones(latent_dim).to(device)

        # Encoder parameters, obtained from a concatenation of u and x
        self.encoder_mean = MLP(data_dim + aux_dim, 
                                latent_dim, 
                                hidden_dim, 
                                n_layers, 
                                activation=activation, 
                                slope=slope,
                                device=device)

        self.encoder_log_var = MLP(data_dim + aux_dim, 
                                   latent_dim, 
                                   hidden_dim, 
                                   n_layers, 
                                   activation=activation, 
                                   slope=slope,
                                   device=device)
        # self.encoder_log_var = lambda _: 0.01 * torch.zeros(latent_dim).to(self.device)

        # Initialise weights of linear layers in the model with xavier uniform initialisation
        self.apply(weights_init)

    def encoder_params(self, x, u):
        xu = torch.cat((x, u), 1)
        mean = self.encoder_mean(xu)
        log_var = self.encoder_log_var(xu)
        return mean, log_var.exp() + 1e-8

    def decoder_params(self, s):
        decoded_means = self.decoder(s)
        return decoded_means, self.decoder_var

    def prior_params(self, u):
        log_var = self.prior_log_var(u)
        mean = self.prior_mean(u)
        return mean, log_var.exp() + 1e-8

    def forward(self, x, u):
        prior_params = self.prior_params(u)
        encoder_params = self.encoder_params(x, u)
        z = self.encoder_dist.sample(*encoder_params)
        decoder_params = self.decoder_params(z)

        #prior_params[0]: [1], prior_params[1]: [64, 2]
        #encoder_params[0]: [64, 2] and encoder_params[1]: [64, 2]
        # z[0]: [2], z[1]: [2]
        # decoder_params[0]: [64, 4] and decoder_params[1]: [64, 4]
        return decoder_params, encoder_params, z, prior_params

    def elbo(self, x, u):
        decoder_params, encoder_params, z, prior_params = self.forward(x, u)
        log_px_z = self.decoder_dist.log_pdf(x, *decoder_params) # log p(x|z)
        log_qz_xu = self.encoder_dist.log_pdf(z, *encoder_params) # log q(z|x,u)
        log_pz_u = self.prior_dist.log_pdf(z, *prior_params) # log p(z|u)

        return (log_px_z + log_pz_u - log_qz_xu).mean(), z

