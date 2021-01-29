from numbers import Number

import numpy as np
import torch
from torch import distributions as dist
from torch import nn
from torch.nn import functional as F

import pdb


def weights_init(module):
    """ Initialize weights of given module with xavier_uniform_ if the module is a fully connected layer. """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight.data)


def str_to_act(act, slope=0.1):
    """ Given a string, returns the corresponding activation function. """
    if act == 'relu':
        return nn.ReLU()
    elif act == 'lrelu':
        return nn.LeakyReLU(slope)
    elif act == 'tanh':
        return nn.Tanh()
    elif act == 'xtanh':
        return xTanh(alpha=slope)
    elif act == 'sigmoid':
        return nn.Sigmoid()
    elif act == 'none':
        return nn.Identity()
    else:
        ValueError('Incorrect activation: {}'.format(act))


class xTanh(nn.Module):
    def __init__(self, alpha=0.1):
        super(xTanh, self).__init__()

        self.alpha = alpha

    def forward(self, x):
        """ Hyperbolic tangent plus an additional linear term. """
        return x.tanh() + self.alpha * x


class Skip_Layer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Skip_Layer, self).__init__()

        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.fc(x) + x


class MLP(nn.Module):
    """ Multilayer Perceptron (MLP) as a PyTorch module. """
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers, activation='none', slope=.1, device='cpu', ln = False, skip = False):
        super(MLP, self).__init__()

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

        # Build a list of layers
        dimensions = [self.input_dim] + self.hidden_dim
        layers = []

        for i in range(1, len(dimensions)):
            if dimensions[i - 1] == dimensions[i] and skip:
                layers.append(Skip_Layer(dimensions[i - 1], dimensions[i]))
            else:
                layers.append(nn.Linear(dimensions[i - 1], dimensions[i]))
            if ln:
                layers.append(nn.LayerNorm(dimensions[i]))

            layers.append(str_to_act(self.activation[i-1]))

        layers.append(nn.Linear(dimensions[-1], self.output_dim))

        # Create nn.Sequential with all layers and move the model to the correct device
        self.layers = nn.Sequential(*layers)
        self.to(self.device)

    def forward(self, x):
        """ Performs a forward pass through the layers of the model using nn.Sequential. """
        return self.layers(x)


class Dist:
    """ Base class for all distributions. """
    def __init__(self):
        pass

    def sample(self, *args):
        pass

    def log_pdf(self, *args, **kwargs):
        pass


class Normal(Dist):
    """ Normal distribution using PyTorch, without fixed mean or (co)variance. """
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        self._dist = dist.normal.Normal(torch.zeros(1).to(self.device), torch.ones(1).to(self.device))
        self.name = 'gauss'

        # Compute normalizing constant and its log for usage in log_pdf and log_pdf_full
        self.c = 2 * np.pi * torch.ones(1).to(self.device)
        self.log_c = torch.log(self.c)

    def sample(self, mu, v):
        """ Draw a sample from the distribution. """
        eps = self._dist.sample(mu.size()).squeeze()
        scaled = eps.mul(v.sqrt())
        return scaled.add(mu)

    def log_pdf(self, x, mu, v, reduce=True, param_shape=None):
        """ Compute the log-pdf of a normal distribution with diagonal covariance. """
        if param_shape is not None:
            mu, v = mu.view(param_shape), v.view(param_shape)
        lpdf = -0.5 * (self.log_c + v.log() + (x - mu).pow(2).div(v))
        if reduce:
            return lpdf.sum(dim=-1)
        else:
            return lpdf

    def log_pdf_full(self, x, mu, v):
        """
        Compute the log-pdf of a normal distribution with full covariance.

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
        Compute the log of the absolute value of determinants for a batch of 2D matrices. Uses torch.slogdet
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
    """ LaPlace distribution using PyTorch without fixed mean or (co)variance. """
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        self._dist = dist.laplace.Laplace(torch.zeros(1).to(self.device), torch.ones(1).to(self.device) / np.sqrt(2))
        self.name = 'laplace'

    def sample(self, mu, b):
        """ Draw a sample from the distribution. """
        eps = self._dist.sample(mu.size())
        scaled = eps.mul(b)
        return scaled.add(mu)

    def log_pdf(self, x, mu, b, reduce=True, param_shape=None):
        """ Compute the log-pdf of a laplace distribution with diagonal covariance. """
        if param_shape is not None:
            mu, b = mu.view(param_shape), b.view(param_shape)
        lpdf = -torch.log(2 * b) - (x - mu).abs().div(b)
        if reduce:
            return lpdf.sum(dim=-1)
        else:
            return lpdf


class iVAE(nn.Module):
    """ Identifiable Variational Autoencoder. """
    def __init__(self, latent_dim, data_dim, aux_dim,
                 prior=None, decoder=None, encoder=None,
                 n_layers=3, hidden_dim=50, activation='lrelu', slope=.1,
                 trainable_prior_mean=True, device='cpu',
                 ln = False, skip = False):

        super(iVAE, self).__init__()

        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.aux_dim = aux_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.activation = activation
        self.slope = slope
        self.device = device

        # Set distributions, defaulting to Normal distributions if none were provided
        self.prior_dist = Normal(device=device) if prior is None else prior
        self.decoder_dist = Normal(device=device) if decoder is None else decoder
        self.encoder_dist = Normal(device=device) if encoder is None else encoder

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
                           device=device,
                           ln = ln,
                           skip = skip)
        # Fixed decoder variance
        self.decoder_var = .01 * torch.ones(latent_dim).to(device)

        # Encoder parameters, obtained from a concatenation of u and x
        self.encoder_mean = MLP(data_dim + aux_dim,
                                latent_dim,
                                hidden_dim,
                                n_layers,
                                activation=activation,
                                slope=slope,
                                device=device,
                                ln = ln,
                                skip = skip)
        self.encoder_log_var = MLP(data_dim + aux_dim,
                                   latent_dim,
                                   hidden_dim,
                                   n_layers,
                                   activation=activation,
                                   slope=slope,
                                   device=device,
                                   ln = ln,
                                   skip = skip)

        # Initialise weights of linear layers in the model with xavier uniform initialisation
        self.apply(weights_init)

    def encoder_params(self, x, u):
        """ Encode x and u to get means and variances in the latent space. """
        xu = torch.cat((x, u), 1)
        mean = self.encoder_mean(xu)
        log_var = self.encoder_log_var(xu)
        return mean, log_var.exp() + 1e-8

    def decoder_params(self, s):
        """ Decode s to get means and variances in the output space. """
        decoded_means = self.decoder(s)
        return decoded_means, self.decoder_var

    def prior_params(self, u):
        """ Returns the parameters of the prior, which are dependent on u. """
        log_var = self.prior_log_var(u)
        mean = self.prior_mean(u)
        return mean, log_var.exp() + 1e-8

    def forward(self, x, u):
        """ Forward pass through the network, similar to vanilla VAEs. """
        prior_params = self.prior_params(u)
        encoder_params = self.encoder_params(x, u)
        z = self.encoder_dist.sample(*encoder_params)
        decoder_params = self.decoder_params(z)

        return decoder_params, encoder_params, z, prior_params

    def elbo(self, x, u):
        """ Compute ELBO given x and u. """
        decoder_params, encoder_params, z, prior_params = self.forward(x, u)
        log_px_z = self.decoder_dist.log_pdf(x, *decoder_params) # log p(x|z)
        log_qz_xu = self.encoder_dist.log_pdf(z, *encoder_params) # log q(z|x,u)
        log_pz_u = self.prior_dist.log_pdf(z, *prior_params) # log p(z|u)

        # Results of the log_pdf can be verified (when using Normal()) by comparing it with PyTorch methods:
        # All three methods provide the same result, however the log_pdf implemented here (from iVAE code) is faster
        # log_px_z = dist.multivariate_normal.MultivariateNormal(decoder_params[0], torch.diag_embed(decoder_params[1])).log_prob(x)
        # log_qz_xu = dist.multivariate_normal.MultivariateNormal(encoder_params[0], torch.diag_embed(encoder_params[1])).log_prob(z)
        # log_pz_u = dist.multivariate_normal.MultivariateNormal(prior_params[0], torch.diag_embed(prior_params[1])).log_prob(z)

        # log_px_z = dist.independent.Independent(dist.normal.Normal(decoder_params[0], decoder_params[1].sqrt()), 0).log_prob(x).sum(-1)
        # log_qz_xu = dist.independent.Independent(dist.normal.Normal(encoder_params[0], encoder_params[1].sqrt()), 0).log_prob(z).sum(-1)
        # log_pz_u = dist.independent.Independent(dist.normal.Normal(prior_params[0], prior_params[1].sqrt()), 0).log_prob(z).sum(-1)

        return (log_px_z + log_pz_u - log_qz_xu).mean(), z
