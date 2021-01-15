import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from lib.metrics import mean_corr_coef as mcc
from lib.metrics import *
from lib.models import iVAE

from lib.utils2 import model_and_data_from_log

def plot_latent_correlation(dset, model):
    # Obtain source (s) and approximation (z_est)
    x = dset.x
    u = dset.u
    s = dset.s.detach().numpy()
    if isinstance(model, iVAE):
        _, z_est = model.elbo(x, u)
    else:
        import operator
        from functools import reduce
        total_num_examples = x.shape[0]
        model.set_mask(total_num_examples)
        z_est, _ = model.inference(x, u)

    z_est = z_est.detach().numpy()

    # Find corresponding dimensions between s and z_dim
    d = x.shape[1]
    cc = np.corrcoef(s, z_est, rowvar=False)[:d, d:]
    pairs = linear_sum_assignment(-1 * abs(cc))

    # Create plots of source and approximation
    fig, ax = plt.subplots(1, z_est.shape[1], figsize = (2.5*d, 2.5))
    for i in range(z_est.shape[1]):
        p1, p2 = pairs[0][i], pairs[1][i]
        corr = cc[p1, p2].round(4)
        ax[i].set_title('corr: ' + str(abs(corr)))
        ax[i].plot(s[:25, p1], linestyle = 'dashed')
        if corr < 0:
            ax[i].plot(-1 * z_est[:25, p2])
        else:
            ax[i].plot(z_est[:25, p2])
    plt.show()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('dir', metavar='dir', type=str,
                    help='experiment directory')

    args = parser.parse_args()
    model, dset = model_and_data_from_log(args.dir)

    plot_latent_correlation(dset, model)
