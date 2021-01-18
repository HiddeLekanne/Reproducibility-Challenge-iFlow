import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from lib.data import *
from lib.iFlow import *
from lib.metrics import mean_corr_coef as mcc
from lib.metrics import *
from lib.models import iVAE

from lib.utils2 import model_and_data_from_log

def model_estimations(x, u, model):
    if isinstance(model, iVAE):
        _, z_est = model.elbo(x, u)
    else:
        import operator
        from functools import reduce
        total_num_examples = x.shape[0]
        model.set_mask(total_num_examples)
        z_est, _ = model.inference(x, u)

    z_est = z_est.detach().numpy()
    return z_est

def permutate_corresponding_dimensions(s, z):
    d = z.shape[1]
    cc = np.corrcoef(s, z, rowvar=False)[:d, d:]
    pairs = linear_sum_assignment(-1 * abs(cc))
    z[:, pairs[0]] = z[:, pairs[1]]
    return z

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

def create_2D_performance_sub_plot(x, labels, ax, cmap, title="", mcc=None):
    if mcc:
        title += f' (MCC: {mcc:.2f})'
    
    ax.scatter(x[:, 0], x[:, 1], c=torch.argmax(labels, dim=1), cmap=cmap, alpha=0.9, s=3)
    ax.set_title(label=title)
    ax.set_xticks([])
    ax.set_yticks([])

def create_2D_performance_plot(dset, model_iVAE, model_iFlow):
    # Assert correct model type to assure same layout
    assert isinstance(model_iVAE, iVAE)
    assert isinstance(model_iFlow, iFlow)

    # Unpack data set
    x = dset.x
    u = dset.u
    s = dset.s.detach().numpy()

    # Obtain approximations
    z_est_iVAE = model_estimations(x, u, model_iVAE)
    z_est_iFlow = model_estimations(x, u, model_iFlow)

    # Find corresponding dimensions between s with iVAE/iFlow estimations
    z_est_iVAE = permutate_corresponding_dimensions(s, z_est_iVAE)
    z_est_iFlow = permutate_corresponding_dimensions(s, z_est_iFlow)

    # Define colormap
    N = len(u[0])
    cmap = plt.cm.jet
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)

    # Create subplots for source, observation and iFlow and iVAE estimations
    fig, axs = plt.subplots(1, 4)
    create_2D_performance_sub_plot(s, u, ax=axs[0], cmap=cmap, title="Original sources")
    create_2D_performance_sub_plot(x, u, ax=axs[1], cmap=cmap, title="Observations")
    create_2D_performance_sub_plot(z_est_iFlow, u, ax=axs[2], cmap=cmap, title="iFlow", mcc=mcc(s, z_est_iFlow))
    create_2D_performance_sub_plot(z_est_iVAE, u, ax=axs[3], cmap=cmap, title="iVAE", mcc=mcc(s, z_est_iVAE))
    plt.show()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('dir', metavar='dir', type=str,
                    help='experiment directory')
    parser.add_argument('dir2', metavar='dir2', type=str,
                    help='other experiment directory')
    parser.add_argument('-s', '--seed', type=int, default=None, 
                    help='random seed (default 1)')

    args = parser.parse_args()

    model, dset = model_and_data_from_log(args.dir)
    plot_latent_correlation(dset, model)

    model_iFlow, dset_iFlow = model_and_data_from_log(args.dir)
    model_iVAE, dset_iVAE = model_and_data_from_log(args.dir2)

    if args.seed:
        print(f"Creating new data set with seed {args.seed}")
        arg_str = f"1000_5_2_2_3_{args.seed}_gauss_xtanh_u_f"
        path_to_dset = create_if_not_exist_dataset(arg_str=arg_str)
        dset = SyntheticDataset(path_to_dset)
    else:
        print("Using data set on which models were trained")
        dset = dset_iFlow

    create_2D_performance_plot(dset, model_iVAE, model_iFlow)
