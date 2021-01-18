import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import os

from lib.metrics import mean_corr_coef as mcc
from lib.metrics import *
from lib.models import iVAE

from lib.utils2 import model_and_data_from_log

def calculate_mcc(dset, model, n_samples):
    ""
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
    return mcc(z_est, s)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', metavar='dir', type=str, default=None,
                    help='all experiments directory')
    parser.add_argument('--device', metavar='device', default="cpu", type=str, help='device, either cpu or cuda')
    parser.add_argument('--n_samples', metavar='n_samples', default=50, type=int, help="amount of samples to take")
    parser.add_argument('--save_file', default='mcc.npy', type=str, help="file where the results are saved")
    parser.add_argument('--load_file', default=None, type=str, help="file from which to load the results instead of recalculating")

    args = parser.parse_args()

    if not (args.dir or args.load_file):
        raise ValueError("you need to either specify a experiments dir or a results file")


    if not args.load_file:
        experiments = [f.path for f in os.scandir(args.dir) if f.is_dir() ]
        X, Y = [], []
        for experiment in experiments:
            model, dset, metadata = model_and_data_from_log(experiment, args.device)
            model.eval()
            Y.append(calculate_mcc(dset, model, args.n_samples))
            X.append(int(metadata["file"].split("_")[6]))   
        with open(args.save_file, 'wb') as f:
                np.save(f, [X, Y])
    else:
        with open(args.load_file, 'rb') as f:
            X, Y = np.load(f)

    Y = [y for _,y in sorted(zip(X,Y))]
    X = sorted(X)
    print(np.mean(Y), np.std(Y))
    plt.plot(X, Y)
    plt.show()