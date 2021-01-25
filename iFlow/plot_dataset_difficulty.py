import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import os

from lib.metrics import mean_corr_coef as mcc
from lib.metrics import *
from lib.models import iVAE
import json

from lib.utils2 import model_and_data_from_log

# calculate the kl divergence
def kl_divergence_gaussian(meansA, cov_matrixA, meansB, cov_matrixB):
    return 1 / 2 * (np.trace(np.linalg.inv(cov_matrixB) @ cov_matrixA) + \
          (meansB - meansA).T @ np.linalg.inv(cov_matrixB) @ (meansB - meansA) - \
           len(meansA) + np.log((np.linalg.det(cov_matrixB) / np.linalg.det(cov_matrixA))))

def calculate_guassian_parameters(dset):
    means, stds = [], []
    for seg in range(dset.u.shape[1]):
        data = dset.s[1 == dset.u[:, seg], :].numpy()
        means.append(np.mean(data, axis=0))
        stds.append(np.std(data, axis=0))
    return [np.array(means), np.array(stds)]

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
    parser.add_argument('--save_file', default='dd.npy', type=str, help="file where the results are saved")
    parser.add_argument('--load_file', default=None, type=str, help="file from which to load the results instead of recalculating")
    parser.add_argument('--recalulate_KL', default=True, type=bool, help="recalculate KL for prototyping different measures" )

    args = parser.parse_args()

    if not (args.dir or args.load_file):
        raise ValueError("you need to either specify a experiments dir or a results file")


    if not args.load_file:
        experiments = [f.path for f in os.scandir(args.dir) if f.is_dir() ]
        X, Y, KL = [], [], []
        for experiment in experiments:
            try:
                model, dset, metadata = model_and_data_from_log(experiment, args.device)
            except json.decoder.JSONDecodeError:
                print(experiment, "couldn't be loaded")
                continue

            means, stds = calculate_guassian_parameters(dset)

            scores = []
            for seg in range(dset.u.shape[1]):
                for seg_2 in range(seg + 1, dset.u.shape[1]):
                    divergence = kl_divergence_gaussian(means[seg], np.diag(stds[seg]), means[seg_2], np.diag(stds[seg_2]))
                    scores.append(divergence)
            score = np.min(scores) # / (dset.u.shape[1]**2 /2) # / dset.s.shape[1]**2

            model.eval()
            Y.append(calculate_mcc(dset, model, args.n_samples))
            X.append(int(metadata["file"].split("_")[6]))   
        with open(args.save_file, 'wb') as f:
                np.save(f, [X, Y, KL])

    else:
        with open(args.load_file, 'rb') as f:
            X, Y, KL = np.load(f)
            experiments = [f.path for f in os.scandir(args.dir) if f.is_dir() ]
            if args.recalulate_KL:
                X_2 = []
                KL = []
                for experiment in experiments:
                    try:
                        model, dset, metadata = model_and_data_from_log(experiment, args.device)
                    except json.decoder.JSONDecodeError:
                        print(experiment, "couldn't be loaded")
                        continue

                    means, stds = calculate_guassian_parameters(dset)

                    scores = []
                    for seg in range(dset.u.shape[1]):
                        for seg_2 in range(seg + 1, dset.u.shape[1]):
                            divergence = kl_divergence_gaussian(means[seg], np.diag(stds[seg]), means[seg_2], np.diag(stds[seg_2]))
                            scores.append(divergence)
                    score = np.min(scores) # / (dset.u.shape[1]**2 /2) # / dset.s.shape[1]**2
                    
                    KL.append(score)
                    X_2.append(int(metadata["file"].split("_")[6]))
                    # break

    Y_2 = [0] * 100
    for i, y in enumerate(Y):
        Y_2[X_2.index(X[i])] =  y
    
    Y_2 = np.abs(Y_2)
    print(np.corrcoef(KL, Y_2))

    Y_2 = [y for _,y in sorted(zip(KL,Y_2))]
    # KL = [kl for _,kl in sorted(zip(X,KL))]
    KL = sorted(KL)

    print(np.mean(Y), np.std(Y))
    plt.plot(Y_2)
    plt.plot(KL)
    plt.show()