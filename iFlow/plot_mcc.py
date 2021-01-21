import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import torch

from lib.metrics import mean_corr_coef as mcc
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

def plot_experiments(args):
    if not (all(args["dirs"]) or all(args["load_files"])):
        raise ValueError("you need to either specify all experiment dirs or all result files")

    if any(args["dirs"]) and any(args["load_files"]):
        raise ValueError("you need to either all experiment dirs or all saved results files, both are specified")

    length = max(len(args["dirs"]), len(args["load_files"]))

    for i in range(length):
        if not all(args["load_files"]):
            experiments = [f.path for f in os.scandir(args["dirs"][i]) if f.is_dir() ]
            X, Y = [], []
            for experiment in experiments:
                try:
                    model, dset, metadata = model_and_data_from_log(experiment, args["device"])
                except json.decoder.JSONDecodeError:
                    print(experiment, "couldn't be loaded")
                    continue

                model.eval()
                Y.append(calculate_mcc(dset, model, args["n_samples"]))
                X.append(int(metadata["file"].split("_")[7]))   
            if args.get("save_files"):
                with open(args["save_files"][i], 'wb') as f:
                        np.save(f, [X, Y])

            print(X)
            Y = [y for _,y in sorted(zip(X,Y))]
            X = sorted(X)
            plt.plot(X, Y)
        else:
            with open(args["load_files"][i], 'rb') as f:
                X, Y = np.load(f)
        
            print(X)
            Y = [y for _,y in sorted(zip(X,Y))]
            X = sorted(X)
            plt.plot(X, Y)
    plt.show()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--dirs', metavar='dirs', type=str, default=[None],  nargs="*",
                    help='all experiments directories')
    group.add_argument('--load_files', default=[None], type=str, help="files from which to load the results instead of recalculating", nargs="*")

    parser.add_argument('--device', metavar='device', default="cpu", type=str, help='device, either cpu or cuda')
    parser.add_argument('--n_samples', metavar='n_samples', default=50, type=int, help="amount of samples to take")
    parser.add_argument('--save_file', default=[], type=str, help="file where the results are saved")

    args = parser.parse_args()
    args = vars(args.Namespace())

    plot_experiments(args)

