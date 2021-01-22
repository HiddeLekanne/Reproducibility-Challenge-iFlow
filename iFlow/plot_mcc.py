import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import torch

from lib.metrics import mean_corr_coef as mcc
from lib.models import iVAE
from lib.utils2 import model_and_data_from_log

class Experiment_folder:

    def __init__(self, main_dir, device):
        experiment_paths = set([f.path for f in os.scandir(main_dir) if f.is_dir()])
        self.experiments, self.path2seed, self.seed2path = {}, {}, {}
        self.device = device
        for path in experiment_paths:
            with open(path + '/log/log.json') as f:
                metadata = json.load(f)['metadata']
                self.experiments[path] = metadata
                self.path2seed[path] = int(metadata["file"].split("_")[6])
                self.seed2path[int(metadata["file"].split("_")[6])] = path
        self.get_final_performance()

         
    def get_ranked_list(self, attribute):
        if attribute == "final_performance":
            return sorted(self.experiments.values(), key=lambda experiment: int(experiment[attribute]))
        elif attribute == "seed":
            return sorted(self.experiments.values(), key=lambda experiment: int(experiment["file"].split("_")[6]))
        else:
            raise NotImplementedError("the attribute: " + str(attribute) +  " has not been implemented.")
    
    def get_final_performance(self):
        for experiment in self.experiments.keys():
            with open(experiment + '/log/log.json') as f:
                json_file = json.load(f)

            # if already computed
            if json_file["metadata"].get("final_performance"):
                if not self.experiments[experiment].get("final_performance"):
                    self.experiments[experiment]["final_performance"] = json_file["metadata"]["final_performance"]
                continue

            try:
                model, dset, _ = model_and_data_from_log(experiment, self.device)
            except json.decoder.JSONDecodeError:
                print(experiment, "couldn't be loaded")
                continue

            model.eval()
            final_performance = calculate_mcc(dset, model)

            with open(experiment + '/log/log.json', "r+") as f:
                json_file = json.load(f)
                json_file["metadata"]["final_performance"] = final_performance
                f.seek(0)
                json.dump(json_file, f)

    def __len___(self):
        return len(self.experiments)



def model_predict(x, u, model):
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


def calculate_mcc(dset, model):
    ""
    x = dset.x
    u = dset.u
    s = dset.s.detach().numpy()
    z_est = model_predict(x, u, model)
    return mcc(z_est, s)


def plot_mcc(experiment_folders):
    for experiment_folder in experiment_folders:
        Y = []
        for experiment in experiment_folder.get_ranked_list("seed"):
            Y.append(experiment["final_performance"])
        X = list(range(1, len(Y) + 1))
        plt.plot(X, Y)    
    plt.show()


def align_dimensions(s, z):
    d = z.shape[1]
    cc = np.corrcoef(s, z, rowvar=False)[:d, d:]
    pairs = linear_sum_assignment(-1 * abs(cc))
    z[:, pairs[0]] = z[:, pairs[1]]
    return z

# if __name__ == '__main__':

#     parser = argparse.ArgumentParser()
#     group = parser.add_mutually_exclusive_group(required=True)
#     group.add_argument('--dirs', metavar='dirs', type=str, default=[None],  nargs="*",
#                     help='all experiments directories')
#     parser.add_argument('--device', metavar='device', default="cpu", type=str, help='device, either cpu or cuda')

#     args = parser.parse_args()
#     args = vars(args.Namespace())

#     plot_experiments(args)

