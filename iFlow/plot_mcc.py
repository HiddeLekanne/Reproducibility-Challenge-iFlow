import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import torch

from lib.metrics import mean_corr_coef as mcc
from lib.metrics import * 
from lib.models import iVAE
from lib.iFlow import iFlow
from lib.utils2 import model_and_data_from_log

class Experiment_folder:

    def __init__(self, main_dir, device):
        experiment_paths = set([f.path for f in os.scandir(main_dir) if f.is_dir()])
        self.experiments, self.path2seed, self.seed2path = {}, {}, {}
        self.name = os.path.basename(main_dir).split("_")[0]
        self.device = device
        for path in experiment_paths:
            with open(path + '/log/log.json') as f:
                metadata = json.load(f)['metadata']
                self.experiments[path] = metadata
                self.path2seed[path] = int(metadata["file"].split("_")[6])
                self.seed2path[int(metadata["file"].split("_")[6])] = path
        self.get_final_attribute("final_performance")
        self.get_final_attribute("final_loss")

         
    def get_ranked_list(self, attribute):
        if attribute == "final_performance" or attribute == "final_loss":
            return sorted(self.experiments.items(), key=lambda experiment: int(experiment[1][attribute]))
        elif attribute == "seed":
            return sorted(self.experiments.items(), key=lambda experiment: int(experiment[1]["file"].split("_")[6]))
        else :
            raise NotImplementedError("the attribute: " + str(attribute) +  " has not been implemented.")
    
    def get_final_attribute(self, attribute):
        if not (attribute == "final_performance" or attribute == "final_loss"):
            raise NotImplementedError("the attribute: " + str(attribute) +  " has not been implemented.")
        for experiment_path in self.experiments.keys():
            with open(experiment_path + '/log/log.json') as f:
                json_file = json.load(f)

            # if already computed
            if json_file["metadata"].get(attribute):
                if not self.experiments[experiment_path].get(attribute):
                    self.experiments[experiment_path][attribute] = json_file["metadata"][attribute]
                continue

            try:
                model, dset, _ = model_and_data_from_log(experiment_path, self.device)
            except json.decoder.JSONDecodeError:
                print(experiment_path, "couldn't be loaded")
                continue

            model.eval()
            if attribute == "final_performance":
                attribute_value = calculate_mcc(dset, model)
            elif attribute == "final_loss":    
                x = dset.x
                u = dset.u
                attribute_value = float(model_loss(x, u, model).detach().numpy())

            with open(experiment_path + '/log/log.json', "r+") as f:
                json_file = json.load(f)
                json_file["metadata"][attribute] = attribute_value
                f.seek(0)
                json.dump(json_file, f)

    def get_model_from_experiment(self, experiment):
        path = self.seed2path[int(experiment[1]["file"].split("_")[6])]
        return self.get_model_from_path(path)

    def get_model_from_path(self, experiment_path):
        try:
            model, dset, _ = model_and_data_from_log(experiment_path, self.device)
        except json.decoder.JSONDecodeError:
            print(experiment_path, "couldn't be loaded")
        return model, dset

    def __len__(self):
        return len(self.experiments)


def model_loss(x, u, model):
    if isinstance(model, iVAE):
        loss, _ = model.elbo(x, u)
        loss = loss.mul(-1)
    else:
        (log_normalizer, neg_trace, neg_log_det), z_est = model.neg_log_likelihood(x, u)
        loss = log_normalizer + neg_trace + neg_log_det
    return loss


def model_predict(x, u, model):
    if isinstance(model, iVAE):
        _, z_est = model.elbo(x, u)
    else:
        import operator
        from functools import reduce
        total_num_examples = x.shape[0]
        model.set_mask(total_num_examples)
        z_est, loss = model.inference(x, u)

    z_est = z_est.detach().numpy()
    return z_est


def calculate_mcc(dset, model):
    x = dset.x
    u = dset.u
    s = dset.s.detach().numpy()
    z_est = model_predict(x, u, model)
    return mcc(z_est, s)


def plot_attribute(experiment_folders, attribute, names=[]):
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    YS = []
    for experiment_folder in experiment_folders:
        Y = []
        for path, experiment in experiment_folder.get_ranked_list("seed"):
            if attribute == "final_loss":
                Y.append(-experiment[attribute])
            else:
                Y.append(experiment[attribute])
        YS.append(Y)
        X = list(range(1, len(Y) + 1))

        ax.plot(X, Y, label="{}: {mean:.3f} ({std:.3f})".format(experiment_folder.name, mean=np.mean(Y), std=np.std(Y)))
    print(np.corrcoef(YS[0], YS[1], rowvar=False))
    ax.legend()
    return ax

def align_dimensions(s, z):
    d = z.shape[1]
    cc = np.corrcoef(s, z, rowvar=False)[:d, d:]
    pairs = linear_sum_assignment(-1 * abs(cc))
    z[:, pairs[0]] = z[:, pairs[1]]
    return z

def create_2D_performance_sub_plot(x, labels, ax, cmap, title="", mcc=None):
    if mcc:
        title += f' (MCC: {mcc:.2f})'
    
    ax.scatter(x[:, 0], x[:, 1], c=torch.argmax(labels, dim=1), cmap=cmap, alpha=0.9, s=3)
    ax.set_title(label=title)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_latent_correlation(dset, model, n_samples, sample_offset):
    # Obtain source (s) and approximation (z_est)
    x = dset.x
    u = dset.u
    s = dset.s.detach().numpy()
    z_est = model_predict(x, u, model)

    # Find corresponding dimensions between s and z_dim
    d = x.shape[1]
    cc = np.corrcoef(s, z_est, rowvar=False)[:d, d:]
    pairs = linear_sum_assignment(-1 * abs(cc))

    # Create plots of source and approximation
    fig, ax = plt.subplots(1, z_est.shape[1], figsize = (2*d, 2.5), constrained_layout=True)
    for i in range(z_est.shape[1]):
        p1, p2 = pairs[0][i], pairs[1][i]
        corr = cc[p1, p2].round(4)
        ax[i].set_title('corr: ' + str(abs(corr)))
        ax[i].set_ylim([-3, 3])

        # Sample
        s_sampled = s[sample_offset:sample_offset+n_samples, p1]
        z_sampled = z_est[sample_offset:sample_offset+n_samples, p2]

        # Normalize, this way the scale and mean invariance of the metric is represented
        s_scaled_sample = (s_sampled - np.mean(s_sampled)) / np.std(s_sampled)
        z_scaled_sample = (z_sampled - np.mean(z_sampled)) / np.std(z_sampled)

        ax[i].plot(s_scaled_sample , linestyle = 'dashed')
        if corr < 0:
            ax[i].plot(-1 * z_scaled_sample)
        else:
            ax[i].plot(z_scaled_sample)

    if isinstance(model, iVAE):
        fig.suptitle('iVAE')
    else:
        fig.suptitle('iFlow')
    plt.show()


def create_2D_performance_plot(dset, model_iVAE, model_iFlow):
    # Assert correct model type to assure same layout
    assert isinstance(model_iVAE, iVAE)
    assert isinstance(model_iFlow, iFlow)

    # Unpack data set
    x = dset.x
    u = dset.u
    s = dset.s.detach().numpy()

    # Obtain approximations
    z_est_iVAE = model_predict(x, u, model_iVAE)
    z_est_iFlow = model_predict(x, u, model_iFlow)

    # Find corresponding dimensions between s with iVAE/iFlow estimations
    z_est_iVAE = align_dimensions(s, z_est_iVAE)
    z_est_iFlow = align_dimensions(s, z_est_iFlow)

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

# if __name__ == '__main__':

#     parser = argparse.ArgumentParser()
#     parser.add_argument('experiment_dir', metavar='experiment_dir', type=str, help='experiment directory')
#     parser.add_argument('--device', metavar='device', default="cpu", type=str, help='device, either cpu or cuda')
#     parser.add_argument('--n_samples', metavar='n_samples', default=50, type=int, help="amount of samples to take")
#     parser.add_argument('--sample_offset', metavar='sample_offset', default=0, type=int, help="start position of displayed samples")

#     args = parser.parse_args()
#     model, dset, _ = model_and_data_from_log(args.dir, args.device)
#     model.eval()
#     plot_latent_correlation(dset, model, args.n_samples, args.sample_offset)

    # model_iFlow, dset_iFlow = model_and_data_from_log(args.dir)
    # model_iVAE, dset_iVAE = model_and_data_from_log(args.dir2)

    # if args.seed:
    #     print(f"Creating new data set with seed {args.seed}")
    #     arg_str = f"1000_5_2_2_3_{args.seed}_gauss_xtanh_u_f"
    #     path_to_dset = create_if_not_exist_dataset(arg_str=arg_str)
    #     dset = SyntheticDataset(path_to_dset)
    # else:
    #     print("Using data set on which models were trained")
    #     dset = dset_iFlow

    # create_2D_performance_plot(dset, model_iVAE, model_iFlow)


# if __name__ == '__main__':

#     parser = argparse.ArgumentParser()
#     group = parser.add_mutually_exclusive_group(required=True)
#     group.add_argument('--dirs', metavar='dirs', type=str, default=[None],  nargs="*",
#                     help='all experiments directories')
#     parser.add_argument('--device', metavar='device', default="cpu", type=str, help='device, either cpu or cuda')

#     args = parser.parse_args()
#     args = vars(args.Namespace())

#     plot_experiments(args)

