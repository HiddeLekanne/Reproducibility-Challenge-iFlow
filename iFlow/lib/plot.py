from data import *
import matplotlib.pyplot as plt
import torch

def create_sub_plot(x, labels, ax, title="", mcc=None):
    if mcc:
        title += f' (MCC: {mcc:.2f})'

    N = len(labels[0])
    # define the colormap
    cmap = plt.cm.jet
    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # create the new map
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
    
    ax.scatter(x[:, 0], x[:, 1], c=torch.argmax(labels, dim=1), cmap=cmap, alpha=0.9, s=5)
    ax.set_title(label=title)
    ax.set_xticks([])
    ax.set_yticks([])

def create_plot(s, x, iVAE, iFlow, u):
    fig, axs = plt.subplots(1, 4)

    create_sub_plot(s, u, ax=axs[0], title="Original sources")
    create_sub_plot(x, u, ax=axs[1], title="Observations")
    create_sub_plot(iFlow, u, ax=axs[2], title="Observations", mcc=0.91234)
    create_sub_plot(iVAE, u, ax=axs[3], title="Observations", mcc=0.91234)
    plt.show()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='generate artificial data')
    parser.add_argument('nps', type=int, nargs='?', default=2000,
                        help='number of data points per segment')
    parser.add_argument('ns', type=int, nargs='?', default=40, help='number of segments')
    parser.add_argument('dl', type=int, nargs='?', default=2,
                        help='dimension of the latent sources')
    parser.add_argument('dd', type=int, nargs='?', default=2,
                        help='dimension of the data')
    parser.add_argument('-l', '--n-layers', type=int, default=3, dest='nl',
                        help='number of layers in generating MLP - default: 3    ')
    parser.add_argument('-s', '--seed', type=int, default=1, dest='s',
                        help='random seed of generating MLP - default: 1')
    parser.add_argument('-p', '--prior', default='gauss', dest='p',
                        help='data distribution of each independent source - default: `gauss`')
    parser.add_argument('-a', '--activation', default='xtanh', dest='a',
                        help='activation function of the generating MLP - default: `xtanh`')
    parser.add_argument('-u', '--uncentered', action='store_true', default=False,
                        help='Generate uncentered data - default False')
    parser.add_argument('-n', '--noisy', action='store_true', default=False,
                        help='Generate noisy data - default False')
    args = parser.parse_args()
    if args.dd is None:
        args.dd = 4 * args.dl

    kwargs = {"nps": args.nps, "ns": args.ns, "dl": args.dl, "dd": args.dd, "nl": args.nl,
                "p": args.p, "a": args.a, "s": args.s, "uncentered": args.uncentered,
                "noisy": args.noisy}

    path_to_dataset = create_if_not_exist_dataset(**kwargs)

    data = SyntheticDataset(path_to_dataset)

    create_plot(data.s, data.x, data.x, data.x, data.u)
