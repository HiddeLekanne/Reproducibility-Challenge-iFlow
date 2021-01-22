import json
from os import listdir

from lib.data import SyntheticDataset
from lib.models import iVAE
from lib.iFlow import *

def model_and_data_from_log(experiment_dir, device):
    # Read in metadata
    logdir = experiment_dir + '/log'
    with open(logdir + '/log.json') as f:
        metadata = json.load(f)['metadata']
    metadata.update({"device": device})

    # Create dataset
    if ("scratch/" in metadata['file']):
        dset = SyntheticDataset("data/" + metadata["file"].split("_")[6], device)
    else:
        dset = SyntheticDataset(metadata['file'], device)

    # Create model
    if metadata['i_what'] == 'iFlow':
        model = iFlow(args=metadata)
    elif metadata['i_what'] == 'iVAE':
        model = iVAE(dset.latent_dim, \
            dset.data_dim, \
            dset.aux_dim, \
            n_layers=int(metadata['depth']), \
            activation='lrelu', \
            device='cpu', \
            hidden_dim=int(metadata['hidden_dim']), \
            anneal=False) # False

    # Find last checkpoint
    ckpts = listdir(experiment_dir + '/ckpt/1')
    ckpt_ints = [int(c.strip('.pth').split('_')[-1]) for c in ckpts]
    last_ckpt = ckpts[np.argmax(ckpt_ints)]
    # Load in checkpoint
    ckpt = torch.load(experiment_dir + '/ckpt/1/' + last_ckpt, map_location=torch.device(device))
    model.load_state_dict(ckpt['model_state_dict'])

    return model, dset, metadata
