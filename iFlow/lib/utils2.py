import json
from os import listdir
from lib.data import SyntheticDataset
from lib.models import iVAE
from lib.iFlow import *

def model_and_data_from_log(experiment_dir):
    # Read in metadata
    logdir = experiment_dir + '/log'
    with open(logdir + '/log.json') as f:
        metadata = json.load(f)['metadata']
    metadata.update({"device": 'cpu'})

    # Create dataset
    dset = SyntheticDataset(metadata['file'], 'cpu')

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

    # Load in last checkpoint
    last_ckpt = listdir(experiment_dir + '/ckpt/1')[-1]
    ckpt = torch.load(experiment_dir + '/ckpt/1/' + last_ckpt)
    model.load_state_dict(ckpt['model_state_dict'])

    return model, dset
