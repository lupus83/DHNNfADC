#in order to run on specific gpu, run:
#CUDA_VISIBLE_DEVICES=0,1 python train.py --gpu 0
#training script used to check connection in ablation study aux network and size of bottleneck dims

import logging
from pathlib import Path
import datetime
import pdb
import yaml
from shutil import copyfile

from models import ModelFactory
from utils import AdniVolumeDataset, load_heterogeneous_data, split_data, parse_args, get_seeds_to_net_dict
from dl_utils import train_and_evaluate

import torch
from torch.utils.data import DataLoader

SEPARATOR = "_"

LOG = logging.getLogger(__name__)

def main():

    ############################
    # Parse arguments
    ############################
    args = parse_args()

    config = {}
    with open("config.yaml", 'r') as stream:
        try:
            config = yaml.safe_load(stream)  # safe_load should be preferred
        except yaml.YAMLError as exc:
            print(exc)

    LOG.debug(f'GPU: {torch.cuda.current_device()}')

    net = config['model_type']
    assert net in ModelFactory().get_available_models()
    heterogeneous_exp = True if (net in ModelFactory().get_heterogeneous_models()) else False

    DEBUG = False
    if (args.debug):
        DEBUG = True
        exp_name = 'debug'
    else:
        exp_name = config['exp_name']

    exp_path = # PATH RERMOVED FOR PRIVACY
    try:
        exp_path.mkdir()
    except FileExistsError:
        print("")

    copyfile('config.yaml', str(exp_path / ('config' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M") + 'gpu' + str(args.gpu) + '.yaml')))
    logging.basicConfig(filename=str(exp_path / ('log' + str(args.gpu) + '.out')), level=logging.DEBUG)
    LOG.debug(f'##################################################\n{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")}\n##################################################')
    LOG.debug(f'Exp Name: {exp_name}')
    LOG.debug(f'Heterogeneous Net: {heterogeneous_exp}')
    LOG.debug(f'GPU: {str(args.gpu)}')

    # if pretrained model, create dictionary of items
    if config['pretrain'] is not None:
        pretrained_nets = get_seeds_to_net_dict(
            directory=config['pretrain']['dir'],
            model_args=config['pretrain']['ags'],
            seeds=config['seeds'])

    model_args = config['model_args']
    ############################
    # Training routine
    ############################
    for mask in config['masking']:

        in_channels = 2 if mask == 'concat' else 1

        model_args['in_channels'] = in_channels
        # read dataset
        # data is tuple or triplet: data, patients(, vscodes)
        data = load_heterogeneous_data(config['path_to_image_data'], config['path_to_non_image_data'], \
                                        hippocampus=config['hippocampus'], masking=mask, debug=DEBUG, \
                                        visit_codes=config['filter_by_visitcodes'], label_coding=config['label_coding'], \
                                        filter_unreliable_data=config['filter_unreliable_data'])
        
        if config['filter_by_visitcodes']:
            assert len(data) == 3
            vscodes = data[2]
        else:
            assert len(data) == 2
            vscodes = None
        
        non_image_data_ndim = len(data[0][0][1])
        if heterogeneous_exp:
            if 'Dynamic' in config['model_type']:
                model_args['dynamic_conv_args']['ndim_non_img'] = non_image_data_ndim
            elif 'Film' in config['model_type']:
                model_args['filmblock_args']['ndim_non_img'] = non_image_data_ndim
            else:  # ConcatHNN, CheckAux
                model_args['ndim_non_img'] = non_image_data_ndim
        if args.gpu == 0:
            model_args['n_features'] = 32
        elif args.gpu == 1:
            model_args['n_features'] = 64
        else:
            raise ValueError(f'Invalid gpu argument. Must be 0 or 1!')

        for n_hidden_dims in [4, 8, 16]:
            
            model_args['n_hidden_dims'] = n_hidden_dims

            for activation in ['sigmoid', 'tanh', 'linear']:
                model_args['activation'] = activation

                for scale, shift in [(True,True), (True,False), (False,True)]:
                    model_args['scale'] = scale
                    model_args['shift'] = shift

                    # carry out experiment without scaling just for one activation...
                    if scale == False and activation in ['sigmoid', 'tanh']:
                        continue

                    # for dynamic_layer_index in config['all_indices']:

                    #     model_args['indices_dynamic_layers'] = [dynamic_layer_index]
                    for normalize in config['normalize_non_image_data']:

                        for seed in config['seeds']:

                            for test_size in config['test_size']:

                                splits = split_data(data[0], groups=data[1], visit_codes=vscodes, test_size=test_size, seed=seed)
                             
                                for lr in config['learning_rates']:

                                    for weight_decay in config['weight_decay']:

                                        for batch_size in config['batch_size']:

                                            for batch_norm_momentum in config['bn_momentum']:

                                                # create dataset
                                                train_dataset = AdniVolumeDataset(splits["train"], input_dim=config['dim'], masking=mask, augment=config['augment_train_set'], heterog_exp=heterogeneous_exp, normalize_non_image_data=normalize)
                                                eval_dataset = AdniVolumeDataset(splits["valid"], input_dim=config['dim'], masking=mask, augment=config['augment_val_set'], heterog_exp=heterogeneous_exp, normalize_non_image_data=normalize)
                                                
                                                # create dataloader
                                                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=config['shuffle_train_set'], drop_last=True, num_workers=12)
                                                eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=config['shuffle_val_set'], num_workers=12)
                                                
                                                # get pretrained model
                                                pretrained = None
                                                if config['pretrain'] != None:
                                                    pretrained = pretrained_nets[seed]

                                                # create model
                                                model_args['bn_momentum'] = batch_norm_momentum
                                                model = ModelFactory().create_model(net, model_args, pretrained=pretrained, block_grads=config['block_grads'])
                                                
                                                model_dir = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M') \
                                                    + f"_VolMask_{mask}" \
                                                    + f"_Seed_{seed}" \
                                                    + f"_lr_{lr}" \
                                                    + f"_WD_{weight_decay}" \
                                                    + f"_BN_{batch_norm_momentum}" \
                                                    + f"_BS_{batch_size}" \
                                                    + f"_SplitSize_{test_size}" \
                                                    + ('_fixed' if config["block_grads"] else '') \
                                                    + ('_normalized' if normalize else '') \

                                                if 'Film' in config['model_type']:
                                                    model_dir = model_dir \
                                                    + f'_loc{model_args["filmblock_args"]["location"]}'  \
                                                    + f'_activation_{model_args["filmblock_args"]["activation"]}' \
                                                    + ('_scaled' if model_args["filmblock_args"]["scale"] else '') \
                                                    + ('_shifted' if model_args["filmblock_args"]["shift"] else '')

                                                model_dir = model_dir + '_' + net

                                                model_dir = exp_path / model_dir

                                                LOG.debug(f'----------------------------\n')
                                                LOG.debug(f'Model Dir: {str(model_dir)}')
                                                LOG.debug(f'Params net_creation: {model_args}')
                                                LOG.debug(f'Pretrain model: {pretrained}')
                                                LOG.debug(f'Model: {model}')

                                                hparms = {'fixed': config['block_grads'], 'normalized': normalize}
                                                if 'filmblock_args' in model_args.keys():
                                                    hparms.update(model_args['filmblock_args'])
                                    
                                                params = {
                                                    'model': model,
                                                    'model_dir': model_dir,
                                                    'train_loader': train_loader,
                                                    'eval_loader': eval_loader,
                                                    'n_epochs': config['n_epochs'],
                                                    'lr': lr,
                                                    'optimizer': config['optimizer'],
                                                    'betas': tuple(config['betas']),
                                                    'weight_decay': weight_decay,
                                                    'hnn': heterogeneous_exp,
                                                    'hyperparams': {'normalize': normalize, 'freeze': config['block_grads']},
                                                    'transer_learning_exp': config['block_grads'],
                                                    'hyperparams': hparms
                                                }
                                                # if 'SE' in net:
                                                #     params['reduction_ratio'] = model_args['reduction_ratio']

                                                train_and_evaluate(**params)


if __name__ == "__main__":
    main()

