'''
Evaluation tools.
Created by Basile Van Hoorick for TCOW.
'''

import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), 'eval/'))
sys.path.insert(0, os.getcwd())

from __init__ import *

# Internal imports.
import my_utils
import seeker
import pipeline


def load_networks(checkpoint_path, device, logger, epoch=-1):
    '''
    :param checkpoint_path (str): Path to model checkpoint folder or file.
    :param epoch (int): If >= 0, desired checkpoint epoch to load.
    :return (networks, train_args, dset_args, model_args, epoch).
        networks (dict).
        train_args (dict).
        train_dset_args (dict).
        model_args (dict).
        epoch (int).
    '''
    print_fn = logger.info if logger is not None else print

    assert os.path.exists(checkpoint_path)
    if os.path.isdir(checkpoint_path):
        model_fn = f'model_{epoch}.pth' if epoch >= 0 else 'checkpoint.pth'
        checkpoint_path = os.path.join(checkpoint_path, model_fn)

    print_fn('Loading weights from: ' + checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Load all arguments for later use.
    train_args = checkpoint['train_args']
    train_dset_args = checkpoint['dset_args']

    # Get network instance parameters.
    seeker_args = checkpoint['seeker_args']

    model_args = {'seeker': seeker_args}

    # Instantiate networks.
    seeker_net = seeker.Seeker(logger, **seeker_args)
    seeker_net = seeker_net.to(device)
    seeker_net.load_state_dict(checkpoint['net_seeker'])
    networks = {'seeker': seeker_net}
    epoch = checkpoint['epoch']
    print_fn('=> Loaded epoch (1-based): ' + str(epoch + 1))

    return (networks, train_args, train_dset_args, model_args, epoch)


def perform_inference(data_retval, networks, device, logger, all_args, cur_step):
                    #   **pipeline_args):
    '''
    Generates test time predictions.
    :param data_retval (dict): Data loader element.
    :param all_args (dict): train, test, train_dset, test_dset, model.
    '''
    # Following DRY, prepare pipeline instance, *BUT* take care of shared args by updating them.
    used_args = copy.deepcopy(all_args['train'])
    used_args.num_queries = all_args['test'].num_queries

    my_pipeline = pipeline.MyTrainPipeline(used_args, logger, networks, device)
    my_pipeline.set_phase('test')  # This calls eval() on all submodules.

    include_loss = True
    metrics_only = (data_retval['source_name'][0] == 'plugin')

    temp_st = time.time()
    (model_retval, loss_retval) = my_pipeline(
        data_retval, cur_step, cur_step, 0, 1.0, include_loss, metrics_only)
    logger.debug(f'(perform_inference) my_pipeline: {time.time() - temp_st:.3f}s')

    # Calculate various evaluation metrics.
    loss_retval = my_pipeline.process_entire_batch(
        data_retval, model_retval, loss_retval, cur_step, cur_step, 0, 1.0) \
            if loss_retval is not None else None

    # Organize and return relevant info, moving stuff to CPU and/or converting to numpy as needed.
    inference_retval = dict()
    inference_retval['model_retval'] = model_retval
    inference_retval['loss_retval'] = loss_retval
    inference_retval = my_utils.dict_to_cpu(inference_retval)

    return inference_retval
