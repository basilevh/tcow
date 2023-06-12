'''
Evaluation logic.
Created by Basile Van Hoorick for TCOW.
'''

import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), 'eval/'))
sys.path.insert(0, os.getcwd())

from __init__ import *

# Internal imports.
import args
import data
import data_utils
import inference
import logvis
import metrics
import my_utils


def _test_inner(all_args, networks, data_loader, device, logger, step_offset):

    num_steps = len(data_loader)
    start_time = time.time()
    inference_retvals = []

    # for cur_step, data_retval in enumerate(tqdm.tqdm(data_loader)):
    for cur_step, data_retval in enumerate(data_loader):

        real_step = cur_step + step_offset

        if cur_step == 0:
            logger.info(f'Enter first data loader iteration took {time.time() - start_time:.3f}s')

        data_retval['within_batch_idx'] = torch.arange(all_args['test'].batch_size)  # (B).

        # Perform inference (independently per example).
        inference_retval = inference.perform_inference(
            data_retval, networks, device, logger, all_args, real_step)

        # Print and visualize stuff.
        if not(all_args['test'].log_rarely):
            friendly_short_name = logger.handle_test_step(
                real_step, num_steps, data_retval, inference_retval, all_args)
            inference_retval['friendly_short_name'] = friendly_short_name

        # Save all data if desired.
        data_retval_pruned = data_utils.clean_remain_reproducible(data_retval)
        inference_retval['data_retval_pruned'] = data_retval_pruned
        if all_args['test'].store_results:
            logger.save_pickle(
                inference_retval, f'inference_retval_s{real_step}.p', step=real_step)

        # Save some information to be aggregated across the entire test set.
        inference_retval = my_utils.dict_to_cpu(inference_retval)
        inference_retvals.append(inference_retval)

    return inference_retvals


def _test_outer(all_args, networks, device, logger):
    '''
    :param all_args (dict): train, test, train_dset, test_dset, model.
    '''
    outer_start_time = time.time()

    for net in networks.values():
        net.eval()
    torch.set_grad_enabled(False)

    orig_test_args = copy.deepcopy(all_args['test'])

    assert isinstance(all_args['test'].data_path, list)
    actual_data_paths = data_utils.get_data_paths_from_args(all_args['test'].data_path)
    assert isinstance(actual_data_paths, list)

    inference_retvals = []
    step_offset = 0

    logger.info('Starting outer test loop over individual data paths...')
    for outer_step, cur_data_path in enumerate(tqdm.tqdm(actual_data_paths)):

        # Temporarily overwrite value in args object because data.py uses this. We pretend to the
        # callee that only a list of size one was given.
        all_args['test'].data_path = [cur_data_path]

        logger.info('Initializing current data loader...')
        start_time = time.time()

        # Instantiate dataset.
        (cur_test_loader, test_dset_args) = data.create_test_data_loader(
            all_args['train'], all_args['test'], all_args['train_dset'], logger)
        if outer_step == 0:
            logger.info('Final (first) test dataset args: ' + str(test_dset_args))
        all_args['test_dset'] = test_dset_args

        logger.info(f'Took {time.time() - start_time:.3f}s')

        cur_inference_retvals = _test_inner(
            all_args, networks, cur_test_loader, device, logger, step_offset)

        inference_retvals += cur_inference_retvals

        step_offset += len(cur_test_loader)

        del cur_test_loader

    all_args['test'] = orig_test_args  # Restore test_args.

    _test_postprocess(inference_retvals, logger)

    logger.info()
    total_time = time.time() - outer_start_time
    logger.info(f'Total time: {total_time / 3600.0:.3f} hours')

    pass


def _test_postprocess(inference_retvals, logger):

    # Report mean metrics over all examples (consider both per-frame and per-scene).
    if inference_retvals[0]['loss_retval'] is not None:

        metrics_retvals = [x['loss_retval']['metrics'] for x in inference_retvals]

        final_weighted_metrics = metrics.calculate_weighted_averages(metrics_retvals)
        final_unweighted_metrics = metrics.calculate_unweighted_averages(metrics_retvals)
        metrics.pretty_print_aggregated(
            logger, final_weighted_metrics, final_unweighted_metrics, len(metrics_retvals))

        test_results = metrics.test_results_to_dataframe(inference_retvals)

        # Export itemized results to CSV file to allow for easy sorting.
        csv_fp = os.path.join(logger.log_dir, 'itemized_results.csv')
        test_results.to_csv(csv_fp)
        logger.info(f'Exported quantitative results to: {csv_fp}')

        # Sanity check: verify_* should be exactly equal (both keys and values) to final_*_metrics.
        verify_weighted = metrics.calculate_weighted_averages_dataframe(test_results)
        verify_unweighted = metrics.calculate_unweighted_averages_dataframe(test_results)
        for k in verify_weighted.keys():
            if not(np.isnan(verify_weighted[k]) or np.isnan(final_weighted_metrics[k])):
                if not(np.isclose(verify_weighted[k], final_weighted_metrics[k])):
                    logger.error(f'Weighted metric {k} does not match! '
                                 f'{verify_weighted[k]} vs {final_weighted_metrics[k]}')
        for k in verify_unweighted.keys():
            if not(np.isnan(verify_unweighted[k]) or np.isnan(final_unweighted_metrics[k])):
                if not(np.isclose(verify_unweighted[k], final_unweighted_metrics[k])):
                    logger.error(f'Unweighted metric {k} does not match! '
                                 f'{verify_unweighted[k]} vs {final_unweighted_metrics[k]}')

        if len(inference_retvals) >= 20:
            logger.warning()
            logger.warning('Note that the metrics reported here are based on ALL available clips '
                        'sampled from the input video files. If you ran the test script on a Kubric '
                        'or Rubric dataset, we recommend ignoring these numbers and instead running '
                        'pick_represent for a more balanced evaluation (which are also the stats we '
                        'use in our paper). See README for instructions.')
            logger.warning()

    pass


def main(test_args, logger):

    logger.info()
    logger.info('torch version: ' + str(torch.__version__))
    logger.info('torchvision version: ' + str(torchvision.__version__))
    logger.save_args(test_args)

    # NOTE: This current test script is not even dependent on any randomness / seed at all!
    np.random.seed(test_args.seed)
    random.seed(test_args.seed)
    torch.manual_seed(test_args.seed)
    if test_args.device == 'cuda':
        torch.cuda.manual_seed_all(test_args.seed)

    logger.info('Initializing model...')
    start_time = time.time()

    # Instantiate networks and load weights.
    if test_args.device == 'cuda':
        device = torch.device('cuda:' + str(test_args.gpu_id))
    else:
        device = torch.device(test_args.device)
    (networks, train_args, train_dset_args, model_args, epoch) = \
        inference.load_networks(test_args.resume, device, logger, epoch=test_args.epoch)

    logger.info(f'Took {time.time() - start_time:.3f}s')

    if test_args.avoid_wandb < 2:
        logger.init_wandb(PROJECT_NAME, test_args, networks.values(), name=test_args.name,
                        group=test_args.wandb_group)

    # Print test arguments.
    logger.info('Train command args: ' + str(train_args))
    logger.info('Train dataset args: ' + str(train_dset_args))
    logger.info('Final test command args: ' + str(test_args))

    # Combine arguments for later use.
    all_args = dict()
    all_args['train'] = train_args
    all_args['test'] = test_args
    all_args['train_dset'] = train_dset_args
    all_args['model'] = model_args

    # Run actual test loop.
    _test_outer(all_args, networks, device, logger)


if __name__ == '__main__':

    # WARNING: This is slow, but we can detect NaNs this way:
    # torch.autograd.set_detect_anomaly(True)

    np.set_printoptions(precision=3, suppress=True)
    torch.set_printoptions(precision=3, sci_mode=False)

    # https://github.com/pytorch/pytorch/issues/11201
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.cuda.empty_cache()

    test_args = args.test_args()

    logger = logvis.MyLogger(test_args, context='test_' + test_args.name,
                             log_level=test_args.log_level.upper())

    try:

        main(test_args, logger)

    except Exception as e:

        logger.exception(e)

        logger.warning('Shutting down due to exception...')
