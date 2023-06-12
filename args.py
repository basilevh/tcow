'''
Handling of parameters that can be passed to training and testing scripts.
Created by Basile Van Hoorick for TCOW.
'''

from __init__ import *

# Library imports.
import argparse
import os

# Internal imports.
import my_utils


def _str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def _arg2str(arg_value):
    if isinstance(arg_value, bool):
        return '1' if arg_value else '0'
    else:
        return str(arg_value)


def shared_args(parser):
    '''
    These parameters can be passed to both training and testing / evaluation files.
    '''

    # Misc options.
    parser.add_argument('--seed', default=900, type=int,
                        help='Random number generator seed.')
    parser.add_argument('--log_level', default='info', type=str,
                        choices=['debug', 'info', 'warn'],
                        help='Threshold for command line output.')

    # Resource options.
    parser.add_argument('--device', default='cuda', type=str,
                        choices=['cuda', 'cpu'],
                        help='cuda or cpu.')
    parser.add_argument('--batch_size', default=4, type=int,
                        help='Batch size during training or testing.')
    parser.add_argument('--num_workers', default=-1, type=int,
                        help='Number of data loading workers; -1 means automatic.')

    # Logging & checkpointing options.
    parser.add_argument('--checkpoint_root', default='checkpoints/', type=str,
                        help='Path to parent collection of checkpoint folders.')
    parser.add_argument('--log_root', default='logs/', type=str,
                        help='Path to parent collection of logs, visualizations, and results.')
    parser.add_argument('--name', '--tag', default='', type=str,
                        help='Recognizable, unique tag of this experiment for bookkeeping. A good '
                        'practice would be to include a version number.')
    parser.add_argument('--resume', '--checkpoint_name', default='', type=str,
                        help='Tag of checkpoint to resume from. This has to match an experiment '
                        'name that is available under checkpoint_root.')
    parser.add_argument('--epoch', default=-1, type=int,
                        help='If >= 0, desired model epoch to evaluate or resume from (0-based), '
                        'otherwise pick latest.')
    parser.add_argument('--avoid_wandb', default=0, type=int,
                        help='If 1, do not log videos online. If 2, do not log anything online.')
    parser.add_argument('--log_rarely', default=0, type=int,
                        help='If 1, create videos rarely.')

    # Data options (all phases).
    parser.add_argument('--data_path', required=True, type=str, nargs='+',
                        help='Path to dataset root folder(s) (Kubric or plugin or X).')
    parser.add_argument('--use_data_frac', default=1.0, type=float,
                        help='If < 1.0, use a smaller dataset.')
    parser.add_argument('--num_queries', default=1, type=int,
                        help='For query-based trackers, number of objects to track per example in '
                        'the pipeline. At both train and test time, this will happen sequentially '
                        'in a for loop (i.e. it should not significantly affect VRAM usage).')

    # Automatically inferred options (do not assign).
    parser.add_argument('--is_debug', default=False, type=_str2bool,
                        help='Shorter epochs; log and visualize more often.')
    parser.add_argument('--checkpoint_path', default='', type=str,
                        help='Path to current checkpoint directory for this experiment.')
    parser.add_argument('--train_log_path', default='', type=str,
                        help='Path to current train-time logging directory for this experiment.')
    parser.add_argument('--log_path', default='', type=str,
                        help='Switches to train or test depending on the job.')
    parser.add_argument('--wandb_group', default='group', type=str,
                        help='Group to put this experiment in on weights and biases.')


def train_args():

    parser = argparse.ArgumentParser()

    shared_args(parser)

    # Training / misc options.
    parser.add_argument('--num_epochs', default=70, type=int,
                        help='Number of epochs to train for.')
    parser.add_argument('--checkpoint_every', default=2, type=int,
                        help='Store permanent model checkpoint every this number of epochs.')
    parser.add_argument('--learn_rate', default=1e-4, type=float,
                        help='Initial learning rate.')
    parser.add_argument('--lr_decay', default=0.3, type=float,
                        help='Learning rate factor per step for scheduler.')
    parser.add_argument('--do_val_aug', default=True, type=_str2bool,
                        help='If True, perform validation phase with data augmentation.')
    parser.add_argument('--do_val_noaug', default=False, type=_str2bool,
                        help='If True, also perform validation phase with no data augmentation '
                        'after every epoch, in addition to val_aug.')
    parser.add_argument('--val_every', default=2, type=int,
                        help='Epoch interval for validation phase(s).')

    # General data options.
    parser.add_argument('--num_frames', default=24, type=int,
                        help='Video clip length.')
    parser.add_argument('--frame_height', default=240, type=int,
                        help='Post-processed image vertical size.')
    parser.add_argument('--frame_width', default=320, type=int,
                        help='Post-processed image horizontal size.')
    parser.add_argument('--augs_2d', default=True, type=_str2bool,
                        help='Apply random spatial flipping & cropping during train / val.')

    # Kubric data & augmentation options.
    parser.add_argument('--kubric_frame_rate', default=12, type=int,
                        help='Frames per second (FPS) for Kubric / PyBullet simulations.')
    parser.add_argument('--kubric_frame_stride', default=1, type=int,
                        help='Temporal frame interval for model versus source dataset.')
    parser.add_argument('--kubric_max_delay', default=6, type=int,
                        help='To increase diversity, clip frame start will be randomly sampled '
                        'between 0 and this offset value (inclusive) within the dataset video. '
                        'At test time, simply use this value // 2.')
    parser.add_argument('--kubric_reverse_prob', default=0.1, type=float,
                        help='To increase diversity, randomly return temporally flipped videos '
                        'with this probability. At test time, this is always disabled.')
    parser.add_argument('--kubric_palindrome_prob', default=0.1, type=float,
                        help='To increase diversity, randomly play videos forward and backward '
                        'with this probability, to encourage learning about objects returning from '
                        'out-of-frame. We choose backward+forward with 35% probability. The frame '
                        'stride will be doubled with 35% probability. At test time, this is always '
                        'disabled entirely.')

    # Seeker (localizer) options.
    parser.add_argument('--tracker_pretrained', default='1', type=str,
                        help='If False, random initialization. If True, initialize TimeSformer '
                        'network with ImageNet pretrained weights (from ViT Base). If string, '
                        'initialize TimeSformer network with this path to a checkpoint file.')
    parser.add_argument('--attention_type', default='divided_space_time', type=str,
                        choices=['divided_space_time', 'joint_space_time'],
                        help='For TimeSformer only.')
    parser.add_argument('--patch_size', default=16, type=int,
                        help='For TimeSformer only.')
    parser.add_argument('--causal_attention', default=1, type=int,
                        help='For TimeSformer only.')
    parser.add_argument('--norm_embeddings', default=False, type=_str2bool,
                        help='For TimeSformer only.')
    parser.add_argument('--drop_path_rate', default=0.1, type=float,
                        help='For TimeSformer only.')
    parser.add_argument('--network_depth', default=12, type=int,
                        help='For TimeSformer only.')
    parser.add_argument('--seeker_frames', default=[-1], type=int, nargs='+',
                        help='How many input frames the tracker perceives. If < num_frames, this '
                        'implies we must perform forecasting as part of the prediction. If one '
                        'value, then it is always used, but if two values, it is interpreted as an '
                        'inclusive range for sampling random visible clip durations.')
    parser.add_argument('--seeker_query_time', default=0.0, type=float,
                        help='How far into (i.e. which frame of) the video all queries are '
                        'applied.')

    # Loss & optimization options.
    parser.add_argument('--gradient_clip', default=0.3, type=float,
                        help='If > 0, clip gradient L2 norm to this value for stability.')
    parser.add_argument('--optimizer', default='adamw', type=str,
                        choices=['sgd', 'adam', 'adamw', 'lamb'],
                        help='Which optimizer to use for training.')
    parser.add_argument('--track_lw', default=1.0, type=float,
                        help='Weight for target / snitch mask via heatmap loss.')
    parser.add_argument('--occl_mask_lw', default=0.5, type=float,
                        help='Weight for frontmost occluder mask for hierarchical tracking.')
    parser.add_argument('--cont_mask_lw', default=0.5, type=float,
                        help='Weight for outermost container mask for hierarchical tracking.')
    parser.add_argument('--occluded_weight', default=5.0, type=int,
                        help='Frames, as well as pixels, corresponding to mostly invisible snitch '
                        'targets are considered this factor times more important for the loss.')
    parser.add_argument('--occl_cont_zero_weight', default=0.02, type=float,
                        help='When no main ocluder or container is present, still supervise with '
                        'all zero with this small factor.')

    # Hierarchical object segmentation / tracking / heatmap options.
    parser.add_argument('--class_balancing', default=True, type=_str2bool,
                        help='Whether to weight seeker target heatmap 1 and 0 equally.')
    parser.add_argument('--focal_loss', default=False, type=_str2bool,
                        help='Whether to use focal loss instead of BCE for heatmap.')
    parser.add_argument('--aot_loss', default=0.8, type=float,
                        help='Apply bootstrapped BCE and soft Jaccard (= Tversky with alpha = beta '
                        '= 1) losses. One minus this is the default / custom BCE loss.')
    parser.add_argument('--hard_negative_factor', default=3.0, type=float,
                        help='Within every frame, increase loss weight for pixels near (but not '
                        'inside) partially occluded snitch masks. This should improve amodal '
                        'completion segmentation quality.')
    parser.add_argument('--front_occl_thres', default=0.95, type=float,
                        help='Soft occlusion fraction before snitch is considered fully occluded, '
                        'for frontmost occluder segmentation.')
    parser.add_argument('--outer_cont_thres', default=0.75, type=float,
                        help='Lower bound of containment percentage before snitch is considered '
                        'fully contained, for outermost container segmentation.')

    args = parser.parse_args()
    verify_args(args, is_train=True)

    return args


def test_args():

    parser = argparse.ArgumentParser()

    # NOTE: Don't forget to consider this method as well when adding arguments.
    shared_args(parser)

    # Resource options.
    parser.add_argument('--gpu_id', default=0, type=int,
                        help='GPU index.')

    # Plugin data options.
    parser.add_argument('--plugin_frame_rate', default=30, type=int,
                        help='Original frames per second (FPS) for plugin video inputs. '
                        'This value only affects visualization FPS.')
    parser.add_argument('--plugin_prefer_frame_stride', default=3, type=int,
                        help='Representative temporal frame interval for model versus '
                        'source dataset. This value only affects visualization FPS.')
    parser.add_argument('--center_crop', default=True, type=_str2bool,
                        help='For plugin videos, ensure aspect ratios are aligned with the '
                        'training set, i.e. Kubric.')

    # Inference & processing options.
    parser.add_argument('--store_results', default=False, type=_str2bool,
                        help='In addition to generating lossy 2D visuals, save all inputs & '
                        'outputs to disk for later processing, visualizations, metrics, or other '
                        'deep dives.')
    parser.add_argument('--annots_must_exist', default=False, type=_str2bool,
                        help='For plugin videos, only run the model for usage modes where at least '
                        'one target frame is available for calculating metrics.')
    parser.add_argument('--extra_visuals', default=False, type=_str2bool)

    # Automatically inferred options (do not assign).
    parser.add_argument('--test_log_path', default='', type=str,
                        help='Path to current logging directory for this experiment evaluation.')

    args = parser.parse_args()
    verify_args(args, is_train=False)

    return args


def verify_args(args, is_train=False):

    args.is_debug = args.name.startswith('d')
    
    args.wandb_group = ('train' if is_train else 'test') + \
                       ('_debug' if args.is_debug else '')

    if is_train:

        assert args.occl_cont_zero_weight < 0.5

    else:

        # Not supporting batches at test time simplifies things.
        args.batch_size = 1

    if args.num_workers < 0:
        if is_train:
            if args.is_debug:
                args.num_workers = max(int(mp.cpu_count() * 0.30) - 4, 4)
            else:
                args.num_workers = max(int(mp.cpu_count() * 0.45) - 6, 4)
        else:
            args.num_workers = 4
        args.num_workers = min(args.num_workers, 80)
    args.num_workers = int(args.num_workers)

    # If we have no name (e.g. for smaller scripts in eval), assume we are not interested in logging
    # either.
    if args.name != '':

        if args.resume != '':
            resume_name = args.resume
            if args.epoch >= 0:
                args.resume = os.path.join(args.checkpoint_root, args.resume, f'model_{args.epoch}.pth')
            else:
                args.resume = os.path.join(args.checkpoint_root, args.resume, 'checkpoint.pth')

        if is_train:
            # For example, --name v1.
            args.checkpoint_path = os.path.join(args.checkpoint_root, args.name)
            args.train_log_path = os.path.join(args.log_root, args.name)

            os.makedirs(args.checkpoint_path, exist_ok=True)
            os.makedirs(args.train_log_path, exist_ok=True)

            if args.resume != '':
                # Train example: --resume v3 --name dbg4.
                # NOTE: In this case, we wish to bootstrap another already trained model, yet resume
                # in our own new logs folder! The rest is handled by train.py.
                pass

            args.log_path = args.train_log_path

        else:
            assert args.resume != ''
            # Test example: --resume v1 --name t1.
           
            args.checkpoint_path = os.path.join(args.checkpoint_root, resume_name)
            args.train_log_path = os.path.join(args.log_root, resume_name)
            
            assert os.path.exists(args.checkpoint_path) and os.path.isdir(args.checkpoint_path)
            assert os.path.exists(args.train_log_path) and os.path.isdir(args.train_log_path)
            assert os.path.exists(args.resume) and os.path.isfile(args.resume)

            # Ensure that 0-based epoch is always part of the name and log directories.
            epoch = my_utils.get_checkpoint_epoch(args.resume)
            args.name += f'_e{epoch}'

            args.test_log_path = os.path.join(args.train_log_path, 'test_' + args.name)
            args.log_path = args.test_log_path
            os.makedirs(args.test_log_path, exist_ok=True)

    # NOTE: args.log_path is the one actually used by logvis.
