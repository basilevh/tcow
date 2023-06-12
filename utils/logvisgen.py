'''
Logging and visualization logic.
Created by Basile Van Hoorick for TCOW.
'''

import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), 'utils/'))
sys.path.insert(0, os.getcwd())

from __init__ import *


# Library imports.
import imageio
import json
import logging
import os
import wandb
from rich.logging import RichHandler


def _save_video_wrapper(logger, *args, **kwargs):
    logger.save_video(*args, **kwargs)


class MyFilter(logging.Filter):
    '''
    For both console and file logging.
    '''
    def filter(self, record):
        if record.levelno == logging.DEBUG:
            return not('PIL' in record.name) and not('matplotlib' in record.name)
        elif record.levelno == logging.WARNING:
            return not('imageio_ffmpeg' in record.name)
        else:
            return True


class Logger:
    '''
    Provides generic logging and visualization functionality.
    Uses wandb (weights and biases) and pickle.
    '''

    def __init__(self, log_dir=None, context=None, msg_prefix=None, log_level=None):
        '''
        :param log_dir (str): Path to logging folder for this experiment.
        :param context (str): Name of this particular logger instance, for example train / test.
        '''
        self.log_dir = log_dir
        self.context = context
        self.msg_prefix = PROJECT_NAME if msg_prefix is None else msg_prefix
        use_file_io = (log_dir is not None and context is not None)

        if use_file_io:
            self.log_path = os.path.join(self.log_dir, context + '.log')
            self.vis_dir = os.path.join(self.log_dir, 'visuals')
            os.makedirs(self.log_dir, exist_ok=True)
            os.makedirs(self.vis_dir, exist_ok=True)

        # Create rich console and optional file output handler.
        handlers = []

        # NEW (with colors):
        my_handler = rich.logging.RichHandler(
            rich_tracebacks=True, show_time=True, markup=False, show_level=True, show_path=False,
            log_time_format='[%X]', omit_repeated_times=False)
        my_handler.addFilter(MyFilter())
        my_handler.setFormatter(logging.Formatter("%(message)s"))
        handlers.append(my_handler)

        if use_file_io:
            file_handler = logging.FileHandler(self.log_path)
            file_handler.addFilter(MyFilter())
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s : %(levelname)s : %(name)s : %(message)s'))
            handlers.append(file_handler)

        # Configure root & project logger.
        if log_level is None:
            log_level = logging.INFO
        logging.basicConfig(
            level=log_level,
            # format="%(asctime)s : %(levelname)s : %(name)s : %(message)s",
            # format="%(message)s",
            # datefmt="[%X]",
            handlers=handlers)
        self.logger = logging.getLogger(self.msg_prefix)
        self.debug(f'log_level: {log_level}')

        # Instantiate new process with its own queue; initialization will also be done here.
        # https://docs.wandb.ai/guides/track/launch#multiprocess
        # self.log_queue = mp.Queue()
        self.mp_manager = mp.Manager()
        self.initialized = False

        # These lists store losses and other values on an epoch-level time scale. Per key, we have a
        # list of (value, weight) tuples, and we also remember globally whether we want a histogram.
        self.scalar_memory = collections.defaultdict(list)
        self.scalar_memory_hist = dict()

        # This method call initializes some more variables and resets them for every epoch.
        self.accum_buffer_dict = None
        self.already_pushed_set = None
        self.epoch_finished(-1)

    def __getstate__(self):
        # On Windows, some fields in this object (especially those related to multiprocessing)
        # cannot be pickled (which is done when iterating the pytorch dataloader), so we hide them.
        # https://docs.python.org/3/library/pickle.html#handling-stateful-objects
        state = self.__dict__.copy()
        del state['mp_manager']
        # del state['accum_buffer_dict']
        # del state['already_pushed_set']
        return state

    def save_args(self, args):
        '''
        Records all parameters with which the script was called for reproducibility purposes.
        '''
        args_path = os.path.join(self.log_dir, 'args_' + self.context + '.txt')
        with open(args_path, 'w') as f:
            json.dump(args.__dict__, f, indent=2)

    def init_wandb(self, project, args, networks, group='debug', name=None):
        '''
        Initializes the online dashboard, incorporating all PyTorch modules.
        '''
        if name is None:
            name = args.name
        wandb_kwargs = dict(project=project, group=group, config=args, name=name)

        # self.queue_process = mp.Process(target=_init_and_listen,
        #                                 args=(wandb_kwargs, networks, self.log_queue))
        # self.queue_process.start()

        # https://docs.wandb.ai/guides/track/launch#init-start-error
        wandb.init(**wandb_kwargs, settings=wandb.Settings(start_method="fork"))

        if not isinstance(networks, collections.abc.Iterable):
            networks = [networks]
        for net in networks:
            if net is not None:
                wandb.watch(net)

        self.initialized = True

    def debug(self, *args):
        if args == ():
            args = ['']
        self.logger.debug(*args)

    def info(self, *args):
        if args == ():
            args = ['']
        self.logger.info(*args)

    def warning(self, *args):
        if args == ():
            args = ['']
        self.logger.warning(*args)

    def error(self, *args):
        if args == ():
            args = ['']
        self.logger.error(*args)

    def critical(self, *args):
        if args == ():
            args = ['']
        self.logger.critical(*args)

    def exception(self, *args):
        if args == ():
            args = ['']
        self.logger.exception(*args)

    def report_scalar(self, key, value, step=None, remember=False, weight=1.0,
                      commit_histogram=False):
        '''
        Logs a single named value associated with a step.
        If commit_histogram, actual logging is deferred until commit_scalars() is called.
        '''
        if value is None:
            return  # Simply ignore to simplify caller code.
        if not remember:
            if self.initialized:
                wandb.log({key: value}, step=step)
                # self.log_queue.put(({key: value}, step=step))
            else:
                self.debug(f'wandb not initialized1 {str(key)}: {float(str(value)):.5f}')
        else:
            assert weight > 0.0, 'Weight of this reported scalar must be positive.'
            self.scalar_memory[key].append((value, weight))
            self.scalar_memory_hist[key] = commit_histogram

    def commit_scalars(self, keys=None, step=None):
        '''
        Aggregates a bunch of report_scalar() calls for one or more named sets of values and records
        their histograms, i.e. statistical properties.
        '''
        if keys is None:
            keys = list(self.scalar_memory.keys())
        for key in keys:
            if len(self.scalar_memory[key]) == 0:
                continue

            values_weights = np.array(self.scalar_memory[key])  # (N, 2).
            total_weight = np.sum(values_weights[:, 1])
            mean_value = np.sum(np.multiply(values_weights[:, 0], values_weights[:, 1])) / \
                (total_weight + 1e-7)
            self.debug(f'commit_scalars: {key}: mean_value: {mean_value:.5f} '
                       f'total_weight: {total_weight:.5f}')
            if self.initialized:
                if self.scalar_memory_hist[key]:
                    # We are forced to completely ignore weighting in this case.
                    wandb.log({key: wandb.Histogram(values_weights[:, 0])}, step=step)
                else:
                    wandb.log({key: mean_value}, step=step)
            else:
                self.debug(f'^ but wandb not initialized!')

            self.scalar_memory[key].clear()

    def report_histogram(self, key, value, step=None):
        '''
        Directly logs the statistical properties of a named iterable value, such as a list of
        numbers.
        '''
        if self.initialized:
            wandb.log({key: wandb.Histogram(value)}, step=step)
        else:
            self.debug('report_histogram: wandb not initialized')

    def report_single_scalar(self, key, value):
        '''
        Shows a single scalar value as a column or bar plot(?) in the online dashboard.
        '''
        if self.initialized:
            wandb.run.summary[key] = value

    def save_image(self, image, step=None, file_name=None, online_name=None, caption=None,
                   upscale_factor=2, accumulate_online=6):
        '''
        Records a single image to a file in visuals and/or the online dashboard.
        '''
        if image.dtype in [np.float32, np.float64]:
            image = (image * 255.0).astype(np.uint8)

        if upscale_factor > 1:
            image = cv2.resize(
                image,
                (image.shape[1] * upscale_factor, image.shape[0] * upscale_factor),
                interpolation=cv2.INTER_NEAREST)

        if file_name is not None:
            dst_fp = os.path.join(self.vis_dir, file_name)

            parent_dp = str(pathlib.Path(dst_fp).parent)
            if not os.path.exists(parent_dp):
                os.makedirs(parent_dp, exist_ok=True)

            plt.imsave(dst_fp, image)

        if online_name is not None:
            if self.initialized:
                if online_name not in self.accum_buffer_dict:
                    self.accum_buffer_dict[online_name] = self.mp_manager.list()
                self.accum_buffer_dict[online_name].append(wandb.Image(image, caption=caption))
                self._handle_buffer_dicts(online_name, step, accumulate_online, step)
            else:
                self.debug('save_image: wandb not initialized')

    def save_video(self, frames, step=None, file_name=None, online_name=None, caption=None, fps=6,
                   extensions=None, upscale_factor=1, accumulate_online=6, apply_async=False,
                   defer_log=False):
        '''
        Records a single set of frames as a video to a file in visuals and/or the online dashboard.
        '''
        # Ensure before everything else that buffer exists.
        if not(defer_log) and online_name is not None and online_name not in self.accum_buffer_dict:
            self.accum_buffer_dict[online_name] = self.mp_manager.list()

        if apply_async:
            # Start this exact method in parallel.
            kwargs = dict(
                step=step, file_name=file_name, online_name=online_name, caption=caption, fps=fps,
                extensions=extensions, upscale_factor=upscale_factor,
                accumulate_online=accumulate_online, apply_async=False, defer_log=True)
            mp.Process(target=_save_video_wrapper, args=(self, frames), kwargs=kwargs).start()

            # Since we cannot log the usual way (i.e. can't call wandb methods in a separate
            # process), now asynchronously ensure that we are clearing log buffers once in a while.
            # Otherwise, we would have to rely on other modalities in subsequent iterations, or end
            # of epoch, which is not reliable.
            if online_name is not None and self.initialized:
                if self.initialized:
                    self._handle_buffer_dicts(online_name, accumulate_online, step)

            return

        # Duplicate last frame three times for better visibility.
        last_frame = frames[len(frames) - 1:len(frames)]
        frames = np.concatenate([frames, last_frame, last_frame, last_frame], axis=0)

        if frames.dtype in [np.float32, np.float64]:
            frames = (frames * 255.0).astype(np.uint8)

        if upscale_factor > 1:
            frames = [cv2.resize(
                frame,
                (frame.shape[1] * upscale_factor, frame.shape[0] * upscale_factor),
                interpolation=cv2.INTER_NEAREST) for frame in frames]

        for_online_fp = None
        if file_name is not None:

            if extensions is None:
                extensions = ['']

            for ext in extensions:
                used_file_name = file_name + ext
                dst_fp = os.path.join(self.vis_dir, used_file_name)

                parent_dp = str(pathlib.Path(dst_fp).parent)
                if not os.path.exists(parent_dp):
                    os.makedirs(parent_dp, exist_ok=True)

                if dst_fp.lower().endswith('.mp4'):
                    imageio.mimwrite(dst_fp, frames, fps=fps, macro_block_size=None, quality=10)
                elif dst_fp.lower().endswith('.webm'):
                    # https://programtalk.com/python-more-examples/imageio.imread/?ipage=13
                    imageio.mimwrite(dst_fp, frames, fps=fps, codec='libvpx', format='ffmpeg',
                                     ffmpeg_params=["-b:v", "0", "-crf", "14"])
                else:
                    imageio.mimwrite(dst_fp, frames, fps=fps)
                if dst_fp.lower().endswith('.gif') or dst_fp.lower().endswith('.webm'):
                    for_online_fp = dst_fp

        if online_name is not None:
            if self.initialized:
                assert for_online_fp is not None
                self.accum_buffer_dict[online_name].append(
                    # wandb.Video(for_online_fp, caption=caption, fps=fps, format='gif'))
                    wandb.Video(for_online_fp, caption=caption, fps=fps, format='webm'))
                # This should not be done inside an asynchronous call (see above).
                if not(defer_log):
                    self._handle_buffer_dicts(online_name, accumulate_online, step)
            else:
                self.debug('save_video: wandb not initialized')

    def save_gallery(self, frames, step=None, file_name=None, online_name=None, caption=None,
                     upscale_factor=1, accumulate_online=6):
        '''
        Records a single set of frames as a gallery image to a file in visuals and/or the online
        dashboard.
        '''
        if frames.shape[-1] > 3:  # Grayscale: (..., H, W).
            arrangement = frames.shape[:-2]
        else:  # RGB: (..., H, W, 1/3).
            arrangement = frames.shape[:-3]
        if len(arrangement) == 1:  # (A, H, W, 1/3?).
            gallery = np.concatenate(frames, axis=1)  # (H, A*W, 1/3?).
        elif len(arrangement) == 2:  # (A, B, H, W, 1/3?).
            gallery = np.concatenate(frames, axis=1)  # (B, A*H, W, 1/3?).
            gallery = np.concatenate(gallery, axis=1)  # (A*H, B*W, 1/3?).
        else:
            raise ValueError('Too many dimensions to create a gallery.')
        if gallery.dtype in [np.float32, np.float64]:
            gallery = (gallery * 255.0).astype(np.uint8)

        if upscale_factor > 1:
            gallery = cv2.resize(
                gallery,
                (gallery.shape[1] * upscale_factor, gallery.shape[0] * upscale_factor),
                interpolation=cv2.INTER_NEAREST)

        self.save_image(gallery, step=step, file_name=file_name, online_name=online_name,
                        caption=caption, upscale_factor=upscale_factor,
                        accumulate_online=accumulate_online)

    def save_3d(self, object_3d, step=None, online_name=None, caption=None):
        '''
        Records a simple 3D point cloud frame to the online dashboard, optionally with colors.
        '''
        if online_name is not None:
            if self.initialized:
                wandb.log({online_name: wandb.Object3D(object_3d, caption=caption)}, step=step)
            else:
                self.debug('save_3d: wandb not initialized')

    def _handle_buffer_dicts(self, online_name, accumulate_online, step):
        if online_name in self.accum_buffer_dict:
            num_items = len(self.accum_buffer_dict[online_name])
            if num_items >= accumulate_online:
                if self.initialized:
                    to_log = list(self.accum_buffer_dict[online_name])  # Convert ProxyList to list.
                    wandb.log({online_name: to_log}, step=step)
                    self.debug(f'{online_name}: {num_items} (>= {accumulate_online}) items logged.')
                else:
                    self.debug('_handle_buffer_dicts: wandb not initialized')
                self.accum_buffer_dict[online_name] = self.mp_manager.list()
                self.already_pushed_set[online_name] = True  # This adds the key to the set.

    def epoch_finished(self, epoch):
        # These lists store visuals on a step-level time scale, per key.
        # We push and clear them every epoch to:
        # (1) ensure that at least something is logged per epoch;
        # (2) avoid leaking information between epochs.
        # NOTE: Same buffer dictionary is used for image, gallery, video.
        if self.accum_buffer_dict is not None:
            for online_name in self.accum_buffer_dict:
                if not(online_name in self.already_pushed_set):
                    self._handle_buffer_dicts(online_name, 1, epoch)

        # self.accum_buffer_dict = collections.defaultdict(list)
        # self.already_pushed_set = collections.defaultdict(bool)

        # NOTE: These dicts must be made thread-safe because some data can be saved with
        # multiprocessing in an asynchronous manner!
        self.accum_buffer_dict = self.mp_manager.dict()
        self.already_pushed_set = self.mp_manager.dict()
