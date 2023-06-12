'''
Plug in any video loading and processing logic.
Created by Basile Van Hoorick for TCOW.
'''

from __init__ import *

# Internal imports.
import augs
import data_utils


def load_timestamped_named_mask_files(src_dp, src_fn, name, early_resize_height):
    '''
    :param src_dp (str): Directory containing the mask files.
    :param src_fn (str): Video file name, if it exists (otherwise, RGB frames are also in src_dp,
        and this is None).
    :return raw_named_frames (dict) mapping time index to (H, W, 1) arrays.
    '''
    named_fns = sorted(os.listdir(src_dp))
    named_fns = [fn for fn in named_fns if f'_{name}_' in fn or f'_{name}.' in fn]
    if src_fn is not None:
        named_fns = [fn for fn in named_fns if src_fn.split('.')[0] in fn]
    named_fps = [os.path.join(src_dp, fn) for fn in named_fns]

    raw_named_frames = dict()  # Maps source frame index to (H, W, 3) array.
    for fp in named_fps:

        # NOTE: The old system uses myvid_query_10.png, but the new system uses
        # myvid_10_query.png, myvid_70_occl.png, myvid_130_snitch.png, myvid_190_contoccl.png, etc.
        if f'_{name}_' in fp:
            named_frame_idx = int(fp.split(f'_{name}_')[-1].split('.')[0])
        elif f'_{name}.' in fp:
            named_frame_idx = int(fp.split(f'_{name}.')[-2].split('_')[-1].split('/')[-1])
        else:
            raise ValueError(f'Could not parse named frame index from {fp}')

        named_frame = plt.imread(fp)[..., 0:3]  # (H, W, 3).
        
        if early_resize_height is not None and early_resize_height > 0:
            (H1, W1) = named_frame.shape[:2]
            if H1 > early_resize_height:
                (H2, W2) = (early_resize_height, int(round(early_resize_height * W1 / H1)))
                named_frame = cv2.resize(named_frame, (W2, H2), interpolation=cv2.INTER_LINEAR)
        
        named_frame = (named_frame.sum(axis=-1) > 0.1).astype(np.uint8)[..., None]  # (H, W, 1).
        raw_named_frames[named_frame_idx] = named_frame

    return raw_named_frames


class PluginVideoDataset(torch.utils.data.Dataset):
    '''
    X
    '''

    def __init__(self, src_path, logger, phase, num_clip_frames=20, frame_height=240,
                 frame_width=320, frame_rate=30, prefer_frame_stride=3, multiplicity=12,
                 query_time=0.2, annots_must_exist=False, prefetch=False, center_crop=False,
                 early_resize_height=480):
        '''
        Initializes the dataset.
        '''
        self.src_path = src_path
        self.logger = logger
        self.phase = phase
        self.multiplicity = multiplicity

        assert self.phase == 'test'

        # Final clip options.
        self.num_clip_frames = num_clip_frames
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.frame_rate = frame_rate
        self.prefer_frame_stride = prefer_frame_stride
        self.query_time_val = query_time
        self.query_time_idx = int(np.floor(query_time * num_clip_frames))
        self.annots_must_exist = annots_must_exist
        self.prefetch = prefetch
        self.center_crop = center_crop
        self.early_resize_height = early_resize_height

        # Instantiate custom augmentation pipeline to simply perform resizing.
        self.augs_pipeline = augs.MyAugmentationPipeline(
            self.logger, self.num_clip_frames, self.num_clip_frames, self.frame_height,
            self.frame_width, self.prefer_frame_stride, False, False,
            0.0, 0.0, self.center_crop)
        self.to_tensor = torchvision.transforms.ToTensor()

        # Figure out paths.
        if os.path.isdir(self.src_path):
            self.src_dp = self.src_path
            self.src_fp = None
            self.src_fn = None

        else:
            self.src_dp = str(pathlib.Path(self.src_path).parent)
            self.src_fp = self.src_path
            self.src_fn = str(pathlib.Path(self.src_path).name)
            assert os.path.exists(self.src_fp)

        # Prefetch raw input video if enabled.
        if self.prefetch:
            self.raw_frames = self._get_raw_frames()
            self.num_video_frames = len(self.raw_frames)
        else:
            self.raw_frames = None
            self.num_video_frames = self._get_num_video_frames()

        # Prefetch available query and target (snitch / occl / cont / both) masks.
        # NOTE: If src_path is a video file, then query/target index has to refer to 0=based frame
        # index. But if src_path is a folder, then query/target index has to refer to existing
        # 0-based FILE ordering, not underlying frame ordering which may be potentially different!
        # Therefore, always pick the all_frames folder for YouTube-VOS, or carefully name the
        # query/target file correctly.
        # All four dicts map source frame index to (H, W, 1) array.
        self.raw_query_frames = load_timestamped_named_mask_files(
            self.src_dp, self.src_fn, 'query', self.early_resize_height)
        self.raw_snitch_frames = load_timestamped_named_mask_files(
            self.src_dp, self.src_fn, 'snitch', self.early_resize_height)
        self.raw_occl_frames = load_timestamped_named_mask_files(
            self.src_dp, self.src_fn, 'occl', self.early_resize_height)
        self.raw_cont_frames = load_timestamped_named_mask_files(
            self.src_dp, self.src_fn, 'cont', self.early_resize_height)

        # For efficiency, I named some annotation files as both, which marks the main container is
        # the same as the main occluder.
        raw_both_frames = load_timestamped_named_mask_files(
            self.src_dp, self.src_fn, 'contoccl', self.early_resize_height)
        self.raw_occl_frames.update(raw_both_frames)
        self.raw_cont_frames.update(raw_both_frames)

        # Get valid video to clip subsampling modes, incorporating query and target times.
        available_input_inds = sorted(list(range(self.num_video_frames)))
        available_query_inds = sorted(list(self.raw_query_frames.keys()))
        available_target_inds = sorted(list(set(self.raw_snitch_frames.keys())
                                            | set(self.raw_occl_frames.keys())
                                            | set(self.raw_cont_frames.keys())))
        min_target_frames_covered = (1 if self.annots_must_exist else 0)
        self.usage_modes = data_utils.get_usage_modes(
            available_input_inds, available_query_inds, available_target_inds, self.num_clip_frames,
            self.query_time_idx, min_target_frames_covered=min_target_frames_covered)

        self.logger.info(f'(PluginVideoDataset) Valid usage modes (min_target_frames_covered: '
                         f'{min_target_frames_covered}): {self.usage_modes}')

    def __len__(self):
        # return self.multiplicity
        return len(self.usage_modes)

    def __getitem__(self, index):
        # Obtain clip bounds.
        usage_mode_idx = index % len(self.usage_modes)
        (frame_start, frame_stride, target_coverage) = self.usage_modes[usage_mode_idx]
        frame_inds = list(range(frame_start, frame_start + self.num_clip_frames * frame_stride,
                                frame_stride))
        augs_params = self.augs_pipeline.sample_augs_params()
        # ^ frame_inds_load and frame_inds_clip are ignored!

        # Obtain raw input video (either lazy or prefetched).
        # NOTE: raw_frames could be either list or numpy array!
        if self.prefetch:
            raw_frames = self.raw_frames
        else:
            raw_frames = self._get_raw_frames()
        assert len(raw_frames) == self.num_video_frames

        # Loop over all input frames.
        pv_rgb = []
        for f, t in enumerate(frame_inds):
            rgb = raw_frames[t]
            if np.issubdtype(rgb.dtype, np.integer):
                rgb = (rgb / 255.0).astype(np.float32)
            pv_rgb.append(rgb)
        pv_rgb = np.stack(pv_rgb, axis=0)  # (T, Hf, Wf, 3) float in [0, 1].
        (T, Hf, Wf, _) = pv_rgb.shape
        assert T == self.num_clip_frames

        # Obtain query mask.
        pv_query = np.zeros_like(pv_rgb[..., 0:1], dtype=np.uint8)
        pv_query[self.query_time_idx] = self.raw_query_frames[frame_inds[self.query_time_idx]]
        # (T, Hf, Wf, 1) uint8 in [0, 1].

        # Construct sparse target mask (snitch + occluder + container).
        pv_target = np.ones_like(pv_rgb[..., 0:3], dtype=np.int8) * (-1)
        for (t, v) in self.raw_snitch_frames.items():
            f = int(round((t - frame_start) / frame_stride))
            if f >= 0 and f < T:
                pv_target[f, ..., 0] = v[..., 0]
        for (t, v) in self.raw_occl_frames.items():
            f = int(round((t - frame_start) // frame_stride))
            if f >= 0 and f < T:
                pv_target[f, ..., 1] = v[..., 0]
        for (t, v) in self.raw_cont_frames.items():
            f = int(round((t - frame_start) // frame_stride))
            if f >= 0 and f < T:
                pv_target[f, ..., 2] = v[..., 0]
        # (T, Hf, Wf, 3) int8 in [-1, 1].

        # Convert large numpy arrays to torch tensors, putting channel dimension first.
        pv_rgb_tf = rearrange(torch.tensor(pv_rgb, dtype=torch.float32), 'T H W C -> C T H W')
        pv_query_tf = rearrange(torch.tensor(pv_query, dtype=torch.uint8), 'T H W C -> C T H W')
        pv_target_tf = rearrange(torch.tensor(pv_target, dtype=torch.int8), 'T H W C -> C T H W')

        # Apply 2D cropping.
        # NOTE: We must apply the transforms consistently across all tensors.
        # NOTE: Make sure to have mask in key names of tensors where we wish to apply nearest (not
        # bilinear) resizing.
        modalities = {'rgb': pv_rgb_tf, 'query_mask': pv_query_tf, 'target_mask': pv_target_tf}
        modalities_tf = self.augs_pipeline.apply_augs_2d_frames(modalities, augs_params)
        (pv_rgb_tf, pv_query_tf, pv_target_tf) = (
            modalities_tf['rgb'], modalities_tf['query_mask'], modalities_tf['target_mask'])

        # Organize & return results.
        data_retval = dict()
        data_retval['source_name'] = 'plugin'
        data_retval['src_path'] = self.src_path
        data_retval['dset_idx'] = index
        data_retval['scene_idx'] = 0
        data_retval['usage_mode_idx'] = usage_mode_idx
        data_retval['frame_inds'] = frame_inds
        data_retval['augs_params'] = augs_params
        data_retval['frame_start'] = frame_start
        data_retval['frame_stride'] = frame_stride
        data_retval['match_prefer_fstride'] = (frame_stride == self.prefer_frame_stride)

        # Include augmented data.
        data_retval['pv_rgb_tf'] = pv_rgb_tf  # (3, T, Hf, Wf).
        data_retval['pv_query_tf'] = pv_query_tf  # (1, T, Hf, Wf).
        data_retval['pv_target_tf'] = pv_target_tf  # (3, T, Hf, Wf).

        return data_retval

    def _get_raw_frames(self):
        self.logger.warning()

        if os.path.isdir(self.src_path):
            self.logger.warning(
                f'(PluginVideoDataset) Loading {self.src_path} as a directory with sorted images...')
            frames = data_utils.read_all_images(
                self.src_dp, exclude_patterns=['query', 'snitch', 'occl', 'cont'], use_tqdm=True,
                early_resize_height=self.early_resize_height)
            raw_frames = frames
            # List, not numpy array.

        else:
            self.logger.warning(
                f'(PluginVideoDataset) Loading {self.src_path} as a single video file...')
            raw_frames = imageio.mimread(self.src_fp, memtest='2GB')
            # List, not numpy array.

        self.logger.warning()

        return raw_frames

    def _get_num_video_frames(self):

        if os.path.isdir(self.src_path):
            return data_utils.read_all_images(
                self.src_dp, exclude_patterns=['query', 'snitch', 'occl', 'cont'], count_only=True)

        else:
            cap = cv2.VideoCapture(self.src_fp)
            return int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
