'''
Data augmentation / transform logic.
Created by Basile Van Hoorick for TCOW.
'''

from __init__ import *

# Internal imports.
import geometry


class MyAugmentationPipeline:

    def __init__(self, logger, num_frames_load, num_frames_clip, frame_height, frame_width,
                 frame_stride, do_random_augs, augs_2d, reverse_prob, palindrome_prob, center_crop):
        '''
        Initializes the data augmentation pipeline.
        '''
        self.logger = logger
        self.num_frames_load = num_frames_load
        self.num_frames_clip = num_frames_clip
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.frame_stride = frame_stride
        self.do_random_augs = do_random_augs
        self.augs_2d = augs_2d
        self.reverse_prob = reverse_prob
        self.palindrome_prob = palindrome_prob
        self.center_crop = center_crop

        # Define color and resize transforms. Crop is handled at runtime.
        self.color_transform = torchvision.transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)
        self.blur_transform = torchvision.transforms.GaussianBlur(5, sigma=(0.1, 3.5))
        self.grayscale_transform = torchvision.transforms.Grayscale(num_output_channels=3)

        # NOTE: For depth, segmentation, object coordinates, and world coordinates, it is in
        # principle important to avoid introducing unrealistic values that interpolate between big
        # jumps in the source arrays. Therefore, NEAREST seems to be the best option for all except
        # RGB itself. However, in practice, we apply smooth interpolation to all arrays except
        # segmentation, which is nearest.
        self.post_resize_smooth = torchvision.transforms.Resize(
            (frame_height, frame_width),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
            antialias=True)
        self.post_resize_nearest = torchvision.transforms.Resize(
            (frame_height, frame_width),
            interpolation=torchvision.transforms.InterpolationMode.NEAREST,
            antialias=False)

    def sample_augs_params(self):
        '''
        Creates a set of random numbers to use for 2D and/or 3D data augmentation. This method
            ensures that we can maintain consistency across modalities and perturbations.
        :return augs_params (dict).
            If a particular value is None, it means that that augmentation is disabled and no
            transformation should be applied. Otherwise, the value indicates the parameters with
            which to apply the transformation (which could still end up leaving some examples
            unchanged, for example if a flip is sampled as False).
        '''
        # Address temporal things first.
        # Offset index is typically in [0, max_delay], but is fixed to max_delay // 2 at test time.
        # However, if palindrome is True and/or frame_stride_factor > 1, then the range of offset
        # values changes depending on the configuration, so we must simultaneously infer lists of
        # actual frame indices to load and return for simplicity.
        # NOTE: If not kubric, then max_delay = 0 so offset = 0, and all other things are disabled.
        palindrome = False
        reverse = False
        frame_stride_factor = 1
        offset = (self.num_frames_load - self.num_frames_clip) // 2
        
        # NOTE: While frame_inds_load refers to file names, frame_inds_clip refers to positions
        # within frame_inds_load!
        frame_inds_load = list(range(0, self.num_frames_load * self.frame_stride, self.frame_stride))
        frame_inds_clip = list(range(0, self.num_frames_clip))
        
        if self.do_random_augs:

            palindrome = (np.random.rand() < self.palindrome_prob)
            if palindrome:
                reverse = (np.random.rand() < 0.35)
                frame_stride_factor = (2 if np.random.rand() < 0.35 else 1)
            else:
                reverse = (np.random.rand() < self.reverse_prob)
                frame_stride_factor = 1

            if palindrome:
                frame_inds_clip = frame_inds_clip + frame_inds_clip[::-1][1:]
            if reverse:
                frame_inds_clip = frame_inds_clip[::-1]
            if frame_stride_factor > 1:
                frame_inds_clip = frame_inds_clip[::frame_stride_factor]
            
            # Determine offset, now that the number of available frames may have changed.
            num_frames_avail = len(frame_inds_clip)
            assert num_frames_avail >= self.num_frames_clip
            offset = np.random.randint(0, num_frames_avail - self.num_frames_clip + 1)
            frame_inds_clip = frame_inds_clip[offset:offset + self.num_frames_clip]
        
        # Create dictionary with info.
        augs_params = dict()
        augs_params['palindrome'] = palindrome
        augs_params['reverse'] = reverse
        augs_params['frame_stride_factor'] = frame_stride_factor
        augs_params['offset'] = offset
        augs_params['frame_inds_load'] = np.array(frame_inds_load)
        augs_params['frame_inds_clip'] = np.array(frame_inds_clip)
        
        # Next, address other (mostly color & spatial) augmentations. Initialize with default values
        # that correspond to identity (data loader collate does not support None!).
        color_jitter = False
        rgb_blur = False
        rgb_grayscale = False
        horz_flip = False
        crop_rect = -np.ones(4)

        if self.do_random_augs:
            color_jitter = (np.random.rand() < 0.9)
            rgb_blur = (np.random.rand() < 0.2)
            rgb_grayscale = (np.random.rand() < 0.05)

            if self.augs_2d:
                horz_flip = (np.random.rand() < 0.5)
                crop_y1 = np.random.rand() * 0.2
                crop_y2 = np.random.rand() * 0.2 + 0.8
                crop_x1 = np.random.rand() * 0.2
                crop_x2 = np.random.rand() * 0.2 + 0.8
                crop_rect = np.array([crop_y1, crop_y2, crop_x1, crop_x2])

        # Update dictionary.
        augs_params['color_jitter'] = color_jitter
        augs_params['rgb_blur'] = rgb_blur
        augs_params['rgb_grayscale'] = rgb_grayscale
        augs_params['horz_flip'] = horz_flip
        augs_params['crop_rect'] = crop_rect

        return augs_params

    def apply_augs_2d_frames(self, modalities_noaug, augs_params):
        '''
        :param modalities_noaug (dict): Maps frame types (rgb / segm / ...) to original
            (1/3/K, Tv, H, W) tensors.
        :param augs_params (dict): Maps transform names to values (which could be None).
        :return modalities_aug (dict): Maps frame types to augmented (1/3/K, Tc, H, W) tensors.
        '''
        modalities_aug = dict()

        for modality, raw_frames_untrim in modalities_noaug.items():

            # In some cases, some arrays explicitly do not exist (e.g. no xyz or div_segm).
            if len(raw_frames_untrim.shape) < 4:
                modalities_aug[modality] = raw_frames_untrim.clone()
                continue

            # Address temporal things first (always, but test time values are fixed).
            frame_inds_clip = augs_params['frame_inds_clip']
            assert len(frame_inds_clip) == self.num_frames_clip
            raw_frames = raw_frames_untrim[:, frame_inds_clip, :, :]
            assert raw_frames.shape[1] == self.num_frames_clip, \
                f'raw_frames: {raw_frames.shape}  num_frames_clip: {self.num_frames_clip}'
            
            (C, T, H, W) = raw_frames.shape
            assert ((C > 3) == ('div' in modality))
            distort_frames = rearrange(raw_frames, 'C T H W -> T C H W')

            # Apply center crop if needed (test time only).
            if self.center_crop:
                current_ar = W / H
                desired_ar = self.frame_width / self.frame_height
                if current_ar > desired_ar:
                    crop_tf = torchvision.transforms.CenterCrop((H, int(H * desired_ar)))
                    distort_frames = crop_tf(distort_frames)
                elif current_ar < desired_ar:
                    crop_tf = torchvision.transforms.CenterCrop((int(W / desired_ar), W))
                    distort_frames = crop_tf(distort_frames)

            # Apply color perturbation (train time only).
            # NOTE: The trailing dimensions have to be (1/3, H, W) for this to work correctly.
            if 'rgb' in modality:
                if augs_params['color_jitter']:
                    distort_frames = self.color_transform(distort_frames)
                if augs_params['rgb_blur']:
                    distort_frames = self.blur_transform(distort_frames)
                if augs_params['rgb_grayscale']:
                    distort_frames = self.grayscale_transform(distort_frames)

            # Apply random horizontal flip (train time only).
            if augs_params['horz_flip']:
                distort_frames = torch.flip(distort_frames, dims=[-1])

            # Apply crops (train time only).
            # NOTE: These values always pertain to coordinates within post-flip images.
            crop_rect = augs_params['crop_rect']
            if crop_rect is not None and np.all(np.array(crop_rect) >= 0.0):
                (crop_y1, crop_y2, crop_x1, crop_x2) = crop_rect
                crop_frames = distort_frames[..., int(crop_y1 * H):int(crop_y2 * H),
                                             int(crop_x1 * W):int(crop_x2 * W)]
                distort_frames = crop_frames

            # Resize to final size (always).
            if 'segm' in modality or 'mask' in modality:
                # Segmentation masks have integer values.
                resize_frames = self.post_resize_nearest(distort_frames)
            else:
                # RGB, depth, object coordinates.
                resize_frames = self.post_resize_smooth(distort_frames)

            resize_frames = rearrange(resize_frames, 'T C H W -> C T H W')
            modalities_aug[modality] = resize_frames

        return modalities_aug
