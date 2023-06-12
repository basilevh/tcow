'''
Neural network architecture description.
Created by Basile Van Hoorick for TCOW.
'''

import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), 'seeker/'))
sys.path.insert(0, os.path.join(os.getcwd(), 'third_party/aot-benchmark/'))
sys.path.insert(0, os.getcwd())

from __init__ import *

# Internal imports.
import resnet
import vision_tf


class QueryMaskTracker(torch.nn.Module):
    '''
    X
    '''

    def __init__(self, logger, num_total_frames=24, num_visible_frames=16, frame_height=224,
                 frame_width=288, tracker_pretrained=False, attention_type='divided_space_time',
                 patch_size=16, causal_attention=False, norm_embeddings=False, drop_path_rate=0.1,
                 network_depth=12, track_map_stride=4, track_map_resize='bilinear',
                 query_channels=1, output_channels=3, flag_channels=3):
        super().__init__()
        self.logger = logger
        self.num_total_frames = num_total_frames
        self.num_visible_frames = num_visible_frames
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.attention_type = attention_type
        self.patch_size = patch_size
        self.causal_attention = causal_attention
        self.norm_embeddings = norm_embeddings
        self.drop_path_rate = drop_path_rate
        self.network_depth = network_depth
        self.track_map_stride = track_map_stride
        self.track_map_resize = track_map_resize
        self.query_channels = query_channels
        self.output_channels = output_channels
        self.flag_channels = flag_channels

        # Determine precise input shapes.
        self.input_channels = 3 + self.query_channels

        # Determine precise output shapes.
        self.output_channels = output_channels

        # Translate given pretrained info.
        self.pretrained_path = ''
        if isinstance(tracker_pretrained, bool):
            self.tracker_pretrained = tracker_pretrained
        elif isinstance(tracker_pretrained, str):
            # Consistent with _str2bool().
            if tracker_pretrained.lower() in ['1', 'y', 'yes', 't', 'true']:
                self.tracker_pretrained = True  # Defaults to vit_base_patch16_224 (ImageNet).
            elif len(tracker_pretrained) <= 5:
                self.tracker_pretrained = False
            else:
                self.tracker_pretrained = True
                self.pretrained_path = tracker_pretrained  # Custom file path on disk.
        else:
            raise ValueError(f'Invalid tracker_pretrained value: {tracker_pretrained}.')
        self.logger.info(f'(QueryMaskTracker) tracker_pretrained: {self.tracker_pretrained} '
                         f'pretrained_path: {self.pretrained_path}')

        # Instantiate actual network components.
        # Instantiate tracker backbone.
        self.tracker_backbone = vision_tf.MyDenseTimeSformerBackbone(
            self.logger, num_frames=self.num_total_frames, frame_height=self.frame_height,
            frame_width=self.frame_width, patch_dim=self.patch_size,
            in_channels=self.input_channels, pretrained=self.tracker_pretrained,
            pretrained_path=self.pretrained_path, attention_type=self.attention_type,
            causal_attention=self.causal_attention, norm_embeddings=self.norm_embeddings,
            drop_path_rate=self.drop_path_rate, network_depth=self.network_depth)
        self.use_feature_dim = self.tracker_backbone.output_feature_dim

        # This applies to every spatiotemporal patch (typically C x 1 x 16 x 16).
        self.tracker_post_linear = torch.nn.Linear(
            self.use_feature_dim, self.output_channels * self.patch_size * self.patch_size)
        if self.flag_channels > 0:
            self.flag_post_linear = torch.nn.Linear(self.use_feature_dim, self.flag_channels)
            # Flags are typically (occluded, contained, soft_fraction).

        assert self.frame_height % self.patch_size == 0
        assert self.frame_width % self.patch_size == 0

    def forward(self, input_frames, query_mask):
        '''
        Assumes input frames are already blacked out as appropriate.
        :param input_frames (B, 3-7, T, Hf, Wf) tensor.
        :param query_mask (B, C, T, Hf, Wf) tensor.
        :return (output_mask, output_flags).
            output_mask (B, C, T, Hf, Wf) tensor.
            output_flags (B, T, F) tensor.
        '''
        # Append query information in desired way.
        (B, _, T, Hf, Wf) = input_frames.shape
        input_frames = input_frames.type(torch.float32)
        query_mask = query_mask.type(torch.float32)
        assert query_mask.shape[1] == 1

        input_with_query = input_frames.clone()
        input_with_query = torch.cat([input_with_query, query_mask], dim=1)

        (output_features, _) = self.tracker_backbone(input_with_query, None)

        output_features = rearrange(output_features, 'B D T H W -> B T H W D')
        output_patches = self.tracker_post_linear(output_features)  # (B, T, H, W, D).
        output_mask = rearrange(output_patches, 'B T H W (C h w) -> B C T (H h) (W w)',
                                C=self.output_channels, h=self.patch_size, w=self.patch_size)

        # Make output coarser such that optimization (hopefully) focuses less on precise boundaries.
        if self.track_map_stride > 1:
            # Awkward way to do this, but it works.
            output_mask = rearrange(output_mask, 'B C T Hf Wf -> (B T) C Hf Wf')
            output_mask = torch.nn.functional.avg_pool2d(
                output_mask, self.track_map_stride, self.track_map_stride)
            
            if self.track_map_resize == 'nearest':
                output_mask = torch.nn.functional.interpolate(
                    output_mask, scale_factor=self.track_map_stride, mode='nearest')
            elif self.track_map_resize == 'bilinear':
                output_mask = torch.nn.functional.interpolate(
                    output_mask, scale_factor=self.track_map_stride, mode='bilinear',
                    align_corners=True)
            
            output_mask = rearrange(output_mask, '(B T) C Hf Wf -> B C T Hf Wf', B=B, T=T)

        # Calculate extra info per frame.
        if self.flag_channels > 0:
            output_flags = self.flag_post_linear(output_features)  # (B, T, H, W, F).
            output_flags = output_flags.mean(dim=[-2, -3])  # (B, T, F).

        else:
            output_flags = None

        return (output_mask, output_flags)  # (B, C, T, Hf, Wf), (B, T, F).
