'''
Objective functions.
Created by Basile Van Hoorick for TCOW.
'''

from __init__ import *

# Internal imports.
import metrics
import my_utils


def bootstrap_warmup_loss(loss_pixels, topk_frac):
    topk_num = int(topk_frac * loss_pixels.numel())
    loss_bootstrap = torch.topk(loss_pixels.flatten(), k=topk_num)[0]
    loss_bootstrap = loss_bootstrap.mean()
    return loss_bootstrap


def tversky_loss(output_mask_logits, target_mask, alpha=1.0, beta=1.0, eps=0.1):
    if target_mask.mean() >= 1e-6:
        output_mask_probits = torch.sigmoid(output_mask_logits)
        p0 = output_mask_probits
        p1 = 1.0 - output_mask_probits
        g0 = target_mask
        g1 = 1.0 - target_mask
        numerator = torch.sum(p0 * g0)
        denominator = numerator + alpha * torch.sum(p0 * g1) + beta * torch.sum(p1 * g0)
        loss_jaccard = 1.0 - (numerator / (denominator + eps))
    else:
        loss_jaccard = torch.tensor(0.0, device=output_mask_logits.device)
    return loss_jaccard


class MyLosses():
    '''
    Wrapper around the loss functionality such that DataParallel can be leveraged.
    '''

    def __init__(self, train_args, logger, phase):
        self.train_args = train_args
        self.logger = logger
        self.phase = phase
        self.bce_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.l1_loss = torch.nn.L1Loss(reduction='none')
        self.l2_loss = torch.nn.MSELoss(reduction='none')
        self.gnll_loss = torch.nn.GaussianNLLLoss(reduction='none', eps=1e-3)

    def bce_or_focal_loss(self, x, y):
        if self.train_args.focal_loss:
            return torchvision.ops.focal_loss.sigmoid_focal_loss(x, y, reduction='none')
        else:
            return self.bce_loss(x, y)

    def get_mask_track_frame_weights(self, sel_occl_fracs, query_time):
        '''
        Judges the coarse importance of every frame based on how much the snitch is occluded,
            whether the frame is the query time, and whether we are doing forecasting.
        :param sel_occl_fracs (B, Q, T, 3) tensor.
        :param query_time (int): Query frame index for every example.
        :return frame_weights (B, Q, T) tensor of float in [0, inf).
        '''
        (B, Q, T, _) = sel_occl_fracs.shape
        frame_weights = torch.zeros((B, Q, T), dtype=torch.float32, device=sel_occl_fracs.device)

        for b in range(B):
            for q in range(Q):
                for t in range(T):

                    # We have soft occlusion percentages, so simply scale with desired loss weight.
                    frame_weights[b, q, t] += sel_occl_fracs[b, q, t, 0] * \
                        self.train_args.occluded_weight

        # Ensure all zeros (cases where nothing is special) become one.
        frame_weights = frame_weights.clip(min=1.0)

        # Query frame is easiest to solve (at least for visible pixels), so reduce its importance.
        # NOTE: Since this happens after clipping, the final loss weight value can be < 1.
        frame_weights[b, :, query_time] *= 0.2

        return frame_weights

    def get_mask_track_pixel_weights(self, sel_occl_fracs, target_mask, snitch_occl_by_ptr,
                                     no_hard_negatives=False):
        '''
        Judges the fine-grained importance of every pixel per frame based on class balancing, which
            which snitch pixels are occluded, and other advanced features (such as hard negatives).
            NOTE: This still has to be multiplied with frame_weights by the caller.
        :param sel_occl_fracs (B, Q, T, 3) tensor of float in [0, 1].
        :param target_mask (B, Q, T, H, W) tensor of float in [0, 1].
        :param snitch_occl_by_ptr (B, Q, T, H, W) tensor of uint8 in [0, K].
        :param no_hard_negatives (bool).
        :return pixel_weights (B, Q, T, H, W) tensor of float in [0, inf).
        '''
        (B, Q, T, H, W) = snitch_occl_by_ptr.shape
        pixel_weights = torch.ones((B, Q, T, H, W), dtype=torch.float32,
                                   device=sel_occl_fracs.device)

        # Apply class balancing. NOTE: This happens equally to all videos, objects, and frames
        # within this (sub)batch for efficiency.
        if self.train_args.class_balancing:
            pos_mask = (target_mask == 1.0)
            neg_mask = (target_mask == 0.0)

            pos_frac = pos_mask.sum() / pos_mask.numel()
            neg_frac = neg_mask.sum() / neg_mask.numel()
            pos_frac = pos_frac.clip(min=0.05).item()  # In [0.05, 1.0].
            neg_frac = neg_frac.clip(min=0.05).item()  # In [0.05, 1.0].

            # If both classes are already perfectly balanced, this would result in (0.5, 0.5).
            # However, something like (0.1, 0.9) is more likely, which will result in importance
            # correction factors of (1.93, 0.21) for positive and negative masks respectively.
            if pos_frac > neg_frac:
                # Negative class is currently in the minority (smaller area).
                # Make positive less important: multiply by value in [0.12, 1].
                pos_corr = np.power(neg_frac / pos_frac, 0.7)
                # Make negative more important: multiply by value in [1, 2.46].
                neg_corr = np.power(neg_frac / pos_frac, -0.3)
            else:
                # Positive class is currently in the minority (smaller area).
                # Make positive more important: multiply by value in [1, 2.46].
                pos_corr = np.power(pos_frac / neg_frac, -0.3)
                # Make negative less important: multiply by value in [0.12, 1].
                neg_corr = np.power(pos_frac / neg_frac, 0.7)

            # Apply correction factors on broadcasted loss weight mask.
            pixel_weights[neg_mask] *= neg_corr
            pixel_weights[pos_mask] *= pos_corr

        # Apply double importance for all occluded snitch pixels.
        snitch_occl_mask = (snitch_occl_by_ptr != 0)
        pixel_weights[snitch_occl_mask] *= 2.0

        # Apply hard negatives for amodal completion (i.e. partially occluded cases) only. This
        # means that all pixels spatially close (but not on) the ground truth mask become important.
        if self.train_args.hard_negative_factor > 1.0 and not(no_hard_negatives):
            goldilocks_band = int(np.sqrt(H * W) / 12.0)
            if goldilocks_band % 2 == 0:
                goldilocks_band += 1
            hard_negative_mask = rearrange(torchvision.transforms.functional.gaussian_blur(
                rearrange(target_mask, 'B Q T H W -> (B Q T) H W'),
                kernel_size=goldilocks_band, sigma=goldilocks_band),
                '(B Q T) H W -> B Q T H W', B=B, Q=Q, T=T)
            hard_negative_mask = (hard_negative_mask > 0.0)  # Comes down to enlarging operation.
            hard_negative_mask[target_mask >= 0.5] = False
            pixel_weights[hard_negative_mask] *= self.train_args.hard_negative_factor

        return pixel_weights

    def my_occlusion_flag_loss(self, output_flag, target_flag):
        '''
        :param output_flag (B, Q, T, 1?) tensor.
        :param target_flag (B, Q, T, 1?) tensor.
        '''
        in_frame_mask = (target_flag != 2)
        sel_output = output_flag[in_frame_mask].float()
        sel_target = target_flag[in_frame_mask].float()

        loss_occl_flag = self.bce_loss(sel_output, sel_target)
        loss_occl_flag = loss_occl_flag.mean()

        return loss_occl_flag

    def my_mask_loss(self, output_mask_logits, target_mask, final_weights, progress,
                     apply_weights_for_aot):
        '''
        :param output_mask_logits (B, Q, T, H, W) tensor.
        :param target_mask (B, Q, T, H, W) tensor.
        :param final_weights (B, Q, T, H, W) tensor.
        :param progress (float) in [0, 1]: Total progress within the entire training run.
        :param apply_weights_for_aot (bool): Also use final_weights for AOT loss
            (bootstrapped BCE + soft Jaccard).
        '''
        which_frames = final_weights  # (B, Q, T).
        while which_frames.ndim > 3:
            which_frames = which_frames.any(dim=-1)
        while which_frames.ndim < final_weights.ndim:
            which_frames = which_frames[..., None]
        which_frames = which_frames.expand_as(final_weights)

        if which_frames.any() and final_weights.mean() >= 1e-4:
            # This step causes all losses to skip frames altogether where final_weights is all zero.
            output_mask_logits = output_mask_logits[which_frames]  # (N).
            target_mask = target_mask[which_frames]  # (N).
            final_weights = final_weights[which_frames]  # (N).

            loss_bce = self.bce_or_focal_loss(output_mask_logits, target_mask)
            loss_mask_custom = loss_bce * final_weights
            loss_mask_custom = loss_mask_custom.mean()

            if self.train_args.aot_loss > 0.0:
                if apply_weights_for_aot:
                    loss_bce_for_aot = loss_bce * final_weights
                else:
                    loss_bce_for_aot = loss_bce

                # Inspired by AOT: Bootstrapped BCE loss.
                topk_frac = min(max(1.0 - progress * 8.5, 0.15), 1.0)
                loss_bootstrap = bootstrap_warmup_loss(loss_bce_for_aot, topk_frac)

                # Inspired by AOT: Soft Jaccard (Tversky with alpha = beta = 1) loss.
                if apply_weights_for_aot:
                    loss_jaccard = loss_bootstrap  # Applying final_weights is too complicated.
                else:
                    loss_jaccard = tversky_loss(output_mask_logits, target_mask,
                                                alpha=1.0, beta=1.0, eps=0.1)

                loss_aot = (loss_bootstrap + loss_jaccard) / 2.0
                loss_mask = loss_aot * self.train_args.aot_loss + \
                    loss_mask_custom * (1.0 - self.train_args.aot_loss)

            else:
                loss_mask = loss_mask_custom

            # Avoid overweighting extreme cases, such as a single frame only.
            loss_mask = loss_mask * torch.sqrt(which_frames.float().mean())

        else:
            loss_mask = torch.tensor(0.0, device=output_mask_logits.device)

        # Sanity checks.
        assert (target_mask * final_weights >= 0.0).all(), \
            'target_mask * final_weights should only contain non-negative values.'

        return loss_mask

    def per_example(self, data_retval, model_retval, progress, metrics_only):
        '''
        Loss calculations that *can* be performed independently for each example within a batch.
        :param data_retval (dict): Data loader elements.
        :param model_retval (dict): All network output information.
        :param progress (float) in [0, 1]: Total progress within the entire training run.
        :param metrics_only (bool).
        :return loss_retval (dict): Preliminary loss information.
        '''
        return self.per_example_mask_track(data_retval, model_retval, progress, metrics_only)

    def per_example_mask_track(self, data_retval, model_retval, progress, metrics_only):
        if metrics_only:
            # Calculate only evaluation metrics and nothing else.
            metrics_retval = metrics.calculate_metrics_mask_track(data_retval, model_retval)
            loss_retval = dict()
            loss_retval['metrics'] = metrics_retval
            return loss_retval

        # Evaluate entire subbatch for efficiency.
        # NOTE: Take care to ensure output_mask is always logits (not probits) leading up to here.
        source_name = data_retval['source_name'][0]
        if source_name == 'kubric':
            (B, C, T, Hf, Wf) = data_retval['kubric_retval']['pv_rgb_tf'].shape
            query_time = data_retval['kubric_retval']['traject_retval_tf']['query_time'][0].item()
        target_mask = model_retval['target_mask']  # (B, Q, 1/3, T, Hf, Wf).
        output_mask = model_retval['output_mask']  # (B, Q, 1/3, T, Hf, Wf).
        Q = target_mask.shape[1]

        if source_name == 'kubric':
            sel_occl_fracs = model_retval['sel_occl_fracs']  # (B, Q, T, 3) with (f, v, t).
            snitch_occl_by_ptr = model_retval['snitch_occl_by_ptr']  # (B, Q, 1, T, Hf, Wf).

        snitch_weights = None
        loss_track = None
        loss_occl_mask = None
        loss_cont_mask = None

        if self.train_args.track_lw > 0.0:
            if source_name == 'kubric':
                snitch_frame_weights = self.get_mask_track_frame_weights(
                    sel_occl_fracs, query_time)
                # (B, Q, T).
                snitch_pixel_weights = self.get_mask_track_pixel_weights(
                    sel_occl_fracs, target_mask[:, :, 0], snitch_occl_by_ptr[:, :, 0])
                # (B, Q, T, Hf, Wf).
                snitch_weights = snitch_frame_weights[..., None, None] * snitch_pixel_weights
                # (B, Q, T, Hf, Wf).

                loss_track = self.my_mask_loss(
                    output_mask[:, :, 0], target_mask[:, :, 0], snitch_weights, progress, False)

            else:
                raise NotImplementedError()

        if self.train_args.occl_mask_lw > 0.0 and source_name == 'kubric':
            # We must supervise and average only the relevant frames. Target frontmost channel will
            # simply be all zero when there exists no appropriate occluder.
            occl_mask_weights = target_mask[:, :, 1].any(dim=-1).any(dim=-1)  # (B, Q, T).
            occl_mask_weights = occl_mask_weights[..., None, None].expand_as(target_mask[:, :, 1])
            occl_mask_weights = occl_mask_weights.type(torch.float32)
            # (B, Q, T, Hf, Wf).

            # Still gently encourage all-zero in case no full occlusion is occurring.
            occl_mask_weights = occl_mask_weights * (1.0 - self.train_args.occl_cont_zero_weight) + \
                self.train_args.occl_cont_zero_weight

            # NOTE: We explicitly apply frame weights for AOT losses here, as the lack of occluder
            # or container should a priori be considered significantly less important.
            loss_occl_mask = self.my_mask_loss(
                output_mask[:, :, 1], target_mask[:, :, 1], occl_mask_weights, progress, True)

        if self.train_args.cont_mask_lw > 0.0 and source_name == 'kubric':
            # We must supervise and average only the relevant frames. Target outermost channel will
            # simply be all zero when there exists no appropriate container.
            cont_mask_weights = target_mask[:, :, 2].any(dim=-1).any(dim=-1)  # (B, Q, T).
            cont_mask_weights = cont_mask_weights[..., None, None].expand_as(target_mask[:, :, 1])
            cont_mask_weights = cont_mask_weights.type(torch.float32)  # (B, Q, T, Hf, Wf).

            # Still gently encourage all-zero in case no full containment is occurring.
            cont_mask_weights = cont_mask_weights * (1.0 - self.train_args.occl_cont_zero_weight) + \
                self.train_args.occl_cont_zero_weight

            # NOTE: We explicitly apply frame weights for AOT losses here, as the lack of occluder
            # or container should a priori be considered significantly less important.
            loss_cont_mask = self.my_mask_loss(
                output_mask[:, :, 2], target_mask[:, :, 2], cont_mask_weights, progress, True)

        # Calculate preliminary evaluation metrics.
        metrics_retval = metrics.calculate_metrics_mask_track(data_retval, model_retval)

        # Save calculated per-pixel loss weights for visualization / debugging.
        if snitch_weights is not None:
            model_retval['snitch_weights'] = snitch_weights

        # Return results.
        loss_retval = dict()
        loss_retval['track'] = loss_track
        loss_retval['occl_mask'] = loss_occl_mask
        loss_retval['cont_mask'] = loss_cont_mask
        loss_retval['metrics'] = metrics_retval

        return loss_retval

    def entire_batch(self, data_retval, model_retval, loss_retval, cur_step, total_step, epoch,
                     progress):
        '''
        Loss calculations that *cannot* be performed independently for each example within a batch.
        :param data_retval (dict): Data loader elements.
        :param model_retval (dict): All network output information.
        :param loss_retval (dict): Preliminary loss information (per-example, but not batch-wide).
        :param progress (float) in [0, 1]: Total progress within the entire training run.
        :return loss_retval (dict): All loss information.
        '''
        # For debugging:
        old_loss_retval = loss_retval.copy()
        old_loss_retval['metrics'] = loss_retval['metrics'].copy()

        if not('test' in self.phase):
            # Log average value per epoch at train / val time.
            key_prefix = self.phase + '/'
            report_kwargs = dict(remember=True)
        else:
            # Log & plot every single step at test time.
            key_prefix = ''
            report_kwargs = dict(step=cur_step)

        if len(loss_retval.keys()) >= 2:  # Otherwise, assume we had metrics_only enabled.

            # Average all loss values across batch size.
            for (k, v) in loss_retval.items():
                if not('metrics' in k):
                    if torch.is_tensor(v):
                        loss_retval[k] = torch.mean(v)
                    elif v is None:
                        loss_retval[k] = -1.0
                    else:
                        raise RuntimeError(f'loss_retval: {k}: {v}')

            # Obtain total loss per network.
            loss_total_seeker = loss_retval['track'] * self.train_args.track_lw + \
                loss_retval['occl_mask'] * self.train_args.occl_mask_lw + \
                loss_retval['cont_mask'] * self.train_args.cont_mask_lw

            # Convert loss terms (just not the total) to floats for logging.
            for (k, v) in loss_retval.items():
                if torch.is_tensor(v):
                    loss_retval[k] = v.item()

            # Report all loss values.
            self.logger.report_scalar(
                key_prefix + 'loss_total_seeker', loss_total_seeker.item(), **report_kwargs)
            if self.train_args.track_lw > 0.0:
                self.logger.report_scalar(
                    key_prefix + 'loss_track', loss_retval['track'], **report_kwargs)
            if self.train_args.occl_mask_lw > 0.0:
                self.logger.report_scalar(
                    key_prefix + 'loss_occl_mask', loss_retval['occl_mask'], **report_kwargs)
            if self.train_args.cont_mask_lw > 0.0:
                self.logger.report_scalar(
                    key_prefix + 'loss_cont_mask', loss_retval['cont_mask'], **report_kwargs)

            # Return results, i.e. append new stuff to the existing loss_retval dictionary.
            # Total losses are the only entries that are tensors, not just floats.
            # Later in train.py, we will match the appropriate optimizer (and thus network parameter
            # updates) to each accumulated loss value as indicated by the keys here.
            loss_retval['total_seeker'] = loss_total_seeker

        # Weighted average all metrics across batch size.
        # DRY: This is also in metrics.py.
        for (k, v) in loss_retval['metrics'].items():
            if 'count' in k:
                count_key = k
                mean_key = k.replace('count', 'mean')
                short_key = k.replace('count_', '')
                old_counts = loss_retval['metrics'][count_key]
                old_means = loss_retval['metrics'][mean_key]

                # NOTE: Some mean values will be -1.0 but then corresponding counts are always 0.
                new_count = old_counts.sum().item()
                if new_count > 0:
                    new_mean = (old_means * old_counts).sum().item() / (new_count + 1e-7)
                else:
                    new_mean = -1.0
                loss_retval['metrics'][count_key] = new_count
                loss_retval['metrics'][mean_key] = new_mean

                # Report all metrics, but ignore invalid values (e.g. when no occluded or contained
                # stuff exists). At train time, we maintain correct proportions with the weight
                # option. At test time, we log every step anyway, so this does not matter.
                if new_count > 0:
                    self.logger.report_scalar(key_prefix + short_key, new_mean, weight=new_count,
                                              **report_kwargs)

        return loss_retval
