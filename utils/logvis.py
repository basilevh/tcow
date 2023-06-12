'''
Logging and visualization logic.
Created by Basile Van Hoorick for TCOW.
'''

import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), 'utils/'))
sys.path.insert(0, os.getcwd())

from __init__ import *

# Internal imports.
import logvisgen
import visualization


class MyLogger(logvisgen.Logger):
    '''
    Adapts the generic logger to this specific project.
    '''

    def __init__(self, args, context, log_level=None):
        if 'batch_size' in args:
            if args.is_debug:
                self.step_interval = max(16 // args.batch_size, 2)
            else:
                self.step_interval = max(64 // args.batch_size, 2)
        else:
            if args.is_debug:
                self.step_interval = 4
            else:
                self.step_interval = 16
        self.half_step_interval = self.step_interval // 2

        super().__init__(log_dir=args.log_path, context=context, log_level=log_level)

    def handle_train_step(self, epoch, phase, cur_step, total_step, steps_per_epoch,
                          data_retval, model_retval, loss_retval, train_args, test_args):

        if not(('train' in phase and cur_step % self.step_interval == 0) or
               ('val' in phase and cur_step % self.half_step_interval == 0) or
                ('test' in phase)):
            return

        file_name_suffix = self.handle_train_step_mask_track(
            epoch, phase, cur_step, total_step, steps_per_epoch,
            data_retval, model_retval, loss_retval, train_args, test_args)

        return file_name_suffix

    def handle_train_step_mask_track(self, epoch, phase, cur_step, total_step, steps_per_epoch,
                                     data_retval, model_retval, loss_retval, train_args, test_args):
        source_name = data_retval['source_name'][0]
        scene_idx = data_retval['scene_idx'][0].item()
        if source_name == 'kubric':
            kubric_retval = data_retval['kubric_retval']
            frame_first = kubric_retval['frame_inds_load'][0, 0].item()
            frame_last = kubric_retval['frame_inds_load'][0, -1].item()

        # Obtain friendly short name for this step for logging, video saving, and CSV export.
        if not('test' in phase):
            file_name_suffix = ''
            file_name_suffix += f'e{epoch}_p{phase}_s{cur_step}_{source_name[:2]}_d{scene_idx}'

            if source_name == 'kubric':
                file_name_suffix += f'_f{frame_first}_l{frame_last}'
                if kubric_retval['augs_params']['reverse'][0]:
                    file_name_suffix += '_rev'
                if kubric_retval['augs_params']['palindrome'][0]:
                    file_name_suffix += '_pal'

        else:
            # NOTE: Here, we are now also exporting itemized results to CSV files.
            file_name_suffix = ''
            if source_name == 'plugin':
                plugin_name = str(pathlib.Path(data_retval['src_path'][0]).name).split('.')[0]
                frame_start = data_retval['frame_start'][0].item()
                frame_stride = data_retval['frame_stride'][0].item()
                file_name_suffix += f'{plugin_name}_i{frame_stride}_f{frame_start}_s{cur_step}'
            else:
                file_name_suffix += f's{cur_step}_{source_name[:2]}_d{scene_idx}'
                if source_name == 'kubric':
                    file_name_suffix += f'_f{frame_first}_l{frame_last}'

        # Log informative line including loss values & metrics in console.
        to_print = f'[Step {cur_step} / {steps_per_epoch}]  {source_name}  scn: {scene_idx}  '

        if source_name == 'plugin':
            to_print += f'name: {plugin_name}  f_stride: {frame_stride}  f_start: {frame_start}  '

        # NOTE: All wandb stuff for reporting scalars is handled in loss.py.
        # Assume loss may be missing (e.g. at test time).
        if loss_retval is not None:

            if len(loss_retval.keys()) >= 2:
                total_loss_seeker = loss_retval['total_seeker'].item()
                loss_track = loss_retval['track']
                loss_occl_mask = loss_retval['occl_mask']
                loss_cont_mask = loss_retval['cont_mask']
                to_print += (f'tot: {total_loss_seeker:.3f}  '
                             f'sn_t: {loss_track:.3f}  '
                             f'fo_t: {loss_occl_mask:.3f}  '
                             f'oc_t: {loss_cont_mask:.3f}  ')

            # Assume metrics are always present (even if count = 0).
            metrics_retval = loss_retval['metrics']
            snitch_iou = metrics_retval['mean_snitch_iou']
            occl_mask_iou = metrics_retval['mean_occl_mask_iou']
            cont_mask_iou = metrics_retval['mean_cont_mask_iou']
            to_print += (f'sn_iou: {snitch_iou:.3f}  '
                         f'fo_iou: {occl_mask_iou:.3f}  '
                         f'oc_iou: {cont_mask_iou:.3f}  ')

        self.info(to_print)

        log_rarely = 0 if 'test' in phase else train_args.log_rarely
        if log_rarely > 0 and cur_step % (self.step_interval * 16) != self.step_interval * 8:
            return file_name_suffix

        target_mask = None
        snitch_weights = None

        # Save input, prediction, and ground truth data.
        if source_name == 'kubric':
            all_rgb = rearrange(kubric_retval['pv_rgb_tf'][0],
                                'C T H W -> T H W C').detach().cpu().numpy()
        else:
            all_rgb = rearrange(data_retval['pv_rgb_tf'][0],
                                'C T H W -> T H W C').detach().cpu().numpy()
        # (T, H, W, 3).
        if 'seeker_input' in model_retval:
            seeker_rgb = rearrange(model_retval['seeker_input'][0, 0:3],
                                'C T H W -> T H W C').detach().cpu().numpy()
        else:
            seeker_rgb = all_rgb
        # (T, H, W, 3).
        seeker_query_mask = model_retval['seeker_query_mask'][0].detach().cpu().numpy()
        # (Qs, 1, T, H, W).
        output_mask = model_retval['output_mask'][0].sigmoid().detach().cpu().numpy()
        # (Qs, 3, T, H, W).
        target_mask = model_retval['target_mask'][0].detach().cpu().numpy()
        # (Qs, 3, T, H, W).
        if 'snitch_weights' in model_retval:
            snitch_weights = model_retval['snitch_weights'][0].detach().cpu().numpy()
        # (Qs, 1, T, H, W).

        frame_rate = train_args.kubric_frame_rate // train_args.kubric_frame_stride

        if source_name == 'plugin':
            # Add fake query count dimension (Qs = 1).
            seeker_query_mask = seeker_query_mask[None]  # Add fake query count dimension (Qs = 1).
            output_mask = output_mask[None]
            target_mask = target_mask[None]  # Add fake query count dimension (Qs = 1).

            # We want to slow down plugin videos according to how much we are subsampling them
            # temporally for the model, but not too drastically.
            frame_stride = data_retval['frame_stride'][0].item()
            used_frame_stride = (frame_stride +
                                    test_args.plugin_prefer_frame_stride) / 2.5
            frame_rate = int(round(test_args.plugin_frame_rate / used_frame_stride))

        dimmed_rgb = (all_rgb + seeker_rgb) / 2.0  # Indicates when input becomes black.
        # (T, H, W, 3).
        if output_mask is not None:
            (Qs, Cmo) = output_mask.shape[:2]
        else:
            Cmo = 0
        if target_mask is not None:
            (Qs, Cmt) = target_mask.shape[:2]
        else:
            Cmt = 0
        # NOTE: Typically, Cm == 3.

        # Superimpose input video, predicted mask, and borders of query & ground truth masks.
        for q in range(Qs):

            # Construct query & target outlines.
            query_border = visualization.draw_segm_borders(
                seeker_query_mask[q, 0][..., None], fill_white=False)  # (T, H, W).
            snitch_border = visualization.draw_segm_borders(
                target_mask[q, 0][..., None], fill_white=False) \
                if Cmt >= 1 else np.zeros_like(output_mask[q, 0], dtype=np.bool)  # (T, H, W).
            frontmost_border = visualization.draw_segm_borders(
                target_mask[q, 1][..., None], fill_white=False) \
                if Cmt >= 2 else np.zeros_like(output_mask[q, 0], dtype=np.bool)  # (T, H, W).
            outermost_border = visualization.draw_segm_borders(
                target_mask[q, 2][..., None], fill_white=False) \
                if Cmt >= 3 else np.zeros_like(output_mask[q, 0], dtype=np.bool)  # (T, H, W).

            # Draw annotated model input.
            # NOTE: This exact part sometimes takes ~10s!
            vis_input = visualization.create_model_input_video(
                dimmed_rgb, seeker_query_mask[q, 0], query_border)
            # (T, H, W, 3).

            # Draw annotated model output at snitch level.
            # NOTE: This exact part sometimes takes ~10-70s!
            vis_snitch = visualization.create_model_output_snitch_video(
                seeker_rgb, output_mask[q], query_border, snitch_border, grayscale=False)
            # (T, H, W, 3).

            # Draw annotated model output at snitch + frontmost + outermost levels.
            vis_allout = visualization.create_model_output_snitch_occl_cont_video(
                seeker_rgb, output_mask[q], query_border, snitch_border, frontmost_border,
                outermost_border, grayscale=True)
            # (T, H, W, 3).

            # Draw detailed snitch mask per-pixel loss weights for visual debugging.
            if snitch_weights is not None and not('test' in phase):
                vis_slw = visualization.create_snitch_weights_video(seeker_rgb, snitch_weights[q])
                # (T, H, W, 3).

            vis_intgt = None
            if 'test' in phase or ('is_figs' in train_args and train_args.is_figs):
                vis_intgt = visualization.create_model_input_target_video(
                    seeker_rgb, seeker_query_mask[q, 0], target_mask[q], query_border,
                    snitch_border, frontmost_border, outermost_border, grayscale=False)

            vis_extra = []
            if ('test' in phase and test_args.extra_visuals) or \
                    ('is_figs' in train_args and train_args.is_figs):

                # Include raw masks mapped directly as RGB channels without any input video.
                vis_extra.append(np.stack(
                    [target_mask[q, 1], target_mask[q, 0], target_mask[q, 2]], axis=-1))
                vis_extra.append(np.stack(
                    [output_mask[q, 1], output_mask[q, 0], output_mask[q, 2]], axis=-1))

                # Include temporally concatenated & spatially horizontally concatenated versions of
                # (input) + (output + target) or (input + target) + (output + target).
                vis_allout_pause = np.concatenate([vis_allout[0:1]] * 3 + [vis_allout[1:]], axis=0)
                vis_intgt_pause = np.concatenate([vis_intgt[0:1]] * 3 + [vis_intgt[1:]], axis=0)
                vis_extra.append(np.concatenate([vis_input, vis_allout], axis=0))  # (T, H, W, 3).
                vis_extra.append(np.concatenate([vis_intgt_pause, vis_allout], axis=0))  # (T, H, W, 3).
                vis_extra.append(np.concatenate([vis_input, vis_allout_pause], axis=2))  # (T, H, W, 3).
                vis_extra.append(np.concatenate([vis_intgt_pause, vis_allout_pause], axis=2))  # (T, H, W, 3).

            file_name_suffix_q = file_name_suffix + f'_q{q}'
            # Easily distinguish all-zero outputs.
            # file_name_suffix_q += f'_mo{output_mask.mean():.3f}'

            if not('test' in phase):
                wandb_step = epoch
                accumulate_online = 8
                extensions = ['.webm']
            else:
                wandb_step = cur_step
                accumulate_online = 1
                # extensions = ['.gif', '.webm']
                extensions = ['.webm']

            # NOTE: Without apply_async, this part would take by far the most time.
            avoid_wandb = test_args.avoid_wandb if 'test' in phase else train_args.avoid_wandb
            online_name = f'in_p{phase}' if avoid_wandb == 0 else None
            self.save_video(vis_input, step=wandb_step,
                            file_name=f'more/{file_name_suffix_q}_in',
                            online_name=online_name,
                            accumulate_online=accumulate_online,
                            caption=f'{file_name_suffix_q}',
                            extensions=extensions, fps=frame_rate // 2,
                            upscale_factor=2, apply_async=True)
            online_name = f'out_p{phase}_sn' if avoid_wandb == 0 else None
            self.save_video(vis_snitch, step=wandb_step,
                            file_name=f'more/{file_name_suffix_q}_out_sn',
                            online_name=online_name,
                            accumulate_online=accumulate_online,
                            caption=f'{file_name_suffix_q}',
                            extensions=extensions, fps=frame_rate // 2,
                            upscale_factor=2, apply_async=True)
            if Cmo >= 3:
                online_name = f'out_p{phase}_oc' if avoid_wandb == 0 else None
                self.save_video(vis_allout, step=wandb_step,
                                file_name=f'{file_name_suffix_q}_out_oc',
                                online_name=online_name,
                                accumulate_online=accumulate_online,
                                caption=f'{file_name_suffix_q}',
                                extensions=extensions, fps=frame_rate // 2,
                                upscale_factor=2, apply_async=True)
            if snitch_weights is not None and not('test' in phase):
                online_name = None
                self.save_video(vis_slw, step=wandb_step,
                                file_name=f'more/{file_name_suffix_q}_slw',
                                online_name=online_name,
                                accumulate_online=accumulate_online,
                                caption=f'{file_name_suffix_q}',
                                extensions=extensions, fps=frame_rate // 2,
                                upscale_factor=2, apply_async=True)

            if vis_intgt is not None:
                online_name = None
                self.save_video(vis_intgt, step=wandb_step,
                                file_name=f'more/{file_name_suffix_q}_tgt',
                                online_name=online_name,
                                accumulate_online=accumulate_online,
                                caption=f'{file_name_suffix_q}',
                                extensions=extensions, fps=frame_rate // 2,
                                upscale_factor=2, apply_async=True)

            if len(vis_extra) != 0:
                for i, vis in enumerate(vis_extra):
                    self.save_video(vis, step=wandb_step,
                                    file_name=f'more/{file_name_suffix_q}_extra{i}',
                                    online_name=online_name,
                                    accumulate_online=accumulate_online,
                                    caption=f'{file_name_suffix_q}',
                                    extensions=extensions, fps=frame_rate // 2,
                                    upscale_factor=2, apply_async=True)

        return file_name_suffix

    def epoch_finished(self, epoch):
        super().epoch_finished(epoch)
        self.commit_scalars(step=epoch)

    def handle_test_step(self, cur_step, num_steps, data_retval, inference_retval, all_args):
        '''
        :param all_args (dict): train, test, train_dset, test_dset, model.
        '''

        model_retval = inference_retval['model_retval']
        loss_retval = inference_retval['loss_retval']

        file_name_suffix = self.handle_train_step(
            0, 'test', cur_step, cur_step, num_steps, data_retval, model_retval, loss_retval,
            all_args['train'], all_args['test'])

        return file_name_suffix
