'''
Entire training pipeline logic.
Created by Basile Van Hoorick for TCOW.
'''

from __init__ import *

# Internal imports.
import data_kubric
import data_utils
import loss
import my_utils


class MyTrainPipeline(torch.nn.Module):
    '''
    Wrapper around most of the training iteration such that DataParallel can be leveraged.
    '''

    def __init__(self, train_args, logger, networks, device):
        super().__init__()
        self.train_args = train_args
        self.logger = logger
        self.networks = torch.nn.ModuleDict(networks)
        self.device = device
        self.phase = None  # Assigned only by set_phase().
        self.losses = None  # Instantiated only by set_phase().
        # self.kubric_generators = [None for _ in range(train_args.batch_size)]
        self.to_tensor = torchvision.transforms.ToTensor()

    def set_phase(self, phase):
        '''
        Must be called when switching between train / validation / test phases.
        '''
        self.phase = phase
        self.losses = loss.MyLosses(self.train_args, self.logger, phase)

        if 'train' in phase:
            self.train()
            for (k, v) in self.networks.items():
                v.train()
            torch.set_grad_enabled(True)

        else:
            self.eval()
            for (k, v) in self.networks.items():
                v.eval()
            torch.set_grad_enabled(False)

    def forward(self, data_retval, cur_step, total_step, epoch, progress, include_loss,
                metrics_only):
        '''
        Handles one parallel iteration of the training or validation phase.
        Executes the models and calculates the per-example losses.
        This is all done in a parallelized manner to minimize unnecessary communication.
        :param data_retval (dict): Data loader elements.
        :param cur_step (int): Current data loader index.
        :param total_step (int): Cumulative data loader index, including all previous epochs.
        :param epoch (int): Current epoch index (0-based).
        :param progress (float) in [0, 1]: Total progress within the entire training run.
        :param include_loss (bool).
        :param metrics_only (bool).
        :return (model_retval, loss_retval).
            model_retval (dict): All output information.
            loss_retval (dict): Preliminary loss information (per-example, but not batch-wide).
        '''
        # Proceed with source-specific forward logic.
        source_name = data_retval['source_name'][0]
        assert all([x == source_name for x in data_retval['source_name']]), \
            'Cannot mix sources within one batch.'

        if source_name == 'kubric':
            model_retval = self.forward_kubric(data_retval)

        elif source_name == 'plugin':
            model_retval = self.forward_plugin(data_retval)

        if include_loss:
            loss_retval = self.losses.per_example(data_retval, model_retval, progress, metrics_only)
        else:
            loss_retval = None

        return (model_retval, loss_retval)

    def forward_kubric(self, data_retval):
        within_batch_inds = data_retval['within_batch_idx']
        B = within_batch_inds.shape[0]

        # Retrieve data.
        # NOTE: Any array with a dimension M will typically have trailing zeros.
        kubric_retval = data_retval['kubric_retval']
        all_rgb = kubric_retval['pv_rgb_tf']
        # (B, 3, T, Hf, Wf).
        all_segm = kubric_retval['pv_segm_tf']
        # (B, 1, T, Hf, Wf).
        all_div_segm = kubric_retval['pv_div_segm_tf']
        # (B, M, T, Hf, Wf).
        all_rgb = all_rgb.to(self.device)
        all_segm = all_segm.to(self.device)
        all_div_segm = all_div_segm.to(self.device)
        inst_count = kubric_retval['pv_inst_count']
        # (B, 1); acts as Qt value per example.
        query_time = kubric_retval['traject_retval_tf']['query_time']
        # (B).
        occl_fracs = kubric_retval['traject_retval_tf']['occl_fracs_tf']
        # (B, M, T, 3) with (f, v, t).
        occl_cont_dag = kubric_retval['traject_retval_tf']['occl_cont_dag_tf']
        # (B, T, M, M, 3) with (c, od, of).
        # NOTE: ^ Based on non-cropped data.
        target_desirability = kubric_retval['traject_retval_tf']['desirability_tf']
        # (B, M, 7).
        scene_dp = data_retval['scene_dp']

        (T, H, W) = all_rgb.shape[-3:]
        assert T == self.train_args.num_frames
        Qs = self.train_args.num_queries  # Selected per example here.

        # Assemble seeker input (which is always a simple copy now).
        seeker_input = all_rgb  # (B, 3, T, Hf, Wf).

        # Sample either random or biased queries.
        sel_query_inds = my_utils.sample_query_inds(
            B, Qs, inst_count, target_desirability, self.train_args, self.device, self.phase)

        # Loop over selected queries and accumulate outputs & targets.
        all_occl_fracs = []
        all_desirability = []
        all_seeker_query_mask = []
        all_snitch_occl_by_ptr = []
        all_full_occl_cont_id = []
        all_target_mask = []
        all_output_mask = []

        for q in range(Qs):

            # Get query info.
            # NOTE: query_idx is still a (B) tensor, so don't forget to select index.
            # query_idx[b] refers directly to the snitch instance ID we are tracking.
            query_idx = sel_query_inds[:, q]  # (B).
            qt_idx = query_time[0].item()

            # Prepare query mask and ground truths.
            (seeker_query_mask, snitch_occl_by_ptr, full_occl_cont_id, target_mask,
                target_flags) = data_utils.fill_kubric_query_target_mask_flags(
                all_segm, all_div_segm, query_idx, qt_idx, occl_fracs, occl_cont_dag, scene_dp,
                self.logger, self.train_args, self.device, self.phase)

            # Sanity checks.
            if not seeker_query_mask.any():
                raise RuntimeError(
                    f'seeker_query_mask all zero? q: {q} query_idx: {query_idx} qt_idx: {qt_idx}')
            if not target_mask.any():
                raise RuntimeError(
                    f'target_mask all zero? q: {q} query_idx: {query_idx} qt_idx: {qt_idx}')

            # Run seeker to recover hierarchical masks over time.
            (output_mask, output_flags) = self.networks['seeker'](
                seeker_input, seeker_query_mask)  # (B, 3, T, Hf, Wf), (B, T, 3).

            # Save some ground truth metadata, e.g. weighted query desirability, to get a feel for
            # this example or dataset.
            # NOTE: diagonal() appends the combined dimension at the END of the shape.
            # https://pytorch.org/docs/stable/generated/torch.diagonal.html
            cur_occl_fracs = occl_fracs[:, query_idx, :, :].diagonal(0, 0, 1)
            cur_occl_fracs = rearrange(cur_occl_fracs, 'T V B -> B T V')  # (B, T, 3).
            cur_desirability = target_desirability[:, query_idx, 0].diagonal(0, 0, 1)  # (B).

            all_occl_fracs.append(cur_occl_fracs)  # (B, T, 3).
            all_desirability.append(cur_desirability)  # (B).
            all_seeker_query_mask.append(seeker_query_mask)  # (B, 1, T, Hf, Wf).
            all_snitch_occl_by_ptr.append(snitch_occl_by_ptr)  # (B, 1, T, Hf, Wf).
            all_full_occl_cont_id.append(full_occl_cont_id)  # (B, T, 2).
            all_target_mask.append(target_mask)  # (B, 3, T, Hf, Wf).
            all_output_mask.append(output_mask)  # (B, 1/3, T, Hf, Wf).

        sel_occl_fracs = torch.stack(all_occl_fracs, dim=1)  # (B, Qs, T, 3).
        sel_desirability = torch.stack(all_desirability, dim=1)  # (B, Qs).
        seeker_query_mask = torch.stack(all_seeker_query_mask, dim=1)  # (B, Qs, 1, T, Hf, Wf).
        snitch_occl_by_ptr = torch.stack(all_snitch_occl_by_ptr, dim=1)  # (B, Qs, 1, T, Hf, Wf).
        full_occl_cont_id = torch.stack(all_full_occl_cont_id, dim=1)  # (B, Qs, T, 2).
        target_mask = torch.stack(all_target_mask, dim=1)  # (B, Qs, 3, T, Hf, Wf).
        output_mask = torch.stack(all_output_mask, dim=1)  # (B, Qs, 1/3, T, Hf, Wf).

        # Organize & return relevant info.
        # Ensure that everything is on a CUDA device.
        model_retval = dict()
        model_retval['sel_query_inds'] = sel_query_inds.to(self.device)  # (B, Qs).
        model_retval['sel_occl_fracs'] = sel_occl_fracs.to(self.device)  # (B, Qs, T, 3).
        model_retval['sel_desirability'] = sel_desirability.to(self.device)  # (B, Qs).
        model_retval['seeker_input'] = seeker_input.to(self.device)  # (B, 3, T, Hf, Wf).
        model_retval['seeker_query_mask'] = seeker_query_mask.to(self.device)
        # (B, Qs, 1, T, Hf, Wf).
        model_retval['snitch_occl_by_ptr'] = snitch_occl_by_ptr.to(self.device)
        # (B, Qs, 1, T, Hf, Wf).
        model_retval['full_occl_cont_id'] = full_occl_cont_id.to(self.device)
        # (B, Qs, 1, T, Hf, Wf).
        model_retval['target_mask'] = target_mask.to(self.device)  # (B, Qs, 3, T, Hf, Wf).
        model_retval['output_mask'] = output_mask.to(self.device)  # (B, Qs, 1/3, T, Hf, Wf).

        return model_retval

    def forward_plugin(self, data_retval):
        # DRY: This is mostly a simplified version of forward_kubric().
        within_batch_inds = data_retval['within_batch_idx']
        B = within_batch_inds.shape[0]

        all_rgb = data_retval['pv_rgb_tf']  # (B, 3, T, Hf, Wf).
        all_query = data_retval['pv_query_tf']  # (B, 1, T, Hf, Wf).
        all_target = data_retval['pv_target_tf']  # (B, 3, T, Hf, Wf).
        all_rgb = all_rgb.to(self.device)
        all_query = all_query.to(self.device)
        all_target = all_target.to(self.device)

        (T, H, W) = all_rgb.shape[-3:]
        assert T == self.train_args.num_frames
        Qs = 1

        # Assemble seeker input (which is always a simple copy in this case).
        seeker_input = all_rgb  # (B, 3, T, Hf, Wf).
        seeker_query_mask = all_query.type(torch.float32).to(self.device)  # (B, 1, T, Hf, Wf).
        target_mask = all_target.type(torch.float32).to(self.device)  # (B, 3, T, Hf, Wf).

        # Sanity checks.
        if not seeker_query_mask.any():
            raise RuntimeError(f'seeker_query_mask all zero?')

        # Run seeker to recover hierarchical masks over time.
        (output_mask, output_flags) = self.networks['seeker'](
            seeker_input, seeker_query_mask)  # (B, 3, T, Hf, Wf), (B, T, 3).

        # Organize & return relevant info.
        # Ensure that everything is on a CUDA device.
        model_retval = dict()
        model_retval['seeker_input'] = seeker_input.to(self.device)  # (B, 3, T, Hf, Wf).
        model_retval['seeker_query_mask'] = seeker_query_mask.to(self.device)  # (B, 1, T, Hf, Wf).
        model_retval['target_mask'] = target_mask.to(self.device)  # (B, 3, T, Hf, Wf).
        model_retval['output_mask'] = output_mask.to(self.device)  # (B, 3, T, Hf, Wf).
        model_retval['output_flags'] = output_flags.to(self.device)  # (B, T, 3).

        return model_retval

    def process_entire_batch(self, data_retval, model_retval, loss_retval, cur_step, total_step,
                             epoch, progress):
        '''
        Finalizes the training step. Calculates all losses.
        :param data_retval (dict): Data loader elements.
        :param model_retval (dict): All network output information.
        :param loss_retval (dict): Preliminary loss information (per-example, but not batch-wide).
        :param cur_step (int): Current data loader index.
        :param total_step (int): Cumulative data loader index, including all previous epochs.
        :param epoch (int): Current epoch index (0-based).
        :param progress (float) in [0, 1]: Total progress within the entire training run.
        :return loss_retval (dict): All loss information.
        '''
        loss_retval = self.losses.entire_batch(
            data_retval, model_retval, loss_retval, cur_step, total_step, epoch, progress)

        return loss_retval
