'''
Helper methods for quantitative analyses.
Created by Basile Van Hoorick for TCOW.
'''

from __init__ import *


def calculate_metrics_mask_track(data_retval, model_retval):
    '''
    Calculates (preliminary) useful quantitative results for this (sub-)batch. All of these
        numbers are just for information and not for backpropagation.
    :param data_retval (dict).
    :param model_retval (dict).
    '''
    # NOTE: Take care to ensure output_mask is always logits (not probits) leading up to here.
    source_name = data_retval['source_name'][0]
    output_mask_binary = (model_retval['output_mask'] > 0.0).bool()  # (B, Q?, 1/3, T, Hf, Wf).
    target_mask_binary = (model_retval['target_mask'] > 0.5).bool()  # (B, Q?, 1/3, T, Hf, Wf).

    # Find out which frames either have no annotations, or are too risky.
    ignore_frames = (model_retval['target_mask'] < 0.0).any(dim=-1).any(dim=-1)  # (B, Q?, 1/3, T).

    # Add fake query count dimension (Qs = 1) if needed.
    if source_name == 'plugin':
        output_mask_binary = output_mask_binary[:, None]
        target_mask_binary = target_mask_binary[:, None]
        ignore_frames = ignore_frames[:, None]

    # NOTE: These operations will automatically broadcast the smaller to the larger array.
    intersection_binary = torch.logical_and(output_mask_binary, target_mask_binary)
    union_binary = torch.logical_or(output_mask_binary, target_mask_binary)
    (B, Q, Cmo, T, Hf, Wf) = output_mask_binary.shape
    (B, Q, Cmt, T, Hf, Wf) = target_mask_binary.shape
    # NOTE: Typically, Cm == 3.

    target_mask_areas = target_mask_binary.sum(dim=-1).sum(dim=-1)  # (B, Q, 3, T).
    intersection_areas = intersection_binary.sum(dim=-1).sum(dim=-1)  # (B, Q, 3, T).
    union_areas = union_binary.sum(dim=-1).sum(dim=-1)  # (B, Q, 3, T).
    target_mask_areas = target_mask_areas.detach().cpu().numpy()
    intersection_areas = intersection_areas.detach().cpu().numpy()
    union_areas = union_areas.detach().cpu().numpy()

    snitch_ious = []
    occl_mask_ious = []
    cont_mask_ious = []
    snitch_during_vis_ious = []
    snitch_during_occl_ious = []
    snitch_during_cont_ious = []

    for b in range(B):
        for q in range(Q):
            for t in range(T):

                if target_mask_areas[b, q, 0, t] > 0:
                    # How well does the predicted snitch mask cover the GT snitch?
                    snitch_ious.append(
                        intersection_areas[b, q, 0, t] / (union_areas[b, q, 0, t] + 1e-7))

                if Cmt >= 2 and target_mask_areas[b, q, 1, t] > 0:
                    # How well does the predicted frontmost mask cover the GT frontmost?
                    occl_mask_ious.append(
                        intersection_areas[b, q, 1, t] / (union_areas[b, q, 1, t] + 1e-7))

                if Cmt >= 3 and target_mask_areas[b, q, 2, t] > 0:
                    # How well does the predicted outermost mask cover the GT outermost?
                    cont_mask_ious.append(
                        intersection_areas[b, q, 2, t] / (union_areas[b, q, 2, t] + 1e-7))

                # NOTE: We could look at target_flags here, but this is always 1 if target_mask is
                # non-zero in the occlusion or containment channel anyway.
                if target_mask_areas[b, q, 0, t] > 0 and \
                        Cmt >= 2 and target_mask_areas[b, q, 1, t] == 0:
                    snitch_during_vis_ious.append(snitch_ious[-1])
                
                if target_mask_areas[b, q, 0, t] > 0 and \
                        Cmt >= 2 and target_mask_areas[b, q, 1, t] > 0:
                    snitch_during_occl_ious.append(snitch_ious[-1])

                if target_mask_areas[b, q, 0, t] > 0 and \
                        Cmt >= 3 and target_mask_areas[b, q, 2, t] > 0:
                    snitch_during_cont_ious.append(snitch_ious[-1])

    # Return results.
    metrics_retval = dict()
    metrics_retval['mean_snitch_iou'] = np.mean(snitch_ious) \
        if len(snitch_ious) > 0 else -1.0
    metrics_retval['mean_occl_mask_iou'] = np.mean(occl_mask_ious) \
        if len(occl_mask_ious) > 0 else -1.0
    metrics_retval['mean_cont_mask_iou'] = np.mean(cont_mask_ious) \
        if len(cont_mask_ious) > 0 else -1.0
    metrics_retval['mean_snitch_during_vis_iou'] = np.mean(snitch_during_vis_ious) \
        if len(snitch_during_vis_ious) > 0 else -1.0
    metrics_retval['mean_snitch_during_occl_iou'] = np.mean(snitch_during_occl_ious) \
        if len(snitch_during_occl_ious) > 0 else -1.0
    metrics_retval['mean_snitch_during_cont_iou'] = np.mean(snitch_during_cont_ious) \
        if len(snitch_during_cont_ious) > 0 else -1.0
    metrics_retval['count_snitch_iou'] = len(snitch_ious)
    metrics_retval['count_occl_mask_iou'] = len(occl_mask_ious)
    metrics_retval['count_cont_mask_iou'] = len(cont_mask_ious)
    metrics_retval['count_snitch_during_vis_iou'] = len(snitch_during_vis_ious)
    metrics_retval['count_snitch_during_occl_iou'] = len(snitch_during_occl_ious)
    metrics_retval['count_snitch_during_cont_iou'] = len(snitch_during_cont_ious)

    for (k, v) in metrics_retval.items():
        v = torch.tensor(v, device=model_retval['output_mask'].device)
        if 'count' in k:
            v = v.type(torch.int32).detach()
        else:
            v = v.type(torch.float32).detach()
        metrics_retval[k] = v

    return metrics_retval


def calculate_weighted_averages(metrics_retvals):
    '''
    :param metrics_retvals (list) of metric_retval dicts, one per batch.
    :return (dict) of weighted averages.
    '''
    # DRY: This is also in loss.py.
    final_metrics = dict()
    for k in metrics_retvals[0].keys():
        if 'count' in k:
            count_key = k
            mean_key = k.replace('count', 'mean')
            old_counts = np.array([x[count_key] for x in metrics_retvals])
            old_means = np.array([x[mean_key] for x in metrics_retvals])

            # NOTE: Some mean values will be -1.0 but then corresponding counts are always 0.
            new_count = old_counts.sum()
            if new_count > 0:
                new_mean = np.multiply(old_means, old_counts).sum() / (new_count + 1e-7)
            else:
                new_mean = -1.0
            final_metrics[count_key] = new_count
            final_metrics[mean_key] = new_mean

    return final_metrics


def calculate_unweighted_averages(metrics_retvals, exclude_value=-1.0):
    '''
    :param metrics_retvals (list) of metric_retval dicts, one per batch.
    :return (dict) of unweighted averages.
    '''
    final_metrics = dict()
    for k in metrics_retvals[0].keys():
        if 'count' in k:
            count_key = k
            mean_key = k.replace('count', 'mean')
            all_values = np.array([x[mean_key] for x in metrics_retvals])

            filtered_values = all_values[all_values != exclude_value]
            if len(filtered_values) > 0:
                mean_value = filtered_values.mean()
            else:
                mean_value = np.nan

            final_metrics[count_key] = len(filtered_values)
            final_metrics[mean_key] = mean_value

    return final_metrics


def test_results_to_dataframe(inference_retvals):
    test_results = defaultdict(list)

    for inference_retval in inference_retvals:
        cur_data_retval = inference_retval['data_retval_pruned']
        cur_loss_retval = inference_retval['loss_retval']
        cur_metrics_retval = inference_retval['loss_retval']['metrics']

        test_results['source'].append(cur_data_retval['source_name'][0])
        test_results['dset_idx'].append(cur_data_retval['dset_idx'].item())
        test_results['scene_idx'].append(cur_data_retval['scene_idx'].item())
        if 'scene_dn' in cur_data_retval:
            test_results['scene_dn'].append(cur_data_retval['scene_dn'][0])
        test_results['friendly_short_name'].append(inference_retval['friendly_short_name'])

        for (k, v) in cur_loss_retval.items():
            if not('metrics' in k):
                if torch.is_tensor(v):
                    v = v.item()
                test_results['loss_' + k].append(v)

        for (k, v) in cur_metrics_retval.items():
            test_results[k].append(v)

    test_results = pd.DataFrame(test_results)
    return test_results


def calculate_weighted_averages_dataframe(csv):
    '''
    :param csv (pd.df).
    :return (dict) of weighted averages.
    '''
    final_metrics = dict()
    for k in csv.columns:
        if 'count' in k:
            count_key = k
            mean_key = k.replace('count', 'mean')
            old_counts = np.array(csv[count_key])
            old_means = np.array(csv[mean_key])

            # NOTE: Some mean values will be -1.0 but then corresponding counts are always 0.
            new_count = old_counts.sum()
            if new_count > 0:
                new_mean = np.multiply(old_means, old_counts).sum() / (new_count + 1e-7)
            else:
                new_mean = -1.0
            final_metrics[count_key] = new_count
            final_metrics[mean_key] = new_mean

    return final_metrics


def calculate_unweighted_averages_dataframe(csv, exclude_value=-1.0):
    '''
    :param csv (pd.df).
    :return (dict) of unweighted averages.
    '''
    final_metrics = dict()
    for k in csv.columns:
        if 'count' in k:
            count_key = k
            mean_key = k.replace('count', 'mean')
            all_values = np.array(csv[mean_key])

            filtered_values = all_values[all_values != exclude_value]
            if len(filtered_values) > 0:
                mean_value = filtered_values.mean()
            else:
                mean_value = np.nan

            final_metrics[count_key] = len(filtered_values)
            final_metrics[mean_key] = mean_value

    return final_metrics


def pretty_print_aggregated(logger, weighted_metrics, unweighted_metrics, num_scenes):

    longest_key = max([len(x) for x in weighted_metrics.keys()])

    logger.info()
    for k in sorted(weighted_metrics.keys()):
        if 'count' in k:
            count_key = k
            mean_key = k.replace('count', 'mean')
            short_key = k.replace('count_', '')
            final_count = weighted_metrics[count_key]
            unweighted_mean_value = unweighted_metrics[mean_key]
            logger.info(f'{("unweighted_" + mean_key).ljust(longest_key + 11)}  '
                        f'{(f"(over {num_scenes} scenes)").ljust(18)}:  '
                        f'{unweighted_mean_value:.5f}')
            if final_count > 0:
                logger.report_single_scalar('unweighted_' + short_key, unweighted_mean_value)

    logger.info()
    for k in sorted(weighted_metrics.keys()):
        if 'count' in k:
            count_key = k
            mean_key = k.replace('count', 'mean')
            short_key = k.replace('count_', '')
            final_count = weighted_metrics[count_key]
            weighted_mean_value = weighted_metrics[mean_key]
            logger.info(f'{("weighted_" + mean_key).ljust(longest_key + 8)}  '
                        f'{(f"(over {final_count} frames)").ljust(19)}:  '
                        f'{weighted_mean_value:.5f}')
            if final_count > 0:
                logger.report_single_scalar('weighted_' + short_key, weighted_mean_value)
