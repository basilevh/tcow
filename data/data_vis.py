'''
Dataset and annotation visualization methods, usually for temporary debugging.
Created by Basile Van Hoorick for TCOW.
'''

from __init__ import *

# Internal imports.
import my_utils
import visualization


def depth_to_rgb_vis(depth, max_depth=None):
    '''
    :depth (*, 1) array of float32.
    :return rgb_vis (*, 3) array of uint8.
    '''
    min_depth = 0.0
    if max_depth is None:
        max_depth = max(np.max(depth), 1e-6)

    depth = depth.copy().squeeze(-1)
    depth = np.clip(depth, 0.0, max_depth)
    depth = (depth - min_depth) / (max_depth - min_depth)

    rgb_vis = plt.cm.viridis(2.0 / (depth + 1.0) - 1.0)[..., :3]
    rgb_vis = (rgb_vis * 255.0).astype(np.uint8)

    return rgb_vis


def segm_rgb_to_ids_kubric(segm_rgb):  # , num_inst=None):
    '''
    :param segm_rgb (*, 3) array of RGB values.
    :return segm_ids (*, 1) array of 1-based instance IDs (0 = background).
    '''
    # We assume that hues are distributed across the range [0, 1] for instances in the image, ranked
    # by their integer ID. Check kubric plotting.hls_palette() for more details.
    hsv = matplotlib.colors.rgb_to_hsv(segm_rgb)
    to_rank = hsv[..., 0]  # + hsv[..., 2] * 1e-5
    unique_hues = np.sort(np.unique(to_rank))
    hue_start = 0.01
    assert np.isclose(unique_hues[0], 0.0, rtol=1e-3, atol=1e-3), str(unique_hues)

    # commented this because object ID 0 may not be always visible in every frame:
    # assert np.isclose(unique_hues[1], hue_start, rtol=1e-3, atol=1e-3), str(unique_hues)

    # The smallest jump inbetween subsequent hues determines the highest instance ID that is VALO,
    # which is <= the total number of instances. Skip the very first hue, which is always 0 and
    # corresponds to background.
    hue_steps = np.array([unique_hues[i] - unique_hues[i - 1] for i in range(2, len(unique_hues))])

    # For this sanity check to work, we must never have more than ~95 instances per scene.
    assert np.all(hue_steps >= 1e-2), str(hue_steps)

    # IMPORTANT NOTE: The current VALO set may be a strict SUBSET of the original VALO set (recorded
    # in the metadata), because we already applied frame subsampling here! In practice, this
    # sometimes causes big (i.e. integer multiple) jumps in hue_steps.
    # NEW: Ignore outliers the smart way.
    adjacent_steps = hue_steps[hue_steps <= np.min(hue_steps) * 1.5]
    hue_step = np.mean(adjacent_steps)

    # The jump from background to first instance is a special case, so ensure even distribution.
    nice_rank = to_rank.copy()
    nice_rank[nice_rank >= hue_start] += hue_step - hue_start
    ids_approx = (nice_rank / hue_step)

    segm_ids = np.round(ids_approx)[..., None].astype(np.int32)  # (T, H, W, 1).
    return segm_ids


def segm_ids_to_rgb(segm_ids, num_inst=None):
    '''
    NOTE: This is NOT consistent with segm_rgb_to_ids_kubric(), because background (0) gets mapped
        to red!
    :segm_ids (*, 1) array of uint32.
    :return segm_rgb (*, 3) array of uint8.
    '''
    if num_inst is None:
        num_inst = np.max(segm_ids) + 1
    num_inst = max(num_inst, 1)

    segm_ids = segm_ids.copy().squeeze(-1)
    segm_ids = segm_ids / num_inst

    segm_rgb = plt.cm.hsv(segm_ids)[..., :3]
    segm_rgb = (segm_rgb * 255.0).astype(np.uint8)

    return segm_rgb
