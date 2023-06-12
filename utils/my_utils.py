'''
Miscellaneous tools / utilities / helper methods.
Created by Basile Van Hoorick for TCOW.
'''

from __init__ import *

# Library imports.
import io
import numpy as np
import os


def get_checkpoint_epoch(checkpoint_path):
    '''
    Gets the 0-based epoch index of a stored model checkpoint.
    '''
    epoch_path = checkpoint_path[:-4] + '_epoch.txt'

    if os.path.exists(epoch_path):
        epoch = int(np.loadtxt(epoch_path, dtype=np.int32))

    else:
        # NOTE: This backup method is inefficient but I'm not sure how to do it better.
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        epoch = checkpoint['epoch']

    return epoch


def any_value(my_dict):
    for (k, v) in my_dict.items():
        if v is not None:
            return v
    return None


def traject_to_track_map(trajectory, frame_height, frame_width, map_cell_dim):
    '''
    Converts a track to a map.
    :param trajectory (T, 2): Sequence of [x, y] pairs; values in range [0, 1].
    :param frame_height, frame_width (int): Image dimensions.
    :param map_cell_dim (int): Size of one map cell (typically coarser than pixels).
    :return heatmap (1, T, Hm, Wm): Track heatmap over time.
    '''
    assert frame_height % map_cell_dim == 0
    assert frame_width % map_cell_dim == 0

    T = trajectory.shape[0]
    (Hm, Wm) = (frame_height // map_cell_dim, frame_width // map_cell_dim)
    heatmap = np.zeros((1, T, Hm, Wm), dtype=np.float32)

    for t in range(T):
        cell_x = int(np.floor(trajectory[t][0] * Wm))
        cell_y = int(np.floor(trajectory[t][1] * Hm))
        if 0 <= cell_x and cell_x < Wm and 0 <= cell_y and cell_y < Hm:
            heatmap[0, t, cell_y, cell_x] = 1.0

    return heatmap


def dict_to_cpu(x, ignore_keys=[]):
    '''
    Recursively converts all tensors in a dictionary to CPU.
    '''
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    elif isinstance(x, dict):
        return {k: dict_to_cpu(v, ignore_keys=ignore_keys) for (k, v) in x.items()
        if not(k in ignore_keys)}
    elif isinstance(x, list):
        return [dict_to_cpu(v, ignore_keys=ignore_keys) for v in x]
    else:
        return x


def is_nan_or_inf(x):
    '''
    Returns True if x is NaN or Inf.
    '''
    if torch.is_tensor(x):
        return torch.isnan(x).any() or torch.isinf(x).any()
    else:
        return np.any(np.isnan(x)) or np.any(np.isinf(x))


def get_fourier_positional_encoding_size(num_coords, num_frequencies):
    '''
    Returns the embedding dimensionality of Fourier positionally encoded points.
    :param num_coords (int) = C: Number of coordinate values per point.
    :param num_frequencies (int) = F: Number of frequencies to use.
    '''
    return num_coords * (1 + num_frequencies * 2)  # Identity + (cos + sin) for each frequency.


def apply_fourier_positional_encoding(raw_coords, num_frequencies,
                                      base_frequency=0.1, max_frequency=10.0):
    '''
    Applies Fourier positional encoding (cos + sin) to a set of coordinates. Note that it is the
        caller's responsibility to manage the value ranges of all coordinate dimensions.
    :param raw_coords (*, C) tensor: Points with UVD or XYZ or other values.
    :param num_frequencies (int) = F: Number of frequencies to use.
    :param base_frequency (float) = f_0: Determines coarsest level of detail.
    :param max_frequency (int) = f_M: Determines finest level of detail.
    :return enc_coords (*,  C * (1 + F * 2)): Embedded points.
    '''
    assert num_frequencies > 0
    assert base_frequency > 0
    assert max_frequency > base_frequency

    enc_coords = []
    enc_coords.append(raw_coords.clone())

    for f in range(num_frequencies):
        cur_freq = f * (max_frequency - base_frequency) / (num_frequencies - 1) + base_frequency
        enc_coords.append((raw_coords * 2.0 * np.pi * cur_freq).cos())
        enc_coords.append((raw_coords * 2.0 * np.pi * cur_freq).sin())

    enc_coords = torch.cat(enc_coords, dim=-1)
    return enc_coords


def elitist_shuffle(items, inequality):
    '''
    https://github.com/rragundez/elitist-shuffle
    Shuffle array with bias over initial ranks
    A higher ranked content has a higher probability to end up higher
    ranked after the shuffle than an initially lower ranked one.
    Args:
        items (numpy.array): Items to be shuffled
        inequality (int/float): how biased you want the shuffle to be.
            A higher value will yield a lower probabilty of a higher initially
            ranked item to end up in a lower ranked position in the
            sequence.
    '''
    weights = np.power(
        np.linspace(1, 0, num=len(items), endpoint=False),
        inequality
    )
    weights = weights / np.linalg.norm(weights, ord=1)
    return np.random.choice(items, size=len(items), replace=False, p=weights)


def quick_pca(array, k=3, unique_features=False, normalize=None):
    '''
    array (*, n): Array to perform PCA on.
    k (int) < n: Number of components to keep.
    '''
    n = array.shape[-1]
    all_axes_except_last = tuple(range(len(array.shape) - 1))
    array_flat = array.reshape(-1, n)

    pca = sklearn.decomposition.PCA(n_components=k)
    
    if unique_features:
        # Obtain unique combinations of occluding instance sequences, to avoid bias toward larger
        # object masks.
        unique_combinations = np.unique(array_flat, axis=0)
        pca.fit(unique_combinations)
    
    else:
        pca.fit(array_flat)
    
    result_unnorm = pca.transform(array_flat).reshape(*array.shape[:-1], k)
    
    if normalize is not None:
        per_channel_min = result_unnorm.min(axis=all_axes_except_last, keepdims=True)
        per_channel_max = result_unnorm.max(axis=all_axes_except_last, keepdims=True)
        result = (result_unnorm - per_channel_min) / (per_channel_max - per_channel_min)
        result = result * (normalize[1] - normalize[0]) + normalize[0]

    else:
        result = result_unnorm
    
    result = result.astype(np.float32)
    return result


def ax_to_numpy(ax, dpi=160):
    fig = ax.figure
    with io.BytesIO() as buff:
        fig.savefig(buff, format='raw', dpi=dpi)
        buff.seek(0)
        data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
    w, h = fig.get_size_inches() * dpi
    image = data.reshape((int(h), int(w), -1))
    image = image[..., 0:3] / 255.0
    return image


def disk_cached_call(logger, cache_fp, newer_than, func, *args, **kwargs):
    '''
    Caches the result of a function call to disk, to avoid recomputing it.
    :param cache_fp (str): Path to cache file.
    :param newer_than (float): Cache must be more recent than this UNIX timestamp.
    :param func (callable): Function to call.
    :param args: Positional arguments to pass to function.
    :param kwargs: Keyword arguments to pass to function.
    :return: Result of function call.
    '''
    use_cache = (cache_fp is not None and os.path.exists(cache_fp))
    if use_cache and newer_than is not None and os.path.getmtime(cache_fp) < newer_than:
        logger.info(f'Deleting too old cached result at {cache_fp}...')
        use_cache = False
        os.remove(cache_fp)
    
    if use_cache:
        with open(cache_fp, 'rb') as f:
            result = pickle.load(f)
    
    else:
        result = func(*args, **kwargs)
        
        if cache_fp is not None:
            cache_dp = str(pathlib.Path(cache_fp).parent)
            os.makedirs(cache_dp, exist_ok=True)
            logger.info(f'Caching {func.__name__} result to {cache_fp}...')
            with open(cache_fp, 'wb') as f:
                pickle.dump(result, f)
        
    return result


def calculate_iou_numpy(pred, target):
    '''
    Calculates the intersection-over-union of two binary masks.
    :param pred (*, H, W): Prediction mask.
    :param target (*, H, W): Target mask.
    :return iou (float): Intersection-over-union.
    '''
    assert pred.shape == target.shape
    pred = (pred > 0.5).astype(np.bool)
    target = (target > 0.5).astype(np.bool)
    intersection = np.sum(pred & target)
    union = np.sum(pred | target)
    iou = intersection / union
    return iou


def calculate_iou_torch(pred, target):
    '''
    Calculates the intersection-over-union of two binary masks.
    :param pred (*, H, W): Prediction mask.
    :param target (*, H, W): Target mask.
    :return iou (float): Intersection-over-union.
    '''
    assert pred.shape == target.shape
    pred = (pred > 0.5).bool()
    target = (target > 0.5).bool()
    intersection = torch.sum(torch.logical_and(pred, target))
    union = torch.sum(torch.logical_or(pred, target))
    iou = (intersection / union).item()
    return iou


def read_txt_strip_comments(txt_fp):
    with open(txt_fp, 'r') as f:
        lines = f.readlines()
    lines = [x.split('#')[0] for x in lines]
    lines = [x.strip() for x in lines]
    lines = [x for x in lines if len(x) > 0]
    return lines


def sample_query_inds(B, Qs, inst_count, target_desirability, train_args, device, phase):
    sel_query_inds = torch.zeros(B, Qs, dtype=torch.int64, device=device)
    for b in range(B):

        Qt = inst_count[b].item()  # = K = Number of VALO instances for this example.

        to_rank = target_desirability[b, :Qt, 0]
        to_rank = to_rank.detach().cpu().numpy()
        query_ranking_exact = np.argsort(to_rank)[::-1]

        # Skip invalid (negative) queries based on desirability values. If mask_track, this
        # ensures (nearly) invisible objects (especially in the first frame) are never sampled.
        query_ranking_valid = query_ranking_exact[to_rank[query_ranking_exact] >= 0.0]
        num_valid = len(query_ranking_valid)

        # This should never happen because it is also handled in _load_example_verify().
        assert num_valid >= Qs, \
            f'Not enough valid queries available for batch index {b}.'

        # Apply elitist sorting to increase diversity and avoid overfitting to particular kinds
        # of regions in the video.
        # NOTE: For evaluation, we avoid randomness, so a higher num_queries is recommended!
        if not('test' in phase):
            query_ranking_rough = elitist_shuffle(query_ranking_valid, inequality=9)
        else:
            query_ranking_rough = query_ranking_valid
        query_ranking_rough = torch.tensor(query_ranking_rough, device=device)
        sel_query_inds[b, :] = query_ranking_rough[:Qs]

        # Finally, ensure one track is often uniformly randomly sampled (without replacement) to
        # further balance the distribution, e.g. prevent the model from thinking that everything
        # always moves, or that everything is an object.
        if not('test' in phase):
            random_prob = np.clip(0.2 + Qs * 0.1, 0.3, 0.5)
            # 0.3, 0.4, 0.5, 0.5 for 1, 2, 3, 4.
            if np.random.rand() < random_prob:
                sel_rank_idx = np.random.randint(Qs - 1, num_valid)
                sel_query_inds[b, -1] = query_ranking_rough[sel_rank_idx]

    sel_query_inds = sel_query_inds.cpu()
    return sel_query_inds



if __name__ == '__main__':

    torch.set_printoptions(precision=3, sci_mode=False)