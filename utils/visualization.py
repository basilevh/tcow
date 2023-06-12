'''
Tools / utilities / helper methods pertaining to qualitative deep dives into train / test results.
Created by Basile Van Hoorick for TCOW.
'''

import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), 'eval/'))
sys.path.insert(0, os.getcwd())

from __init__ import *


def draw_text(image, topleft, label, color, size_mult=1.0):
    '''
    :param image (H, W, 3) array of float in [0, 1].
    :param topleft (2) tuple of int: (y, x) coordinates of the top left corner of the text.
    :param label (str): Text to draw.
    :param color (3) tuple of float in [0, 1]: RGB values.
    :param size_mult (float): Multiplier for font size.
    :return image (H, W, 3) array of float in [0, 1].
    '''
    # Draw background and write text using OpenCV.
    label_width = int((16 + len(label) * 8) * size_mult)
    label_height = int(22 * size_mult)
    (y, x) = topleft
    image[y:y + label_height, x:x + label_width] = (0, 0, 0)
    image = cv2.putText(image, label, (x, y + label_height - 8), 2,
                        0.5 * size_mult, color, thickness=int(size_mult))
    return image


def draw_segm_borders(segm, fill_white=False):
    '''
    :param segm (T, H, W, K) array of uint8.
    :return rgb_vis (T, H, W, 3) array of float32.
    '''
    assert segm.ndim == 4

    border_mask = np.abs(segm[:, 1:-1, 1:-1, :] - segm[:, :-2, 1:-1, :]) + \
        np.abs(segm[:, 1:-1, 1:-1, :] - segm[:, 2:, 1:-1, :]) + \
        np.abs(segm[:, 1:-1, 1:-1, :] - segm[:, 1:-1, :-2, :]) + \
        np.abs(segm[:, 1:-1, 1:-1, :] - segm[:, 1:-1, 2:, :])
    border_mask = np.any(border_mask, axis=-1)
    # (T, Hf - 2, Wf - 2) bytes in {0, 1}.
    border_mask = np.pad(border_mask, ((0, 0), (1, 1), (1, 1)), mode='constant')
    # (T, Hf, Wf) bytes in {0, 1}.

    if fill_white:
        border_mask = np.repeat(border_mask[..., None], repeats=3, axis=-1)
        # (T, Hf, Wf, 3) bytes in {0, 1}.
        result = border_mask.astype(np.float32)

    else:
        result = border_mask

    return result


def draw_dashed_circle(image, center, radius, color, segment_length, segment_thickness):
    '''
    :param image (H, W, 3) array of float in [0, 1].
    :param center (2) tuple of int: (y, x) coordinates of the center of the ellipse.
    :param radius (2) tuple of int: (y, x) radii of the ellipse.
    :param color (3) tuple of float in [0, 1]: RGB values.
    :param segment_length (int): Length of the segments of the dashed ellipse.
    :param segment_thickness (int): Thickness of the segments of the dashed ellipse.
    '''
    result = image.copy()
    center_y = center[0]
    center_x = center[1]

    if isinstance(radius, int):
        radius = (radius, radius)
    
    radius_y = radius[0]
    radius_x = radius[1]
    circum_y = 2.0 * np.pi * radius_y
    circum_x = 2.0 * np.pi * radius_x
    circum_avg = np.sqrt(circum_y * circum_x)
    
    num_segments = int(np.round(circum_avg / segment_length / 2.0) * 2)  # Must be even.
    angle_step = 2.0 * np.pi / num_segments

    for i in range(0, num_segments, 2):
        theta0 = i * angle_step
        theta1 = (i + 1) * angle_step
        y0 = int(np.round(center_y + radius_y * np.sin(theta0)))
        y1 = int(np.round(center_y + radius_y * np.sin(theta1)))
        x0 = int(np.round(center_x + radius_x * np.cos(theta0)))
        x1 = int(np.round(center_x + radius_x * np.cos(theta1)))

        result = cv2.line(result, (x0, y0), (x1, y1), color,
                          thickness=segment_thickness, lineType=cv2.LINE_AA)

    return result


def create_model_input_video(seeker_rgb, seeker_query_mask, query_border, apply_pause=True):
    '''
    :param seeker_rgb (T, H, W, 3) array of float32 in [0, 1].
    :param seeker_query_mask (T, H, W) array of float32 in [0, 1].
    :param query_border (T, H, W, 3) array of float32 in [0, 1].
    :return video (T, H, W, 3) array of float32 in [0, 1].
    '''
    query_time = seeker_query_mask.any(axis=(1, 2)).argmax()
    vis = seeker_rgb + seeker_query_mask[..., None]    # (T, H, W, 3).

    vis[query_time] *= 0.6
    vis[query_border, 0] = 0.0
    vis[query_border, 1] = 1.0
    vis[query_border, 2] = 0.0

    # Pause for a bit at query time to make the instance + mask very clear.
    if apply_pause:
        vis = np.concatenate([vis[0:query_time]] +
                            [vis[query_time:query_time + 1]] * 3 +
                            [vis[query_time + 1:]], axis=0)

    video = np.clip(vis, 0.0, 1.0)
    return video


def create_model_output_snitch_video(
        seeker_rgb, output_mask, query_border, snitch_border, grayscale=False):
    '''
    :param seeker_rgb (T, H, W, 3) array of float32 in [0, 1].
    :param output_mask (1/3, T, H, W) array of float32 in [0, 1].
    :param query_border (T, H, W) array of float32 in [0, 1].
    :param snitch_border (T, H, W) array of float32 in [0, 1].
    :param grayscale (bool): Whether to convert the output to grayscale.
    :return video (T, H, W, 3) array of float32 in [0, 1].
    '''
    if grayscale:
        seeker_rgb = seeker_rgb.copy()
        seeker_gray = seeker_rgb[..., 0] * 0.2 + seeker_rgb[..., 1] * 0.6 + seeker_rgb[..., 2] * 0.2
        seeker_rgb[..., 0] = seeker_gray
        seeker_rgb[..., 1] = seeker_gray
        seeker_rgb[..., 2] = seeker_gray

    snitch_heatmap = plt.cm.magma(output_mask[0])[..., 0:3]
    vis = seeker_rgb * 0.6 + snitch_heatmap * 0.5  # (T, H, W, 3).

    vis[query_border] = 0.0
    vis[snitch_border] = 0.0
    vis[query_border, 0] = 1.0
    vis[query_border, 2] = 1.0
    vis[snitch_border, 1] = 1.0

    video = np.clip(vis, 0.0, 1.0)  # (T, H, W, 3).
    return video


def create_model_output_snitch_occl_cont_video(
        seeker_rgb, output_mask, query_border, snitch_border, frontmost_border,
        outermost_border, grayscale=False):
    '''
    :param seeker_rgb (T, H, W, 3) array of float32 in [0, 1].
    :param output_mask (1/3, T, H, W) array of float32 in [0, 1].
    :param query_border (T, H, W) array of float32 in [0, 1].
    :param snitch_border (T, H, W) array of float32 in [0, 1].
    :param frontmost_border (T, H, W) array of float32 in [0, 1].
    :param outermost_border (T, H, W) array of float32 in [0, 1].
    :param grayscale (bool): Whether to convert the output to grayscale.
    :return video (T, H, W, 3) array of float32 in [0, 1].
    '''
    if grayscale:
        seeker_rgb = seeker_rgb.copy()
        seeker_gray = seeker_rgb[..., 0] * 0.2 + seeker_rgb[..., 1] * 0.6 + seeker_rgb[..., 2] * 0.2
        seeker_rgb[..., 0] = seeker_gray
        seeker_rgb[..., 1] = seeker_gray
        seeker_rgb[..., 2] = seeker_gray

    vis = seeker_rgb * 0.6

    vis[..., 1] += output_mask[0] * 0.5  # Snitch = green.
    if output_mask.shape[0] >= 2:
        vis[..., 0] += output_mask[1] * 0.5  # Frontmost occluder = red.
    if output_mask.shape[0] >= 3:
        vis[..., 2] += output_mask[2] * 0.5  # Outermost container = blue.

    vis[query_border] = 0.0
    vis[snitch_border] = 0.0
    vis[frontmost_border] = 0.0
    vis[outermost_border] = 0.0
    vis[query_border] = 1.0  # Always all white.
    vis[snitch_border, 1] = 1.0
    vis[frontmost_border, 0] = 1.0
    vis[outermost_border, 2] = 1.0

    video = np.clip(vis, 0.0, 1.0)  # (T, H, W, 3).
    return video


def create_snitch_weights_video(seeker_rgb, snitch_weights):
    '''
    :param seeker_rgb (T, H, W, 3) array of float32 in [0, 1].
    :param snitch_weights (T, H, W) array of float32 in [0, inf).
    :return video (T, H, W, 3) array of float32 in [0, 1].
    '''
    slw_norm = snitch_weights.max() + 1e-6
    lw_heatmap = plt.cm.viridis(snitch_weights / slw_norm)[..., 0:3]
    vis = seeker_rgb * 0.6 + lw_heatmap * 0.5  # (T, H, W, 3).

    video = np.clip(vis, 0.0, 1.0)  # (T, H, W, 3).
    return video


def create_model_input_target_video(
        seeker_rgb, seeker_query_mask, target_mask, query_border, snitch_border, frontmost_border,
        outermost_border, grayscale=False):
    '''
    :param seeker_rgb (T, H, W, 3) array of float32 in [0, 1].
    :param seeker_query_mask (T, H, W) array of float32 in [0, 1].
    :param target_mask (3, T, H, W) array of float32 in [0, 1].
    :param query_border (T, H, W, 3) array of float32 in [0, 1].
    :param snitch_border (T, H, W) array of float32 in [0, 1].
    :param frontmost_border (T, H, W) array of float32 in [0, 1].
    :param outermost_border (T, H, W) array of float32 in [0, 1].
    :param grayscale (bool): Whether to convert the output to grayscale.
    :return video (T, H, W, 3) array of float32 in [0, 1].
    '''
    if grayscale:
        seeker_rgb = seeker_rgb.copy()
        seeker_gray = seeker_rgb[..., 0] * 0.2 + seeker_rgb[..., 1] * 0.6 + seeker_rgb[..., 2] * 0.2
        seeker_rgb[..., 0] = seeker_gray
        seeker_rgb[..., 1] = seeker_gray
        seeker_rgb[..., 2] = seeker_gray
    
    vis = seeker_rgb.copy()    # (T, H, W, 3).

    # Fill in query mask for clarity.
    vis += seeker_query_mask[..., None] * 0.3

    # Fill in all target mask channels for clarity.
    target_mask = np.clip(target_mask, 0.0, 1.0)
    vis[1:, ..., 1] += target_mask[0, 1:] * 0.2  # Snitch = green, but ignore first frame (query).
    if target_mask.shape[0] >= 2:
        vis[..., 0] += target_mask[1] * 0.2  # Frontmost occluder = red.
    if target_mask.shape[0] >= 3:
        vis[..., 2] += target_mask[2] * 0.2  # Outermost container = blue.

    vis[query_border] = 0.0
    vis[snitch_border] = 0.0
    vis[frontmost_border] = 0.0
    vis[outermost_border] = 0.0
    vis[query_border] = 1.0  # Always all white.
    vis[snitch_border, 1] = 1.0
    vis[frontmost_border, 0] = 1.0
    vis[outermost_border, 2] = 1.0

    video = np.clip(vis, 0.0, 1.0)
    return video
