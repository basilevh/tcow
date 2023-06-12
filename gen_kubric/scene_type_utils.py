'''
Kubric scene creation methods for benchmarking isolated physical concepts and/or emergent phenomena.
Created by Basile Van Hoorick for TCOW.
'''


import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'experimental/'))
sys.path.append(os.path.join(os.getcwd(), 'gen_bench/'))

from __init__ import *


def spawn_background_static(logger, my_kubric, along_x=True):

    my_kubric.insert_static_objects(min_count=4, max_count=4,
                                    any_diameter_range=(1.0, 2.0))

    if along_x:
        # All four objects at -X and +X.
        my_kubric.scene.foreground_assets[-4].position = my_kubric.random_state.uniform(
            (-6.0, -2.0, 1.5), (-4.0, -1.0, 1.5))
        my_kubric.scene.foreground_assets[-3].position = my_kubric.random_state.uniform(
            (-6.0, 1.0, 1.5), (-4.0, 2.0, 1.5))
        my_kubric.scene.foreground_assets[-2].position = my_kubric.random_state.uniform(
            (4.0, -2.0, 1.5), (6.0, -1.0, 1.5))
        my_kubric.scene.foreground_assets[-1].position = my_kubric.random_state.uniform(
            (4.0, 1.0, 1.5), (6.0, 2.0, 1.5))

    else:
        # All four objects at -X, -Y, +X, +Y (one per side).
        my_kubric.scene.foreground_assets[-4].position = my_kubric.random_state.uniform(
            (-6.0, -2.0, 1.5), (-4.0, -2.0, 1.5))
        my_kubric.scene.foreground_assets[-3].position = my_kubric.random_state.uniform(
            (-2.0, -6.0, 1.5), (2.0, -4.0, 1.5))
        my_kubric.scene.foreground_assets[-2].position = my_kubric.random_state.uniform(
            (4.0, -2.0, 1.5), (6.0, -2.0, 1.5))
        my_kubric.scene.foreground_assets[-1].position = my_kubric.random_state.uniform(
            (-2.0, 4.0, 1.5), (2.0, 6.0, 1.5))

    pass


def setup_gravity_bounce(logger, my_kubric):

    spawn_background_static(logger, my_kubric, along_x=True)

    my_kubric.simulate_frames(-100, -1)
    my_kubric.reset_objects_velocity_friction_restitution()

    my_kubric.insert_dynamic_objects(min_count=2, max_count=2,
                                     any_diameter_range=(1.0, 2.0))
    my_kubric.scene.foreground_assets[-2].position = my_kubric.random_state.uniform(
        (-1.0, -2.5, 4.0), (1.0, -1.5, 6.0))
    my_kubric.scene.foreground_assets[-2].velocity = my_kubric.random_state.uniform(
        (-0.5, -0.5, -3.0), (0.5, 0.5, -1.0))
    my_kubric.scene.foreground_assets[-1].position = my_kubric.random_state.uniform(
        (-1.0, 1.5, 4.0), (1.0, 2.5, 6.0))
    my_kubric.scene.foreground_assets[-1].velocity = my_kubric.random_state.uniform(
        (-0.5, -0.5, -3.0), (0.5, 0.5, -1.0))

    pass


def setup_fall_onto_carrier(logger, my_kubric):

    spawn_background_static(logger, my_kubric, along_x=False)

    my_kubric.insert_static_objects(min_count=1, max_count=1, force_carriers=1,
                                    container_carrier_diameter_range=(2.0, 3.0))
    my_kubric.scene.foreground_assets[-1].position = [0.0, 0.0, 1.5]

    my_kubric.simulate_frames(-100, -1)
    my_kubric.reset_objects_velocity_friction_restitution()

    my_kubric.insert_dynamic_objects(min_count=1, max_count=1,
                                     any_diameter_range=(0.5, 1.5))
    my_kubric.scene.foreground_assets[-1].position = my_kubric.random_state.uniform(
        (-0.5, -0.5, 4.0), (0.5, -0.5, 6.0))
    my_kubric.scene.foreground_assets[-1].velocity = my_kubric.random_state.uniform(
        (-0.5, -0.5, -1.0), (0.5, 0.5, 0.0))

    pass


def setup_fall_into_container(logger, my_kubric):

    spawn_background_static(logger, my_kubric, along_x=False)

    my_kubric.insert_static_objects(min_count=1, max_count=1, force_containers=1,
                                    container_carrier_diameter_range=(2.0, 3.0),
                                    simple_containers_only=True)
    my_kubric.scene.foreground_assets[-1].position = [0.0, 0.0, 1.5]

    my_kubric.simulate_frames(-100, -1)
    my_kubric.reset_objects_velocity_friction_restitution()

    my_kubric.insert_dynamic_objects(min_count=1, max_count=1,
                                     any_diameter_range=(0.5, 1.5))
    my_kubric.scene.foreground_assets[-1].position = my_kubric.random_state.uniform(
        (-0.5, -0.5, 4.0), (0.5, -0.5, 6.0))
    my_kubric.scene.foreground_assets[-1].velocity = my_kubric.random_state.uniform(
        (-0.5, -0.5, -1.0), (0.5, 0.5, 0.0))

    pass


def setup_slide_box_friction(logger, my_kubric):

    spawn_background_static(logger, my_kubric, along_x=True)

    my_kubric.simulate_frames(-100, -1)
    my_kubric.reset_objects_velocity_friction_restitution()

    my_kubric.insert_dynamic_objects(min_count=2, max_count=2, force_boxes=2,
                                     box_diameter_range=(1.5, 2.0))
    my_kubric.scene.foreground_assets[-2].position = my_kubric.random_state.uniform(
        (1.5, -2.0, 1.0), (2.0, -1.5, 1.5))
    my_kubric.scene.foreground_assets[-2].velocity = my_kubric.random_state.uniform(
        (0.0, 4.5, 0.0), (0.0, 5.0, 0.0))
    my_kubric.scene.foreground_assets[-1].position = my_kubric.random_state.uniform(
        (-2.0, 1.5, 1.0), (-1.5, 2.0, 1.5))
    my_kubric.scene.foreground_assets[-1].velocity = my_kubric.random_state.uniform(
        (0.0, -5.0, 0.0), (0.0, -4.5, 0.0))

    pass


def setup_slide_box_collide(logger, my_kubric):

    spawn_background_static(logger, my_kubric, along_x=True)

    my_kubric.simulate_frames(-100, -1)
    my_kubric.reset_objects_velocity_friction_restitution()

    my_kubric.insert_dynamic_objects(min_count=2, max_count=2, force_boxes=2,
                                     box_diameter_range=(1.5, 2.0))
    my_kubric.scene.foreground_assets[-2].position = my_kubric.random_state.uniform(
        (0.0, -4.0, 1.0), (0.0, -3.5, 1.5))
    my_kubric.scene.foreground_assets[-2].velocity = my_kubric.random_state.uniform(
        (0.0, 4.5, 0.0), (0.0, 5.0, 0.0))
    my_kubric.scene.foreground_assets[-1].position = my_kubric.random_state.uniform(
        (0.0, 3.5, 1.0), (0.0, 4.0, 1.5))
    my_kubric.scene.foreground_assets[-1].velocity = my_kubric.random_state.uniform(
        (0.0, -5.0, 0.0), (0.0, -4.5, 0.0))

    pass


def setup_box_push_carrier_slide(logger, my_kubric):

    spawn_background_static(logger, my_kubric, along_x=True)

    my_kubric.insert_static_objects(min_count=1, max_count=1, force_carriers=1,
                                    container_carrier_diameter_range=(2.0, 3.0))
    my_kubric.scene.foreground_assets[-1].position = [0.0, 0.0, 1.5]

    my_kubric.simulate_frames(-100, -1)
    my_kubric.reset_objects_velocity_friction_restitution()
    
    my_kubric.insert_dynamic_objects(min_count=1, max_count=1,
                                     any_diameter_range=(0.5, 1.5))
    my_kubric.scene.foreground_assets[-1].position = my_kubric.random_state.uniform(
        (-0.5, -0.5, 2.0), (0.5, -0.5, 4.0))
    my_kubric.scene.foreground_assets[-1].velocity = my_kubric.random_state.uniform(
        (-0.5, -0.5, -1.0), (0.5, 0.5, 0.0))

    my_kubric.insert_dynamic_objects(min_count=1, max_count=1, force_boxes=1,
                                     box_diameter_range=(1.5, 2.0))
    my_kubric.scene.foreground_assets[-1].position = my_kubric.random_state.uniform(
        (0.0, 4.5, 1.0), (0.0, 5.0, 1.5))
    my_kubric.scene.foreground_assets[-1].velocity = my_kubric.random_state.uniform(
        (0.0, -6.0, 0.0), (0.0, -5.5, 0.0))
    my_kubric.scene.foreground_assets[-1].mass *= 2.0

    pass


def setup_box_push_container_slide(logger, my_kubric):

    spawn_background_static(logger, my_kubric, along_x=True)

    my_kubric.insert_static_objects(min_count=1, max_count=1, force_containers=1,
                                    container_carrier_diameter_range=(2.0, 2.5),
                                    simple_containers_only=True)
    my_kubric.scene.foreground_assets[-1].position = [0.0, 0.0, 2.0]
    
    my_kubric.simulate_frames(-100, -1)
    my_kubric.reset_objects_velocity_friction_restitution()
    
    my_kubric.insert_dynamic_objects(min_count=1, max_count=1,
                                     any_diameter_range=(0.5, 1.5))
    my_kubric.scene.foreground_assets[-1].position = my_kubric.random_state.uniform(
        (-0.5, -0.5, 2.5), (0.5, -0.5, 4.5))
    my_kubric.scene.foreground_assets[-1].velocity = my_kubric.random_state.uniform(
        (-0.5, -0.5, -1.0), (0.5, 0.5, 0.0))

    my_kubric.insert_dynamic_objects(min_count=1, max_count=1, force_boxes=1,
                                     box_diameter_range=(1.5, 2.0))
    my_kubric.scene.foreground_assets[-1].position = my_kubric.random_state.uniform(
        (0.0, 5.0, 1.0), (0.0, 5.5, 1.5))
    my_kubric.scene.foreground_assets[-1].velocity = my_kubric.random_state.uniform(
        (0.0, -6.0, 0.0), (0.0, -5.5, 0.0))
    my_kubric.scene.foreground_assets[-1].mass *= 3.0

    pass


def apply_setup_for_type(logger, my_kubric, scene_type):
    if scene_type == 'gravity_bounce':
        setup_gravity_bounce(logger, my_kubric)

    elif scene_type == 'fall_onto_carrier':
        setup_fall_onto_carrier(logger, my_kubric)

    elif scene_type == 'fall_into_container':
        setup_fall_into_container(logger, my_kubric)

    elif scene_type == 'slide_box_friction':
        setup_slide_box_friction(logger, my_kubric)

    elif scene_type == 'slide_box_collide':
        setup_slide_box_collide(logger, my_kubric)

    elif scene_type == 'box_push_carrier_slide':
        setup_box_push_carrier_slide(logger, my_kubric)

    elif scene_type == 'box_push_container_slide':
        setup_box_push_container_slide(logger, my_kubric)

    else:
        raise ValueError(f'Unknown scene type: {scene_type}')
