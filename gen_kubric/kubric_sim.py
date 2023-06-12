'''
Single Kubric simulator instance.
Adapted from movi_def_worker.py.
Created by Basile Van Hoorick for TCOW.
'''

import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'data/'))
sys.path.append(os.path.join(os.getcwd(), 'eval/'))

from __init__ import *

# Library imports.
import pandas as pd
import time
import tempfile

# Internal imports.
from kubric_constants import *


CONTAINER_CARRIER_SPAWN_REGION = [(-4, -4, 0), (4, 4, 4)]
STATIC_SPAWN_REGION = [(-7, -7, 0), (7, 7, 7)]
DYNAMIC_SPAWN_REGION = [(-5, -5, 1), (5, 5, 6)]
VELOCITY_RANGE = [(-4, -4, -1), (4, 4, 1)]


class MyKubricSimulatorRenderer:
    '''
    This class is capable of generating Kubric scenes in a customizable way.
    It can be used both for offline and online simulation and rendering.
    '''

    def __init__(self, logger, frame_width=256, frame_height=192, num_frames=24, frame_rate=12,
                 motion_blur=True, render_samples_per_pixel=32, split_backgrounds=False,
                 split_objects=False, render_use_gpu=False, render_cpu_threads=-1,
                 scratch_dir=None, mass_est_fp=None, dome_friction_range=[0.3, 0.4],
                 dome_restit_range=[0.6, 0.7], object_friction_range=[0.4, 0.5],
                 object_restit_range=[0.6, 0.7]):
        '''
        Initializes the context for Kubric and stores some data distribution parameters.
        '''
        # WARNING / NOTE: We CANNOT import bpy outside of the actual thread / process using it!
        # Otherwise, a lot of problems / crashes will occur that do not seem to be easily fixable.
        # Therefore, kubric stuff also has to be placed here because it itself imports bpy.
        import bpy
        import kubric as kb
        import kubric.simulator
        import kubric.renderer

        # Keep the bpy and Kubric imports alive throughout the lifetime of this object instance.
        # Feels weird but this seems to be the only way to use their functionality in other methods.
        self.bpy_module = bpy
        self.kb_module = kb

        self.logger = logger
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.num_frames = num_frames
        self.frame_rate = frame_rate
        self.motion_blur = motion_blur
        self.render_samples_per_pixel = render_samples_per_pixel
        self.split_backgrounds = split_backgrounds
        self.split_objects = split_objects
        self.render_use_gpu = render_use_gpu
        self.render_cpu_threads = render_cpu_threads
        self.mass_est_fp = mass_est_fp
        self.dome_friction_range = dome_friction_range
        self.dome_restit_range = dome_restit_range
        self.object_friction_range = object_friction_range
        self.object_restit_range = object_restit_range

        # Initialize scratch directory.
        if scratch_dir is None:
            scratch_dir = tempfile.mkdtemp()
        else:
            # scratch_dir = os.path.join(scratch_dir, str(np.random.randint(1000000000, 9999999999)))
            os.makedirs(scratch_dir, exist_ok=True)
            self.logger.info(f'Using scratch directory: {scratch_dir}.')

        # Initialize variables.
        self.scene = None
        self.scratch_dir = scratch_dir
        self.simulator = None
        self.renderer = None

        # Prepare asset sources; same as MOVi-D/E/F.
        self.gso_source = kb.AssetSource.from_manifest(
            'gs://kubric-public/assets/GSO/GSO.json')
        self.hdri_source = kb.AssetSource.from_manifest(
            'gs://kubric-public/assets/HDRI_haven/HDRI_haven.json')
        self.kubasic_source = kb.AssetSource.from_manifest(
            'gs://kubric-public/assets/KuBasic/KuBasic.json')
        self.gso_shoe_ids = self._asset_ids_from_contains(GSO_SHOE_CONTAINS)
        self.gso_box_ids = self._asset_ids_from_contains(GSO_BOX_CONTAINS)

        # Set rendering options.
        if self.render_cpu_threads >= 1:
            self.bpy_module.context.scene.render.threads_mode = 'FIXED'
            self.bpy_module.context.scene.render.threads = render_cpu_threads
        os.environ['KUBRIC_USE_GPU'] = '1' if render_use_gpu else '0'

        # Set physics parameters.
        # Default values for kubcon_v8 / kubbench_v2:
        # self.dome_friction = 0.3
        # self.dome_restitution = 0.7
        # self.object_friction = 0.5
        # self.object_restitution = 0.7
        # From kubcon_v9 / kubbench_v3 onwards: use 4 provided ranges.

        # Load mass knowledge.
        # It is recommended to use mass_min_max_dict, which maps asset IDs to practical mass ranges.
        if self.mass_est_fp is not None:
            mass_est_list = pd.read_csv(self.mass_est_fp, header=None, names=['id', 'samples'])
            self.mass_samples_dict = {id: np.fromstring(samples[1:-1], dtype=np.float32, sep=' ')
                                      for (id, samples) in mass_est_list.values}
            self.mass_min_max_dict = {id: (samples.mean() * 0.5, samples.mean() * 1.5)
                                      for (id, samples) in self.mass_samples_dict.items()}
        else:
            self.mass_samples_dict = dict()
            self.mass_min_max_dict = dict()

    def _asset_ids_from_contains(self, contains_list):
        '''
        Extracts GSO asset IDs from a list of patterns to match that may be part of the asset name.
        '''
        ids_list = self.gso_source.all_asset_ids()
        ids_list = [asset_id for asset_id in ids_list
                    if any([pattern in asset_id.lower()
                            for pattern in contains_list])]
        return ids_list

    def _setup_camera(self, start_yaw_deg=0.0, first_time=False):
        '''
        X
        '''
        self.scene.camera = self.kb_module.PerspectiveCamera(focal_length=34.0, sensor_width=32.0)

        if first_time:
            movement_speed = self.random_state.uniform(0.0, 8.0)

            # NOTE: Compared to random half sphere sampling (as is default in MOVi), this alternative is
            # slightly biased toward lower viewing angles.
            cam_start_radius = self.random_state.uniform(11.0, 13.0)
            cam_start_z = self.random_state.uniform(0.1, cam_start_radius - 1.0)
            xy_radius = np.sqrt((cam_start_radius ** 2 - cam_start_z ** 2))
            cam_start_x = xy_radius * np.cos(start_yaw_deg * np.pi / 180.0)
            cam_start_y = xy_radius * np.sin(start_yaw_deg * np.pi / 180.0)
            fix_start = (cam_start_x, cam_start_y, cam_start_z)

            (camera_start, camera_end) = self.construct_linear_camera_motion(
                movement_speed, inner_radius=10.0, outer_radius=13.0, z_offset=0.1, fix_start=fix_start)

            x_look = self.random_state.uniform(-1.0, 1.0)
            y_look = self.random_state.uniform(-1.0, 1.0)
            z_look = self.random_state.uniform(0.0, 2.0)
            xyz_look = np.array([x_look, y_look, z_look])

        else:
            # We wish to reuse and repeat all movement parameters, just from a different angle.
            rel_yaw_deg = start_yaw_deg - self.last_start_yaw_deg
            camera_start = _rotate_yaw(self.last_camera_start, rel_yaw_deg)
            camera_end = _rotate_yaw(self.last_camera_end, rel_yaw_deg)
            xyz_look = _rotate_yaw(self.last_xyz_look, rel_yaw_deg)

        # Linearly interpolate the camera position between these two points while keeping it focused
        # on the center of the scene. We start one frame early and end one frame late to ensure that
        # forward and backward optical flow remain consistent for the last and first frames.
        for frame in range(-1, self.num_frames + 2):
            interp = ((frame + 1) / (self.num_frames + 3))
            self.scene.camera.position = ((1.0 - interp) * np.array(camera_start) +
                                          interp * np.array(camera_end))
            self.scene.camera.look_at(xyz_look)
            self.scene.camera.keyframe_insert('position', frame)
            self.scene.camera.keyframe_insert('quaternion', frame)

        # Save parameters for possible rotations in the future.
        self.last_start_yaw_deg = start_yaw_deg
        self.last_camera_start = camera_start
        self.last_camera_end = camera_end
        self.last_xyz_look = xyz_look

    def _get_random_asset_of_kind(self, object_kind, allow_complex=True):
        '''
        X
        '''
        upside_down = False

        if object_kind == 'any':
            # Manage dataset splits as a function of phase, if applicable.
            if self.split_objects:
                (train_ids, test_ids) = self.gso_source.get_test_split(fraction=0.1)
                if self.phase == 'train':
                    active_subset = train_ids
                elif self.phase == 'test':
                    active_subset = test_ids
            else:
                active_subset = self.gso_source.all_asset_ids()

        elif object_kind == 'container':
            # In the general container case, set 5% to be hats, 10% to be shoes, and the rest any.
            dice = self.random_state.rand()
            if allow_complex and dice < 0.05:
                active_subset = GSO_HAT_IDS
                upside_down = True
            elif allow_complex and dice < 0.15:
                active_subset = self.gso_shoe_ids
            else:
                active_subset = GSO_CONTAINER_IDS

        elif object_kind == 'carrier':
            active_subset = GSO_CARRIER_IDS

        elif object_kind == 'box':
            active_subset = self.gso_box_ids

        else:
            raise ValueError(object_kind)

        asset_id = self.random_state.choice(active_subset)
        return (asset_id, upside_down)

    def prepare_next_scene(self, phase, random_seed, camera_yaw_deg=None):
        '''
        X
        '''
        # NOTE: Both val_aug and val_noaug will use the train background / object splits!
        # Test splits are therefore reserved for the actual test set only.
        if 'val' in phase:
            phase = 'train'
        assert phase in ['train', 'test']

        start_time = time.time()
        self.phase = phase
        self.random_seed = random_seed
        self.random_state = np.random.RandomState(seed=random_seed)

        # Instantiate actual Kubric scene object.
        self.scene = self.kb_module.Scene(frame_start=0, frame_end=self.num_frames - 1,
                                          frame_rate=self.frame_rate,
                                          resolution=(self.frame_width, self.frame_height))

        # =============================
        # Prepare PyBullet and Blender.

        # Reuse simulator instance throughout the entire process lifetime to avoid PyBullet errors.
        if self.simulator is None:
            self.simulator = self.kb_module.simulator.PyBullet(self.scene,
                                                               scratch_dir=self.scratch_dir)
        else:
            self.simulator.scene = self.scene
            self.simulator.scratch_dir = self.scratch_dir

        # Refresh renderer every time to ensure we start fresh.
        motion_blur_value = self.random_state.uniform(0.4, 0.8) if self.motion_blur else 0.0
        self.renderer = self.kb_module.renderer.Blender(
            self.scene, scratch_dir=self.scratch_dir, adaptive_sampling=False, use_denoising=True,
            samples_per_pixel=self.render_samples_per_pixel, motion_blur=motion_blur_value)

        # Manage dataset splits as a function of phase, if applicable.
        if self.split_backgrounds:
            (train_ids, test_ids) = self.hdri_source.get_test_split(fraction=0.1)
            if self.phase == 'train':
                active_subset = train_ids
            elif self.phase == 'test':
                active_subset = test_ids
        else:
            active_subset = self.hdri_source.all_asset_ids()

        # ====================================
        # Populate scene with HDRI background.
        hdri_id = self.random_state.choice(active_subset)
        background_hdri = self.hdri_source.create(asset_id=hdri_id, name='bg_hdri')
        self.scene.metadata['background'] = hdri_id

        # Paint dome object (half sphere) with the selected image.
        self.dome = self.kubasic_source.create(
            asset_id='dome', name='dome', friction=1.0, restitution=0.0, static=True,
            background=True)
        assert isinstance(self.dome, self.kb_module.FileBasedObject)
        self.scene += self.dome

        # Apply random yaw because camera position may be fixed.
        # NOTE: Disabled for kubcon_v9 because it causes misalignment with HDRI ambient light.
        # self.kb_module.rotation_sampler(axis='Z')(self.dome, self.random_state)

        dome_blender = self.dome.linked_objects[self.renderer]
        texture_node = dome_blender.data.materials[0].node_tree.nodes['Image Texture']
        texture_node.image = self.bpy_module.data.images.load(background_hdri.filename)

        # Make the illumination and shadows consistent with the background.
        self.renderer._set_ambient_light_hdri(background_hdri.filename)

        # ========================
        # Setup camera trajectory.
        if camera_yaw_deg is None:
            camera_yaw_deg = self.random_state.uniform(0.0, 360.0)
        self._setup_camera(start_yaw_deg=camera_yaw_deg, first_time=True)

        return time.time() - start_time

    def set_camera_yaw(self, new_yaw_deg):
        '''
        X
        '''
        start_time = time.time()

        self._setup_camera(start_yaw_deg=new_yaw_deg, first_time=False)

        return time.time() - start_time

    def _fix_mass_knowledge(self, asset_id, obj):
        '''
        X
        '''
        auto_mass = obj.mass
        obj.metadata['auto_mass'] = obj.mass
        obj.metadata['auto_density'] = obj.mass / obj.metadata['volume']

        if asset_id in self.mass_min_max_dict:
            # Assume there is some inherent inaccuracy / uncertainty in the mass estimation, by
            # applying uniformly random noise within +/- 50% bounds.
            gpt_mass = max(self.random_state.uniform(*self.mass_min_max_dict[asset_id]), 1e-4)

            # Apply geometric mean between the default and knowledge-derived values.
            obj.mass = np.sqrt(gpt_mass * auto_mass)

            obj.metadata['override_mass'] = obj.mass
            obj.metadata['override_density'] = obj.mass / obj.metadata['volume']
            self.logger.info(f'Set mass to {obj.mass * 1000.0:.1f} g for {asset_id}, '
                             f'which amounts to a factor x{obj.mass / auto_mass:.1f} change in density.')

        elif len(self.mass_min_max_dict) != 0:
            # Always call this to update random_state to maintain consistency / reproducibility.
            stub_call = self.random_state.uniform(0.5, 1.5)
            self.logger.warning(f'No mass knowledge for {asset_id}.')

        return obj

    def insert_static_objects(self, min_count=8, max_count=12, force_containers=0,
                              force_carriers=0, any_diameter_range=(0.75, 2.5),
                              container_carrier_diameter_range=(1.25, 3.0),
                              simple_containers_only=False):
        '''
        X
        '''
        start_time = time.time()

        total_static_objects = self.random_state.randint(min_count, max_count + 1)
        num_any_objects = max(total_static_objects - force_containers - force_carriers, 0)
        object_kinds = ['container'] * force_containers + \
            ['carrier'] * force_carriers + \
            ['any'] * num_any_objects
        assert len(object_kinds) >= total_static_objects
        self.logger.info('Randomly placing %d containers, %d carriers, and %d static/any objects:',
                         force_containers, force_carriers, num_any_objects)

        for i, object_kind in enumerate(object_kinds):
            # NOTE: While carrier is always carrier, container may be generic, hat, or shoe.
            # NOTE: Dataset split handling only happens if object_kind is any.
            (asset_id, upside_down) = self._get_random_asset_of_kind(
                object_kind, allow_complex=not(simple_containers_only))
            obj = self.gso_source.create(asset_id=asset_id, name=f'static_{i:03d}')
            assert isinstance(obj, self.kb_module.FileBasedObject)

            # Overwrite mass with provided knowledge, if available.
            obj = self._fix_mass_knowledge(asset_id, obj)

            # Make all objects of roughly similar size, though containers and carriers should be
            # slightly bigger.
            axis_diameter = self.random_state.uniform(*any_diameter_range) \
                if object_kind == 'any' else \
                self.random_state.uniform(*container_carrier_diameter_range)
            scale_factor = axis_diameter / np.max(obj.bounds[1] - obj.bounds[0])
            obj.scale = scale_factor
            obj.metadata['axis_diameter'] = axis_diameter
            obj.metadata['scale_factor'] = scale_factor
            obj.metadata['mass_pre'] = obj.mass
            obj.mass *= np.power(scale_factor, 3.0)  # Volumetric scaling law is important here.
            obj.metadata['mass_post'] = obj.mass

            # Turn hats upside down. The initial quaternion is (w, x, y, z) = (1, 0, 0, 0).
            if upside_down and object_kind in ['container', 'carrier']:
                # This rotates 180 degrees around X, so flips Y and Z.
                obj.quaternion = np.array([0, 1, 0, 0])
            obj.metadata['initial_quaternion'] = obj.quaternion

            # Insert object into scene, ensuring that all contents remain disjoint.
            # NOTE: For containers and carriers to work, they must stay mostly upright when placed.
            self.scene += obj
            rotation_axis = 'Z' if object_kind in ['container', 'carrier'] else None
            spawn_region = STATIC_SPAWN_REGION if object_kind == 'any' else \
                CONTAINER_CARRIER_SPAWN_REGION if object_kind in ['container', 'carrier'] else None
            self.move_until_no_overlap(
                obj, self.simulator, rotation_axis=rotation_axis, spawn_region=spawn_region,
                rng=self.random_state)

            obj.friction = 1.0
            obj.restitution = 0.0
            obj.metadata['insert_order'] = i
            obj.metadata['object_kind'] = object_kind
            obj.metadata['is_dynamic'] = False

        return time.time() - start_time

    def insert_dynamic_objects(self, min_count=4, max_count=6, force_boxes=0,
                               any_diameter_range=(0.5, 2.0), box_diameter_range=(0.75, 2.0)):
        '''
        X
        '''
        start_time = time.time()

        total_dynamic_objects = self.random_state.randint(min_count, max_count + 1)
        num_any_objects = max(total_dynamic_objects - force_boxes, 0)
        object_kinds = ['box'] * force_boxes + \
            ['any'] * num_any_objects
        assert len(object_kinds) >= total_dynamic_objects
        self.logger.info('Randomly placing %d boxes and %d dynamic/any objects:',
                         force_boxes, num_any_objects)

        for i, object_kind in enumerate(object_kinds):
            # NOTE: While carrier is always carrier, container may be generic, hat, or shoe.
            # NOTE: Dataset split handling only happens if object_kind is any.
            (asset_id, upside_down) = self._get_random_asset_of_kind(object_kind)
            obj = self.gso_source.create(asset_id=asset_id, name=f'dynamic_{i:03d}')
            assert isinstance(obj, self.kb_module.FileBasedObject)

            # Overwrite mass with provided knowledge, if available.
            obj = self._fix_mass_knowledge(asset_id, obj)

            # Make all objects of roughly similar size, though dynamic objects should be slightly
            # smaller than static ones, especially containers.
            axis_diameter = self.random_state.uniform(*any_diameter_range) \
                if object_kind == 'any' else \
                self.random_state.uniform(*box_diameter_range)
            scale_factor = axis_diameter / np.max(obj.bounds[1] - obj.bounds[0])
            obj.scale = scale_factor
            obj.metadata['axis_diameter'] = axis_diameter
            obj.metadata['scale_factor'] = scale_factor
            obj.metadata['mass_pre'] = obj.mass
            obj.mass *= np.power(scale_factor, 3.0)  # Volumetric scaling law is important here.
            obj.metadata['mass_post'] = obj.mass

            # Insert object into scene, ensuring that all contents remain disjoint.
            self.scene += obj
            rotation_axis = 'Z' if object_kind in ['box'] else None
            spawn_region = DYNAMIC_SPAWN_REGION
            self.move_until_no_overlap(
                obj, self.simulator, rotation_axis=rotation_axis, spawn_region=spawn_region,
                rng=self.random_state)

            # Assign random horizontal velocity with a strong bias toward the center of the scene.
            init_vel_independent = self.random_state.uniform(*VELOCITY_RANGE)
            init_pull_direction = np.array([obj.position[0], obj.position[1], 0.0])
            init_pull_factor = self.random_state.uniform(0.6, 1.2)
            init_vel_combined = init_vel_independent - init_pull_direction * init_pull_factor
            obj.velocity = init_vel_combined

            # NOTE: This (friction & restitution) is new relative to kubcon_v7.
            obj.friction = self.random_state.uniform(*self.object_friction_range)
            obj.restitution = self.random_state.uniform(*self.object_restit_range)
            obj.metadata['init_pull_factor'] = init_pull_factor
            obj.metadata['insert_order'] = i
            obj.metadata['object_kind'] = object_kind
            obj.metadata['is_dynamic'] = True
            obj.metadata['is_snitch'] = False

        return time.time() - start_time

    def insert_snitch(self, at_x=0.0, at_y=0.0, at_z=4.5, vel_x=0.0, vel_y=0.0, vel_z=-3.5,
                      size_meters=0.35, gso_asset_id='Vtech_Roll_Learn_Turtle'):
        '''
        X
        '''
        start_time = time.time()

        obj = self.gso_source.create(asset_id=gso_asset_id, name='snitch')
        assert isinstance(obj, self.kb_module.FileBasedObject)

        # Assign fixed size to snitch; smaller than almost all other objects.
        axis_diameter = size_meters
        scale_factor = axis_diameter / np.max(obj.bounds[1] - obj.bounds[0])
        obj.scale = scale_factor
        obj.metadata['axis_diameter'] = axis_diameter
        obj.metadata['scale_factor'] = scale_factor

        # Insert snitch into scene at desired location.
        # NOTE: This happens without regard for intersections.
        self.scene += obj
        obj.position = np.array([at_x, at_y, at_z])
        obj.velocity = np.array([vel_x, vel_y, vel_z])
        obj.metadata['is_dynamic'] = True
        obj.metadata['is_snitch'] = True

        self.scene.metadata['insert_snitch_args'] = dict()
        self.scene.metadata['insert_snitch_args']['at'] = (at_x, at_y, at_z)
        self.scene.metadata['insert_snitch_args']['vel'] = (vel_x, vel_y, vel_z)
        self.scene.metadata['insert_snitch_args']['size_meters'] = size_meters
        self.scene.metadata['insert_snitch_args']['gso_asset_id'] = gso_asset_id

        return time.time() - start_time

    def reset_objects_velocity_friction_restitution(self):
        '''
        X
        '''
        start_time = time.time()

        for obj in self.scene.foreground_assets:
            if hasattr(obj, 'velocity'):
                obj.velocity = np.array([0.0, 0.0, 0.0])
                obj.friction = self.random_state.uniform(*self.object_friction_range)
                obj.restitution = self.random_state.uniform(*self.object_restit_range)

        # Correct floor phyisical properties.
        self.dome.friction = self.random_state.uniform(*self.dome_friction_range)
        self.dome.restitution = self.random_state.uniform(*self.dome_restit_range)

        return time.time() - start_time

    def perturb_object_positions(self, max_offset_meters=0.01):
        '''
        Randomly translate all foreground objects by uniformly random noise vectors.
        '''
        start_time = time.time()

        for obj in self.scene.foreground_assets:

            # Sample perturbation, but ensure we can't glitch into the floor.
            translation_x = self.random_state.uniform(-max_offset_meters, max_offset_meters)
            translation_y = self.random_state.uniform(-max_offset_meters, max_offset_meters)
            translation_z = self.random_state.uniform(0.0, max_offset_meters)
            translation_meters = np.array([translation_x, translation_y, translation_z])

            # Apply offset to instance position.
            obj.position = obj.position + translation_meters
            obj.metadata['perturbation'] = translation_meters

        return time.time() - start_time

    def simulate_frames(self, frame_start, frame_end):
        '''
        :param frame_start (int): First frame (inclusive) to simulate.
        :param frame_end (int): Last frame (inclusive) to simulate.
        :return (animations, collisions, runtime).
            runtime (float): Total runtime of this call in seconds.
        '''
        start_time = time.time()

        (animations, collisions) = self.simulator.run(frame_start, frame_end)

        self.last_animations = animations
        self.last_collisions = collisions

        self.last_data_stack = None  # Mark as invalidated (must call render_frames again).

        return (animations, collisions, time.time() - start_time)

    def render_frames(self, frame_start, frame_end,
                      return_layers=['rgba', 'forward_flow', 'depth', 'normal',
                                     'object_coordinates', 'segmentation']):
        '''
        :param frame_start (int): First frame (inclusive) to render.
        :param frame_end (int): Last frame (inclusive) to render.
        :param return_layers (list of str): Keys of data modalities to process.
        :return (data_stack, runtime).
            data_stack: Dictionary containing a subset or all of these items:
                rgba: (T, H, W, 4) uint8.
                forward_flow: (T, H, W, 2) float32.
                depth: (T, H, W, 1) float32.
                normal: (T, H, W, 3) uint16.
                object_coordinates: (T, H, W, 3) uint16.
                segmentation: (T, H, W, 1) uint32.
            runtime (float): Total runtime of this call in seconds.
        '''
        start_time = time.time()

        # Set number of CPU cores to use (again), if specified.
        # https://docs.blender.org/api/current/bpy.types.RenderSettings.html
        if self.render_cpu_threads >= 1:
            self.bpy_module.context.scene.render.threads_mode = 'FIXED'
            self.bpy_module.context.scene.render.threads = self.render_cpu_threads

        # Render the selected subset of frames.
        data_stack = self.renderer.render(
            frames=list(range(frame_start, frame_end + 1)), return_layers=return_layers)

        # Perform postprocessing to obtain higher-level annotations.
        # Rank such that id = 1 is visually the biggest / most visible.
        # NOTE: Somewhat counterintuitively, some instances will *never* be visible, so their
        # visibility value will be exactly 0. We can ignore these objects by focusing on
        # sorted_valo_fg_assets.
        self.kb_module.compute_visibility(data_stack['segmentation'], self.scene.assets)
        self.sorted_fg_assets = sorted(
            self.scene.foreground_assets, key=lambda asset: np.sum(asset.metadata['visibility']),
            reverse=True)
        self.sorted_valo_fg_assets = [asset for asset in self.sorted_fg_assets
                                      if np.max(asset.metadata['visibility']) > 0]
        data_stack['segmentation'] = self.kb_module.adjust_segmentation_idxs(
            data_stack['segmentation'], self.scene.assets, self.sorted_fg_assets)
        self.scene.metadata['num_instances'] = len(self.sorted_fg_assets)
        self.scene.metadata['num_valo_instances'] = len(self.sorted_valo_fg_assets)
        # NOTE: How to interpret the segmentation array contents?:
        # When the value of data_stack['segmentation'] is 5, we point to self.sorted_fg_assets[4].
        # Therefore we also point to metadata['instances'][4]; this list is created in get_metadata.
        # Finally, when the segmentation value is 0, this means background (i.e. dome) because it is
        # not in the list of instances / foreground assets.

        # Store results for if we want to obtain metadata later.
        self.last_data_stack = data_stack
        
        self.last_div_data = None  # Mark as unused or not yet calculated.

        return (data_stack, time.time() - start_time)

    def render_frames_divided_objects(self, frame_start, frame_end,
                    return_layers=['rgba', 'depth', 'segmentation']):
        '''
        :param frame_start (int): First frame (inclusive) to render.
        :param frame_end (int): Last frame (inclusive) to render.
        Isolates all objects that are visible at least once (VALO) and renders a full video
            separately for each of them (by making all others invisible), in order to generate
            complete instance segmentation masks that always persist through occlusions otherwise
            caused by other objects in the input video.
            NOTE: This functionality is by definition very thorough and therefore expensive.
        :return (div_data, runtime).
            div_data (dict): Similar to data_stack.
                rgba: (T, H, W, 4, K) uint8.
                depth: (T, H, W, 1, K) float32.
                segmentation: (T, H, W, 1, K) uint8.
                For every array, channel dimension follows the order of sorted_(valo_)fg_assets.
            runtime (float): Total runtime of this call in seconds.
        '''
        start_time = time.time()

        (T, H, W, _) = self.last_data_stack['segmentation'].shape
        K = len(self.sorted_valo_fg_assets)

        object_scales = copy.deepcopy([asset.scale for asset in self.sorted_valo_fg_assets])
        raw_stacks = []
        self.logger.info(f'Called render_frames_divided_objects with {K} VALO instances...')

        for k in range(K):
            # Make all objects invisible except the current one.
            # NOTE: There is no direct visibility flag available, so we have to resize it into
            # nothing instead.
            for asset in self.sorted_fg_assets:
                asset.scale = 0.0
            self.sorted_valo_fg_assets[k].scale = copy.deepcopy(object_scales[k])

            # Re-render the selected subset of frames.
            data_stack = self.renderer.render(
                frames=list(range(frame_start, frame_end + 1)), return_layers=return_layers)

            # Reorder IDs to match render_frames(), such that the array values should always become
            # k + 1 at iteration k.
            data_stack['segmentation'] = self.kb_module.adjust_segmentation_idxs(
                data_stack['segmentation'], self.scene.assets, self.sorted_fg_assets)

            # Store current render output.
            raw_stacks.append(data_stack)

        # Construct array with combined results.
        div_data = dict()
        for layer in return_layers:
            div_data[layer] = np.stack([data_stack[layer] for data_stack in raw_stacks], axis=-1)
        
        self.last_div_data = div_data

        return (div_data, time.time() - start_time)

    def get_metadata(self, exclude_collisions=False):
        '''
        :param exclude_collisions (bool): Exclude collisions from metadata.
        '''
        start_time = time.time()

        self.kb_module.post_processing.compute_bboxes(
            self.last_data_stack['segmentation'], self.sorted_fg_assets)

        metadata = dict()
        metadata['scene'] = self.kb_module.get_scene_metadata(self.scene)
        metadata['camera'] = self.kb_module.get_camera_info(self.scene.camera)
        metadata['instances'] = self.kb_module.get_instance_info(self.scene, self.sorted_fg_assets)

        # NOTE: collisions generate a rather huge amount of information (~20 MB), so we omit them
        # for views beyond the first one.
        if not exclude_collisions:
            metadata['collisions'] = self.kb_module.process_collisions(
                self.last_collisions, self.scene, assets_subset=self.sorted_fg_assets)

        metadata['dome'] = self.kb_module.get_instance_info(self.scene, [self.dome])[0]
        metadata['random_seed'] = self.random_seed

        # Store results for if we want to obtain annotations later.
        self.last_metadata = metadata

        return (metadata, time.time() - start_time)

    def write_all_data(self, output_dir):
        '''
        X
        '''
        start_time = time.time()

        os.makedirs(output_dir, exist_ok=True)
        self.kb_module.write_image_dict(self.last_data_stack, output_dir, is_divided=False)
        
        if self.last_div_data is not None:
            self.kb_module.write_image_dict(self.last_div_data, output_dir, is_divided=True)

        return time.time() - start_time

    def save_simulator_state(self):
        '''
        X
        '''
        pass

    def save_renderer_state(self):
        '''
        X
        '''
        pass

    def construct_linear_camera_motion(
            self, movement_speed: float, inner_radius: float = 9.0, outer_radius: float = 12.0,
            z_offset: float = 0.1, fix_start=None):
        '''
        Sample a linear path which starts and ends within a half-sphere shell.
        '''
        for _ in range(1024):
            if fix_start is None:
                camera_start = np.array(self.kb_module.sample_point_in_half_sphere_shell(
                    inner_radius, outer_radius, z_offset))
            else:
                camera_start = np.array(fix_start)
            direction = self.random_state.rand(3) - 0.5
            movement = direction / (np.linalg.norm(direction) + 1e-7) * movement_speed
            camera_end = camera_start + movement
            if (inner_radius <= np.linalg.norm(camera_end) <= outer_radius and
                    camera_end[2] > z_offset):
                return camera_start, camera_end
        raise RuntimeError('Could not find a valid camera path.')

    def move_until_no_overlap(self, asset, simulator, rotation_axis=None,
                              spawn_region=((-1, -1, -1), (1, 1, 1)), max_trials=100, rng=None):
        return self.kb_module.resample_while(
            asset,
            samplers=[self.kb_module.rotation_sampler(axis=rotation_axis),
                      self.kb_module.position_sampler(spawn_region)],
            condition=simulator.check_overlap,
            max_trials=max_trials,
            rng=rng)


def _rotate_yaw(xyz, yaw_deg):
    xyz = xyz.astype(np.float64)
    yaw_rad = yaw_deg * np.pi / 180.0
    rotation_matrix = np.array([[np.cos(yaw_rad), -np.sin(yaw_rad), 0.0],
                                [np.sin(yaw_rad), np.cos(yaw_rad), 0.0],
                                [0.0, 0.0, 1.0]], dtype=np.float64)
    xyz_new = np.matmul(xyz, rotation_matrix.T)
    xyz_new = xyz_new.astype(np.float32)
    return xyz_new


if __name__ == '__main__':

    import kubric as kb
    import logvisgen

    np.set_printoptions(precision=3, suppress=True)

    if 0:
        frame_width = 256
        frame_height = 192
        num_frames = 24
        frame_rate = num_frames // 2

    if 1:
        frame_width = 512
        frame_height = 384
        num_frames = 60
        frame_rate = num_frames // 2

    logger = logvisgen.Logger()

    # NOTE: This instance must only be created once per process!
    my_kubric = MyKubricSimulatorRenderer(
        logger, frame_width=frame_width, frame_height=frame_height, num_frames=num_frames,
        frame_rate=frame_rate)

    for trial in range(4):

        cur_name = f'data_kubric_sim_v4_{trial}'

        start_time = time.time()

        t = my_kubric.prepare_next_scene('train', 334 + trial)
        logger.info(f'prepare_next_scene took {t:.2f}s')

        t = my_kubric.insert_static_objects(force_containers=trial, force_carriers=trial)
        logger.info(f'insert_static_objects took {t:.2f}s')

        (_, _, t) = my_kubric.simulate_frames(-60, -1)
        logger.info(f'simulate_frames took {t:.2f}s')

        t = my_kubric.reset_objects_velocity_friction_restitution()
        logger.info(f'reset_objects_velocity_friction_restitution took {t:.2f}s')

        t = my_kubric.insert_dynamic_objects()
        logger.info(f'insert_dynamic_objects took {t:.2f}s')

        (_, _, t) = my_kubric.simulate_frames(0, 1)
        logger.info(f'simulate_frames took {t:.2f}s')

        (data_stack1, t) = my_kubric.render_frames(0, 1)
        logger.info(f'render_frames took {t:.2f}s')

        t = my_kubric.insert_snitch()
        logger.info(f'insert_snitch took {t:.2f}s')

        (_, _, t) = my_kubric.simulate_frames(2, num_frames - 1)
        logger.info(f'simulate_frames took {t:.2f}s')

        (data_stack2, t) = my_kubric.render_frames(2, num_frames - 1)
        logger.info(f'render_frames took {t:.2f}s')

        # NOTE: data_stack2 contains both first and second render call results!

        (metadata, t) = my_kubric.get_metadata()
        logger.info(f'get_metadata took {t:.2f}s')

        rgb_annot = data_stack2['rgba'][..., :3].copy()
        (T, H, W, _) = rgb_annot.shape
        snitch_metadata = [x for x in metadata['instances']
                           if x['asset_id'] == 'Vtech_Roll_Learn_Turtle'
                           and x['axis_diameter'] == 0.35][0]

        snitch_bboxes = list(zip(snitch_metadata['bbox_frames'], snitch_metadata['bboxes']))
        for (t, bbox) in snitch_bboxes:
            (x1, x2) = int(bbox[1] * W), int(bbox[3] * W)
            (y1, y2) = int(bbox[0] * H), int(bbox[2] * H)
            rgb_annot[t, y1 - 2:y1, x1 - 2:x2 + 2, :] = [255, 255, 0]
            rgb_annot[t, y2:y2 + 2, x1 - 2:x2 + 2, :] = [255, 255, 0]
            rgb_annot[t, y1 - 2:y2 + 2, x1 - 2:x1, :] = [255, 255, 0]
            rgb_annot[t, y1 - 2:y2 + 2, x2:x2 + 2, :] = [255, 255, 0]

        snitch_imgpos = snitch_metadata['image_positions']
        for (t, imgpos) in enumerate(snitch_imgpos):
            (x, y) = int(imgpos[0] * W), int(imgpos[1] * H)
            rgb_annot[t, y - 2:y + 2, x - 2:x + 2, :] = [0, 255, 255]

        imageio.mimwrite(f'experimental/unit_test/{cur_name}_rgb.gif',
                         rgb_annot, format='gif', fps=frame_rate)
        imageio.mimwrite(f'experimental/unit_test/{cur_name}_rgb.mp4',
                         rgb_annot, format='ffmpeg', fps=frame_rate, quality=10)

        t = my_kubric.write_all_data(f'experimental/unit_test/{cur_name}')
        logger.info(f'write_all_data took {t:.2f}s')

        kb.write_json(metadata, f'experimental/unit_test/{cur_name}_metadata.json')

        logger.info(f'All together took {time.time() - start_time:.2f}s')

    logger.info()

    pass
