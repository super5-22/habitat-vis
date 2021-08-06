# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import math
import multiprocessing
import os
import random
import time
from enum import Enum

import numpy as np
import cv2 as cv
from PIL import Image
from settings import default_sim_settings, make_cfg
from vissim import HabitatVisSim
from utils import list2dict, observation2image
from interaction import INTERACTION_MAPPING, MODE_MAPPING

import habitat_sim
import habitat_sim.agent
import habitat_sim.utils.datasets_download as data_downloader
from habitat_sim.nav import ShortestPath
from habitat_sim.physics import MotionType
from habitat_sim.utils.common import d3_40_colors_rgb, quat_from_angle_axis
from habitat_sim.utils import viz_utils as vut

_barrier = None


class DemoRunnerType(Enum):
    BENCHMARK = 1
    EXAMPLE = 2
    AB_TEST = 3


class ABTestGroup(Enum):
    CONTROL = 1
    TEST = 2


class DemoRunner:
    def __init__(self, sim_settings, simulator_demo_type):
        if simulator_demo_type == DemoRunnerType.EXAMPLE:
            self.set_sim_settings(sim_settings)
        self._demo_type = simulator_demo_type

    def set_sim_settings(self, sim_settings):
        self._sim_settings = sim_settings.copy()

    def save_color_observation(self, obs, total_frames):
        color_obs = obs["color_sensor"]
        color_img = Image.fromarray(color_obs, mode="RGBA")
        if self._demo_type == DemoRunnerType.AB_TEST:
            if self._group_id == ABTestGroup.CONTROL:
                color_img.save("test.rgba.control.%05d.png" % total_frames)
            else:
                color_img.save("test.rgba.test.%05d.png" % total_frames)
        else:
            color_img.save("test.rgba.%05d.png" % total_frames)

    def save_semantic_observation(self, obs, total_frames):
        semantic_obs = obs["semantic_sensor"]
        semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
        semantic_img.putpalette(d3_40_colors_rgb.flatten())
        semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
        if self._demo_type == DemoRunnerType.AB_TEST:
            if self._group_id == ABTestGroup.CONTROL:
                semantic_img.save("test.sem.control.%05d.png" % total_frames)
            else:
                semantic_img.save("test.sem.test.%05d.png" % total_frames)
        else:
            semantic_img.save("test.sem.%05d.png" % total_frames)

    def save_depth_observation(self, obs, total_frames):
        depth_obs = obs["depth_sensor"]
        depth_img = Image.fromarray((depth_obs / 10 * 255).astype(np.uint8), mode="L")
        if self._demo_type == DemoRunnerType.AB_TEST:
            if self._group_id == ABTestGroup.CONTROL:
                depth_img.save("test.depth.control.%05d.png" % total_frames)
            else:
                depth_img.save("test.depth.test.%05d.png" % total_frames)
        else:
            depth_img.save("test.depth.%05d.png" % total_frames)

    def output_semantic_mask_stats(self, obs, total_frames):
        semantic_obs = obs["semantic_sensor"]
        counts = np.bincount(semantic_obs.flatten())
        total_count = np.sum(counts)
        print(f"Pixel statistics for frame {total_frames}")
        for object_i, count in enumerate(counts):
            sem_obj = self._sim.semantic_scene.objects[object_i]
            cat = sem_obj.category.name()
            pixel_ratio = count / total_count
            if pixel_ratio > 0.01:
                print(f"obj_id:{sem_obj.id},category:{cat},pixel_ratio:{pixel_ratio}")

    def init_agent_state(self, agent_id):
        # initialize the agent at a random start state
        agent = self._sim.initialize_agent(agent_id)
        start_state = agent.get_state()

        # force starting position on first floor (try 100 samples)
        num_start_tries = 0
        while start_state.position[1] > 0.5 and num_start_tries < 100:
            start_state.position = self._sim.pathfinder.get_random_navigable_point()
            num_start_tries += 1
        agent.set_state(start_state)

        if not self._sim_settings["silent"]:
            print(
                "start_state.position\t",
                start_state.position,
                "start_state.rotation\t",
                start_state.rotation,
            )

        return start_state

    def compute_shortest_path(self, start_pos, end_pos):
        self._shortest_path.requested_start = start_pos
        self._shortest_path.requested_end = end_pos
        self._sim.pathfinder.find_path(self._shortest_path)
        print("shortest_path.geodesic_distance", self._shortest_path.geodesic_distance)

    def _interactive_keybutton(self, keyboard_button, total_action, action_names):
        # TODO: this logic is a little bit dirty
        # Control the agent or the 3rd view
        if keyboard_button in INTERACTION_MAPPING.keys():
            if self._sim.control_agent_id == 0:
                total_action[0] = INTERACTION_MAPPING[keyboard_button]
                print(f"Press keyboard: {keyboard_button}, execute {INTERACTION_MAPPING[keyboard_button]}")
            else:
                if self._sim.user_mode:
                    if INTERACTION_MAPPING[keyboard_button] in action_names:
                        total_action[self._sim.display_agent_id] = INTERACTION_MAPPING[keyboard_button]
                        print(f"Press keyboard: {keyboard_button}, execute {INTERACTION_MAPPING[keyboard_button]}")
        else:
            if self._sim.user_mode and self._sim.control_agent_id != 0:
                try:
                    total_action.pop(self._sim.control_agent_id)
                except KeyError:
                    # for keys not as agents' actions
                    pass

        not_destroy = self._sim.change_display(keyboard_button)
        try:
            print(f"Press keyboard: {keyboard_button}, execute {MODE_MAPPING[keyboard_button]}")
        except KeyError:
            print(f"Press keyboard: {keyboard_button}")

        return total_action, not_destroy

    def do_time_steps(self):
        total_sim_step_time = 0.0
        total_frames = 0
        start_time = time.time()
        try:
            action_names = list(
                self._cfg.agents[1].action_space.keys()
            )
        except IndexError:
            action_names = list(self._cfg.agents[0].action_space.keys())

        observations = self._sim.reset()
        agents_num = len(self._sim.agents)

        not_destroy = True

        while True:
            print(f"Simulating {total_frames} frames")
            if total_frames == 1:
                start_time = time.time()
            action_start = time.time()
            total_action = {}
            for i in range(1, agents_num):
                action = random.choice(action_names)
                total_action[i] = action
            action_time = time.time() - action_start
            if not self._sim_settings["silent"]:
                print("action", total_action, "action_time:", action_time)

            imshow_start = time.time()
            keyboard_botton = self.interactive_display(observations, not_destroy)
            imshow_end = time.time()
            print(f"imshow: {imshow_end - imshow_start}s, {1.0 / (imshow_end - imshow_start)}fps")

            total_action, not_destroy = self._interactive_keybutton(keyboard_botton, total_action, action_names)

            start_step_time = time.time()

            print("keyboard time:", start_step_time - imshow_end)

            observations = self._sim.step(total_action)
            time_per_step = time.time() - start_step_time

            # get simulation step time without sensor observations
            total_sim_step_time += self._sim._previous_step_time

            if self._sim_settings["save_png"]:
                if self._sim_settings["color_sensor"]:
                    self.save_color_observation(observations, total_frames)
                if self._sim_settings["depth_sensor"]:
                    self.save_depth_observation(observations, total_frames)
                if self._sim_settings["semantic_sensor"]:
                    self.save_semantic_observation(observations, total_frames)

            state = self._sim.last_state()

            if not self._sim_settings["silent"]:
                print("position\t", state.position, "\t", "rotation\t", state.rotation)

            if self._sim_settings["compute_shortest_path"]:
                self.compute_shortest_path(
                    state.position, self._sim_settings["goal_position"]
                )

            if self._sim_settings["compute_action_shortest_path"]:
                self._action_path = self.greedy_follower.find_path(
                    self._sim_settings["goal_position"]
                )
                print("len(action_path)", len(self._action_path))

            if (
                self._sim_settings["semantic_sensor"]
                and self._sim_settings["print_semantic_mask_stats"]
            ):
                self.output_semantic_mask_stats(observations, total_frames)

            total_frames += 1

            if keyboard_botton == ord('q'):
                break
            end_time = time.time()
            frame_time = end_time - action_start
            print(f"frame_time: {frame_time}, fps: {1.0 / frame_time}, "
                  f"per_step_time: {time_per_step}, sps: {1.0 / time_per_step}")

        perf = {}
        perf["total_time"] = end_time - start_time
        perf["frame_time"] = perf["total_time"] / total_frames
        perf["fps"] = 1.0 / perf["frame_time"]
        perf["time_per_step"] = time_per_step
        perf["avg_sim_step_time"] = total_sim_step_time / total_frames

        return perf

    def interactive_display(self, observations, not_destroy):
        if not not_destroy:
            cv.destroyAllWindows()
        frame = observation2image(observations, self._sim.display_agent_id, self._sim.display_3rd_color)
        cv.imshow(self._sim.display_name, frame)
        keyboard_botton = cv.waitKey(1)
        return keyboard_botton

    def save_video(self, observations, k):
        if not os.path.exists(self._sim_settings['output_path']):
            os.makedirs(self._sim_settings['output_path'])
        for agent_id, v in observations.items():
            if k in v[0].keys():
                vut.make_video(
                    v,
                    k,
                    self._sim.SENSOR2TYPE[k],
                    self._sim_settings['output_path'] + f"/agent_{agent_id}_{k}",
                    open_vid=True,
                )

    def print_semantic_scene(self):
        if self._sim_settings["print_semantic_scene"]:
            scene = self._sim.semantic_scene
            print(f"House center:{scene.aabb.center} dims:{scene.aabb.sizes}")
            for level in scene.levels:
                print(
                    f"Level id:{level.id}, center:{level.aabb.center},"
                    f" dims:{level.aabb.sizes}"
                )
                for region in level.regions:
                    print(
                        f"Region id:{region.id}, category:{region.category.name()},"
                        f" center:{region.aabb.center}, dims:{region.aabb.sizes}"
                    )
                    for obj in region.objects:
                        print(
                            f"Object id:{obj.id}, category:{obj.category.name()},"
                            f" center:{obj.aabb.center}, dims:{obj.aabb.sizes}"
                        )
            input("Press Enter to continue...")

    def init_common(self):
        self._cfg, cfg = make_cfg(self._sim_settings)
        scene_file = self._sim_settings["scene"]

        if (
            not os.path.exists(scene_file)
            and scene_file == default_sim_settings["scene"]
        ):
            print(
                "Test scenes not downloaded locally, downloading and extracting now..."
            )
            data_downloader.main(["--uids", "habitat_test_scenes"])
            print("Downloaded and extracted test scenes data.")

        # create a simulator (Simulator python class object, not the backend simulator)
        self._sim = HabitatVisSim(cfg, self._cfg)

        random.seed(self._sim_settings["seed"])
        self._sim.seed(self._sim_settings["seed"])

        recompute_navmesh = self._sim_settings.get("recompute_navmesh")
        if recompute_navmesh or not self._sim.pathfinder.is_loaded:
            if not self._sim_settings["silent"]:
                print("Recomputing navmesh")
            navmesh_settings = habitat_sim.NavMeshSettings()
            navmesh_settings.set_defaults()
            self._sim.recompute_navmesh(self._sim.pathfinder, navmesh_settings)

        # initialize the agent at a random start state
        return self.init_agent_state(self._sim_settings["default_agent"])

    def _bench_target(self, _idx=0):
        self.init_common()

        best_perf = None
        for _ in range(3):

            if _barrier is not None:
                _barrier.wait()
                if _idx == 0:
                    _barrier.reset()

            perf = self.do_time_steps()
            # The variance introduced between runs is due to the worker threads
            # being interrupted a different number of times by the kernel, not
            # due to difference in the speed of the code itself.  The most
            # accurate representation of the performance would be a run where
            # the kernel never interrupted the workers, but this isn't
            # feasible, so we just take the run with the least number of
            # interrupts (the fastest) instead.
            if best_perf is None or perf["frame_time"] < best_perf["frame_time"]:
                best_perf = perf

        self._sim.close()
        del self._sim

        return best_perf

    @staticmethod
    def _pool_init(b):
        global _barrier
        _barrier = b

    def benchmark(self, settings, group_id=ABTestGroup.CONTROL):
        self.set_sim_settings(settings)
        nprocs = settings["num_processes"]
        # set it anyway, but only be used in AB_TEST mode
        self._group_id = group_id

        barrier = multiprocessing.Barrier(nprocs)
        with multiprocessing.Pool(
            nprocs, initializer=self._pool_init, initargs=(barrier,)
        ) as pool:
            perfs = pool.map(self._bench_target, range(nprocs))

        res = {k: [] for k in perfs[0].keys()}
        for p in perfs:
            for k, v in p.items():
                res[k] += [v]

        return dict(
            frame_time=sum(res["frame_time"]),
            fps=sum(res["fps"]),
            total_time=sum(res["total_time"]) / nprocs,
            avg_sim_step_time=sum(res["avg_sim_step_time"]) / nprocs,
        )

    def example(self):
        start_state = self.init_common()

        # initialize and compute shortest path to goal
        if self._sim_settings["compute_shortest_path"]:
            self._shortest_path = ShortestPath()
            self.compute_shortest_path(
                start_state.position, self._sim_settings["goal_position"]
            )

        # set the goal headings, and compute action shortest path
        if self._sim_settings["compute_action_shortest_path"]:
            agent_id = self._sim_settings["default_agent"]
            self.greedy_follower = self._sim.make_greedy_follower(agent_id=agent_id)

            self._action_path = self.greedy_follower.find_path(
                self._sim_settings["goal_position"]
            )
            print("len(action_path)", len(self._action_path))

        # print semantic scene
        self.print_semantic_scene()

        perf = self.do_time_steps()
        self._sim.close()
        del self._sim

        return perf
