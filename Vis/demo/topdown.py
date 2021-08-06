import cv2

import numpy as np
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import (
    quaternion_from_coeff,
    quaternion_rotate_vector,
)
from habitat.utils.visualizations import fog_of_war, maps


MAP_THICKNESS_SCALAR: int = 128


class TopDownMap:

    def __init__(self, sim):
        self._sim = sim
        self._grid_delta = 3
        self._step_count = None
        self._map_resolution = 1024
        self._previous_xy_location = [None for _ in range(len(self._sim.agents))]
        self._top_down_map = None
        self._shortest_path_points = None
        self.line_thickness = int(
            np.round(self._map_resolution * 2 / MAP_THICKNESS_SCALAR)
        )
        self.point_padding = 2 * int(
            np.ceil(self._map_resolution / MAP_THICKNESS_SCALAR)
        )
        self.get_original_map()

    def get_original_map(self):
        top_down_maps = []
        fog_mask = []
        for i in range(len(self._sim.agents)):
            top_down_map = maps.get_topdown_map_from_sim(
                self._sim,
                map_resolution=self._map_resolution,
                draw_border=True,
            )
            fog_of_war_mask = np.zeros_like(top_down_map)
            top_down_maps.append(top_down_map)
            fog_mask.append(fog_of_war_mask)
        self._top_down_map = top_down_maps
        self._fog_of_war_mask = fog_mask

    def _draw_point(self, position, point_type):
        t_x, t_y = maps.to_grid(
            position[2],
            position[0],
            self._top_down_map.shape[0:2],
            sim=self._sim,
        )
        self._top_down_map[
            t_x - self.point_padding : t_x + self.point_padding + 1,
            t_y - self.point_padding : t_y + self.point_padding + 1,
        ] = point_type

    def _draw_goals_view_points(self, episode):
        for goal in episode.goals:
            if self._is_on_same_floor(goal.position[1]):
                try:
                    if goal.view_points is not None:
                        for view_point in goal.view_points:
                            self._draw_point(
                                view_point.agent_state.position,
                                maps.MAP_VIEW_POINT_INDICATOR,
                            )
                except AttributeError:
                    pass

    def _draw_goals_positions(self, episode):
        for goal in episode.goals:
            if self._is_on_same_floor(goal.position[1]):
                try:
                    self._draw_point(
                        goal.position, maps.MAP_TARGET_POINT_INDICATOR
                    )
                except AttributeError:
                    pass

    def _draw_goals_aabb(self, episode):
        for goal in episode.goals:
            try:
                sem_scene = self._sim.semantic_annotations()
                object_id = goal.object_id
                assert int(
                    sem_scene.objects[object_id].id.split("_")[-1]
                ) == int(
                    goal.object_id
                ), f"Object_id doesn't correspond to id in semantic scene objects dictionary for episode: {episode}"

                center = sem_scene.objects[object_id].aabb.center
                x_len, _, z_len = (
                    sem_scene.objects[object_id].aabb.sizes / 2.0
                )
                # Nodes to draw rectangle
                corners = [
                    center + np.array([x, 0, z])
                    for x, z in [
                        (-x_len, -z_len),
                        (-x_len, z_len),
                        (x_len, z_len),
                        (x_len, -z_len),
                        (-x_len, -z_len),
                    ]
                    if self._is_on_same_floor(center[1])
                ]

                map_corners = [
                    maps.to_grid(
                        p[2],
                        p[0],
                        self._top_down_map.shape[0:2],
                        sim=self._sim,
                    )
                    for p in corners
                ]

                maps.draw_path(
                    self._top_down_map,
                    map_corners,
                    maps.MAP_TARGET_BOUNDING_BOX,
                    self.line_thickness,
                )
            except AttributeError:
                pass

    def _draw_shortest_path(
        self, episode, agent_position
    ):
        _shortest_path_points = (
            self._sim.get_straight_shortest_path_points(
                agent_position, episode.goals[0].position
            )
        )
        self._shortest_path_points = [
            maps.to_grid(
                p[2], p[0], self._top_down_map.shape[0:2], sim=self._sim
            )
            for p in _shortest_path_points
        ]
        maps.draw_path(
            self._top_down_map,
            self._shortest_path_points,
            maps.MAP_SHORTEST_PATH_COLOR,
            self.line_thickness,
        )

    def _is_on_same_floor(
        self, height, ref_floor_height=None, ceiling_height=2.0
    ):
        if ref_floor_height is None:
            ref_floor_height = self._sim.get_agent(0).state.position[1]
        return ref_floor_height < height < ref_floor_height + ceiling_height

    def reset_metric(self, episode):
        self._step_count = 0
        self.get_original_map()
        for i in range(len(self._sim.agents)):
            agent_position = self._sim.get_agent_state(i).position
            a_x, a_y = maps.to_grid(
                agent_position[2],
                agent_position[0],
                self._top_down_map[i].shape[0:2],
                sim=self._sim,
            )
            self._previous_xy_location[i] = (a_y, a_x)

            self.update_fog_of_war_mask(np.array([a_x, a_y]), i)

            # draw source and target parts last to avoid overlap
            # self._draw_goals_view_points(episode)
            # self._draw_goals_aabb(episode)
            # self._draw_goals_positions(episode)

            # self._draw_shortest_path(episode, agent_position)

            # self._draw_point(
            #     episode.start_position, maps.MAP_SOURCE_POINT_INDICATOR
            # )

    def update_metric(self, agent_id):
        self._step_count += 1
        for i in range(len(self._sim.agents)):
            house_map, map_agent_x, map_agent_y = self.update_map(self._sim.get_agent_state(i).position, i)
            if i == agent_id:
                self._metric = {
                    "map": house_map,
                    "fog_of_war_mask": self._fog_of_war_mask[agent_id],
                    "agent_map_coord": (map_agent_x, map_agent_y),
                    "agent_angle": self.get_polar_angle(agent_id),
                }
        return self._metric

    def get_polar_angle(self, agent_id):
        agent_state = self._sim.get_agent_state(agent_id)
        # quaternion is in x, y, z, w format
        ref_rotation = agent_state.rotation

        heading_vector = quaternion_rotate_vector(
            ref_rotation.inverse(), np.array([0, 0, -1])
        )

        phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        z_neg_z_flip = np.pi
        return np.array(phi) + z_neg_z_flip

    def update_map(self, agent_position, agent_id):
        a_x, a_y = maps.to_grid(
            agent_position[2],
            agent_position[0],
            self._top_down_map[agent_id].shape[0:2],
            sim=self._sim,
        )
        # Don't draw over the source point
        if self._top_down_map[agent_id][a_x, a_y] != maps.MAP_SOURCE_POINT_INDICATOR:
            color = 10 + min(
                self._step_count * 245 // 1000, 245
            )
            # / MAX steps

            thickness = self.line_thickness
            cv2.line(
                self._top_down_map[agent_id],
                self._previous_xy_location[agent_id],
                (a_y, a_x),
                color,
                thickness=thickness,
            )
        self.update_fog_of_war_mask(np.array([a_x, a_y]), agent_id)

        self._previous_xy_location[agent_id] = (a_y, a_x)
        return self._top_down_map[agent_id], a_x, a_y

    def update_fog_of_war_mask(self, agent_position, agent_id):
        self._fog_of_war_mask[agent_id] = fog_of_war.reveal_fog_of_war(
            self._top_down_map[agent_id],
            self._fog_of_war_mask[agent_id],
            agent_position,
            self.get_polar_angle(agent_id),
            fov=90,
            max_line_len=5.0
            / maps.calculate_meters_per_pixel(
                self._map_resolution, sim=self._sim
            ),
        )
