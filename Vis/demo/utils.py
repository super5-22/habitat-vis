import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image

from habitat.utils.visualizations import maps
from habitat_sim.utils.common import d3_40_colors_rgb


# convert 3d points to 2d topdown coordinates
def convert_points_to_topdown(pathfinder, points, meters_per_pixel):
    points_topdown = []
    bounds = pathfinder.get_bounds()
    for point in points:
        # convert 3D x,z to topdown x,y
        px = (point[0] - bounds[0][0]) / meters_per_pixel
        py = (point[2] - bounds[0][2]) / meters_per_pixel
        points_topdown.append(np.array([px, py]))
    return points_topdown


# display a topdown map with matplotlib
def display_map(topdown_map, key_points=None, block=True):
    plt.figure(figsize=(12, 8))
    ax = plt.subplot(1, 1, 1)
    ax.axis("off")
    plt.imshow(topdown_map)
    # plot points on map
    if key_points is not None:
        for point in key_points:
            plt.plot(point[0], point[1], marker="o", markersize=10, alpha=0.8)
    plt.show(block=block)


def shortest_path_to_waypoints(shortest_path):
    # Convert dense waypoints of the shortest path to coarse waypoints
    # in which the collinear waypoints are merged.
    assert len(shortest_path) > 0

    if len(shortest_path[0]) == 3:
        shortest_path_temp = []
        for p in shortest_path:
            shortest_path_temp.append(np.array([p[0], p[2]]))
        shortest_path = shortest_path_temp

    waypoints = []
    valid_waypoint = None
    prev_waypoint = None
    cached_slope = None
    for waypoint in shortest_path:
        if valid_waypoint is None:
            valid_waypoint = waypoint
        elif cached_slope is None:
            cached_slope = waypoint - valid_waypoint
        else:
            cur_slope = waypoint - prev_waypoint
            cosine_angle = np.dot(cached_slope, cur_slope) / \
                           (np.linalg.norm(cached_slope) * np.linalg.norm(cur_slope))
            if np.abs(cosine_angle - 1.0) > 1e-3:
                waypoints.append(valid_waypoint)
                valid_waypoint = prev_waypoint
                cached_slope = waypoint - valid_waypoint

        prev_waypoint = waypoint

    # Add the last two valid waypoints
    waypoints.append(valid_waypoint)
    waypoints.append(shortest_path[-1])

    # Remove the first waypoint because it's the same as the initial pos
    waypoints.pop(0)

    return waypoints


#  convert 3d position to 2d top-down position
def xyzpos2xzpos(pos):
    return (pos[0], pos[2])


# convert 3d points to 2d topdown coordinates
def convert_points_to_topdown(pathfinder, points, meters_per_pixel):
    points_topdown = []
    bounds = pathfinder.get_bounds()
    for point in points:
        # convert 3D x,z to topdown x,y
        px = (point[0] - bounds[0][0]) / meters_per_pixel
        py = (point[2] - bounds[0][2]) / meters_per_pixel
        points_topdown.append((px, py))
    return points_topdown


# convert 2d topdown coordinates to 3d points in 2d view
def convert_topdown_to_2dpoints(pathfinder, points_topdown, meters_per_pixel):
    points2d = []
    bounds = pathfinder.get_bounds()
    for point in points_topdown:
        # convert topdown x,y to 3d x,z
        p2dx = point[0] * meters_per_pixel + bounds[0][0]
        p2dy = point[1] * meters_per_pixel + bounds[0][2]
        points2d.append((p2dx, p2dy))
    return points2d


# check hierarchy of contours whether it is a big contour containing a lof of child contours
def check_hierarchy(hierarchy):
    assert hierarchy[0, 0, -1] == -1, "the first contour is not the biggest contour"
    assert hierarchy[0, 1:, -1].sum() == 0, "some of the small contours have child contours"
    return True


def contours2obstacles(contours, map_shape):
    h, w = map_shape
    p1 = np.array([0.0, 0.0])
    p2 = np.array([1.0 * w, 0.0])
    p3 = np.array([1.0 * w, 1.0 * h])
    p4 = np.array([0.0, 1.0 * h])
    line1 = [p1, p2]
    line2 = [p3, p4]
    frontier = contours[0][:, 0, :]  # (N, 2)
    distance_line1 = points_distance_line(frontier, line1)
    distance_line2 = points_distance_line(frontier, line2)
    upper_point_ind = distance_line1.argmin()
    lower_point_ind = distance_line2.argmin()
    assert upper_point_ind != lower_point_ind
    obstacles = []
    frontier_obstacle = split_contour(frontier, (p1, p2, p3, p4), upper_point_ind, lower_point_ind)
    obstacles += frontier_obstacle
    obstacles += contours[1:]
    return obstacles


def split_contour(frontier, map, upper_ind, lower_ind):
    # The outer contour is counterclockwise while inner contour is clockwise, so here frontier is ccw
    lt, rt, rb, lb = map
    upper_point = frontier[upper_ind]
    upper_map_point = np.array([upper_point[0], lt[1]]).reshape(1, 2)
    lower_point = frontier[lower_ind]
    lower_map_point = np.array([lower_point[0], rb[1]]).reshape(1, 2)
    if upper_ind < lower_ind:
        left_contour = frontier[upper_ind: lower_ind + 1]
        right_contour = np.concatenate((frontier[lower_ind:], frontier[:upper_ind + 1]), axis=0)
    else:
        left_contour = np.concatenate((frontier[upper_ind:], frontier[:lower_ind + 1]), axis=0)
        right_contour = frontier[lower_ind: upper_ind + 1]
    lt = lt.reshape(1, 2)
    rt = rt.reshape(1, 2)
    rb = rb.reshape(1, 2)
    lb = lb.reshape(1, 2)
    # We select from the upper_point->lower_point->lower_map_point->lb->lt->upper_map_point->upper_point to form left
    left_obstacle = np.concatenate((left_contour, lower_map_point, lb, lt, upper_map_point), axis=0).reshape(-1, 1, 2)
    # We select from the lower_point->upper_point->upper_map_point->rt->rb->lower_map_point->lower_point to form right
    right_obstacle = np.concatenate((right_contour, upper_map_point, rt, rb, lower_map_point), axis=0).reshape(-1, 1, 2)
    return [left_obstacle, right_obstacle]


def display_contours(map_uint8, contours):
    for idx, contour in enumerate(contours):
        # if idx == 0:
        #     continue
        N, _, _ = contour.shape
        cv.drawContours(map_uint8, [contour.astype(np.int32)], 0, 255, 3)
        cv.imshow(str(idx), map_uint8)
        cv.waitKey(0)


#  the distance of a point to a line (indicated by 2 points)
def point_distance_line(point, line):
    p1, p2 = line
    vec1 = p1 - point
    vec2 = p2 - point
    distance = np.abs(np.cross(vec1, vec2)) / np.linalg.norm(p1 - p2)
    return distance


# the distance of a lot of points (N, 2) to a line
def points_distance_line(points, line):
    p1, p2 = line
    A = p2[1] - p1[1]
    B = p1[0] - p2[0]
    C = (p1[1] - p2[1]) * p1[0] + (p2[0] - p1[0]) * p1[1]
    distance = np.abs(A * points[:, 0] + B * points[:, 1] + C) / np.sqrt(A ** 2 + B ** 2)
    return distance


def list2dict(list_obj):
    # only support depth = 1
    assert len(list_obj) > 0
    ks = list_obj[0].keys()
    temp_dict = {}
    for k in ks:
        temp_dict[k] = []
    for o in list_obj:
        for k, v in o.items():
            temp_dict[k].append(v)
    return temp_dict


def filter_rgb_sensor(obs):
    return obs[:, :, [2, 1, 0]]


def filter_depth_sensor(obs, min_depth=0.0, max_depth=10.0, norm=True):
    if isinstance(obs, np.ndarray):
        obs = np.clip(obs, min_depth, max_depth)

        obs = np.expand_dims(
            obs, axis=2
        )  # make depth observation a 3D array
    else:
        obs = obs.clamp(min_depth, max_depth)

        obs = obs.unsqueeze(-1)

    if norm:
        # normalize depth observation to [0, 1]
        obs = (obs - min_depth) / (
            max_depth - min_depth
        )

    return obs.squeeze()


def process_rgb(obs):
    rgb = filter_rgb_sensor(obs)
    if not isinstance(rgb, np.ndarray):
        rgb = rgb.cpu().numpy()
    return rgb


def process_depth(obs):
    depth_map = filter_depth_sensor(obs) * 255.0
    if not isinstance(depth_map, np.ndarray):
        depth_map = depth_map.cpu().numpy()

    depth_map = depth_map.astype(np.uint8)
    depth_map = np.stack([depth_map for _ in range(3)], axis=2)
    return depth_map

def process_semantic(obs, agent_id):
    semantic_obs = obs[agent_id]["semantic_sensor"]
    semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
    semantic_img.putpalette(d3_40_colors_rgb.flatten())
    semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
    semantic_img = semantic_img.convert("RGBA")
    semantic_img = np.array(semantic_img)[..., : -1]
    return semantic_img


def observation2image(observation, agent_id, display_color=True):
    egocentric_view_l = []
    # the forth channel indicates alpha
    rgb = process_rgb(observation[agent_id]["color_sensor"][..., :-1])
    egocentric_view_l.append(rgb)
    # egocentric_view_l.append(rgb)

    # draw depth map
    depth_map = process_depth(observation[agent_id]["depth_sensor"].squeeze())
    egocentric_view_l.append(depth_map)

    # draw semantic map
    semantic_map = process_semantic(observation, agent_id)
    egocentric_view_l.append(semantic_map)

    assert (
        len(egocentric_view_l) > 0
    ), "Expected at least one visual sensor enabled."
    egocentric_view = np.concatenate(egocentric_view_l, axis=0)

    if display_color:
        view_3rd = process_rgb(observation[0]["color_sensor"][..., :-1])
    else:
        # view_3rd = process_depth(observation[0]["depth_sensor"].squeeze())  -- transform to depth
        # view_3rd = semantic_img    --transform to semantic
        view_3rd = maps.colorize_draw_agent_and_fit_to_height(
            observation[agent_id]["top_down_map"], 600
        )

    frame = np.concatenate([view_3rd, egocentric_view], axis=1)

    # Draw top_down_map is time-consuming,
    # the colorize costs about 80% of drawing, the default color and fog color are half and half

    # if "top_down_map" in observation.keys():
    #     top_down_map = maps.colorize_draw_agent_and_fit_to_height(
    #         observation["top_down_map"], egocentric_view.shape[0]
    #     )
    #     frame = np.concatenate((egocentric_view, top_down_map), axis=1)
    return frame
