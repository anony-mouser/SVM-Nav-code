import cv2
import torch
import numpy as np
from typing import Tuple
import torch.nn.functional as F
from collections import Sequence
from scipy.spatial.distance import cdist
from skimage.morphology import remove_small_objects, closing, disk, dilation

from vlnce_baselines.utils.pose import get_agent_position, threshold_poses


def get_grid(pose: torch.Tensor, grid_size: Tuple, device: torch.device):
    """
    Input:
        `pose` FloatTensor(bs, 3)
        `grid_size` 4-tuple (bs, _, grid_h, grid_w)
        `device` torch.device (cpu or gpu)
    Output:
        `rot_grid` FloatTensor(bs, grid_h, grid_w, 2)
        `trans_grid` FloatTensor(bs, grid_h, grid_w, 2)

    """
    pose = pose.float()
    x = pose[:, 0]
    y = pose[:, 1]
    t = pose[:, 2]

    bs = x.size(0)
    t = t * np.pi / 180. # t.shape=[bs]
    cos_t = t.cos()
    sin_t = t.sin()

    theta11 = torch.stack([cos_t, -sin_t,
                           torch.zeros(cos_t.shape).float().to(device)], 1) # [bs, 3]
    theta12 = torch.stack([sin_t, cos_t,
                           torch.zeros(cos_t.shape).float().to(device)], 1)
    theta1 = torch.stack([theta11, theta12], 1) # [bs, 2, 3] rotation matrix

    theta21 = torch.stack([torch.ones(x.shape).to(device),
                           -torch.zeros(x.shape).to(device), x], 1)
    theta22 = torch.stack([torch.zeros(x.shape).to(device),
                           torch.ones(x.shape).to(device), y], 1)
    theta2 = torch.stack([theta21, theta22], 1) # [bs, 2, 3] translation matrix

    rot_grid = F.affine_grid(theta1, torch.Size(grid_size))
    trans_grid = F.affine_grid(theta2, torch.Size(grid_size))

    return rot_grid, trans_grid


def get_obstacle(map: np.ndarray, kernel_size: int=3) -> np.ndarray:
    obstacle = map[0, ...]
    obstacle = remove_small_objects(
        obstacle.astype(bool), 
        min_size=50, # you can try different minimum object size
        connectivity=5)
    selem = disk(kernel_size)
    obstacle = closing(obstacle, selem)
    
    return obstacle.astype(bool)


def get_objects(map: np.ndarray, kernel_size: int=3) -> np.ndarray:
    objects = np.zeros(map.shape[-2:])
    for obj in map[5:, ...]:
        obj = remove_small_objects(obj.astype(bool), min_size=50)
        selem = disk(kernel_size)
        obj = closing(obj, selem)
        objects += obj
    
    return objects.astype(bool)


def get_explored_area(map: np.ndarray, kernel_size: int=3) -> np.ndarray:
    explored_area = map[1, ...]
    selem = disk(kernel_size)
    explored_area = closing(explored_area, selem)
    explored_area = remove_small_objects(explored_area.astype(bool), min_size=200)
    
    return explored_area


def process_floor(map: np.ndarray, kernel_size: int=3) -> np.ndarray:
    explored_area = map[1, ...]
    obstacles = map[0, ...]
    objects = np.zeros(map.shape[-2:])
    for obj in map[5:, ...]:
        objects += obj
    free_mask = 1 - np.logical_or(obstacles, objects)
    free_space = explored_area * free_mask
    floor = remove_small_objects(free_space.astype(bool), min_size=500)
    floor = closing(floor, selem=disk(kernel_size))
    
    return floor


def find_frontiers(map: np.ndarray) -> np.ndarray:
    # floor = process_floor(map)
    floor = get_floor_area(map)
    explored_area = get_explored_area(map)
    contours, _ = cv2.findContours(explored_area.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image = np.zeros(map.shape[-2:], dtype=np.uint8)
    image = cv2.drawContours(image, contours, -1, (255, 255, 255), thickness=3)
    res = np.logical_and(floor, image)
    res = remove_small_objects(res.astype(bool), min_size=30)
    
    return res.astype(np.uint8)


def get_traversible_area(map: np.ndarray) -> np.ndarray:
    objects = get_objects(map)
    obstacles = get_obstacle(map)
    traversible = 1 - (objects + obstacles)
    traversible_area = np.sum(traversible)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(traversible.astype(np.uint8))
    if nb_components > 2:
        areas = [np.sum(output == i) for i in range(1, nb_components)]
        left_areas = [traversible_area - item for item in areas]
        min_idx = left_areas.index(min(left_areas)) + 1
        res = np.ones(map.shape[-2:])
        for i in range(nb_components):
            if i != min_idx:    
                res[output == i] = 0
        return res
    else:
        return traversible

def get_floor_area(map: np.ndarray) -> np.ndarray:
    traversible = get_traversible_area(map)
    floor = process_floor(map)
    res = np.logical_xor(floor, traversible)
    res = remove_small_objects(res, min_size=100)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(res.astype(np.uint8))
    if nb_components > 2:
        areas = [np.sum(output == i) for i in range(1, nb_components)]
        max_id = areas.index(max(areas)) + 1
        for i in range(1, nb_components):
            if i != max_id:
                floor = np.logical_or(floor, output==i)
                
    return floor.astype(bool)


def get_nearest_nonzero_waypoint(arr: np.ndarray, start: Sequence) -> np.ndarray:
    nonzero_indices = np.argwhere(arr != 0)
    if len(nonzero_indices) > 0:
        distances = cdist([start], nonzero_indices)
        nearest_index = np.argmin(distances)
        
        return np.array(nonzero_indices[nearest_index])
    else:
        return np.array([int(start[0]), int(start[1])])


def angle_between_vectors(vector1: np.ndarray, vector2: np.ndarray) -> np.ndarray:
    dot_product = np.dot(vector1, vector2)
    vector1_length = np.linalg.norm(vector1)
    vector2_length = np.linalg.norm(vector2)
    angle = np.arccos(dot_product / (vector1_length * vector2_length))
    
    return np.degrees(angle)


def angle_to_vector(angle: float) -> np.ndarray:
    angle_rad = np.radians(angle)
    x = np.cos(angle_rad)
    y = np.sin(angle_rad)
    
    return np.array([x, y])


def process_destination(destination: np.ndarray, full_map: np.ndarray) -> np.ndarray:
    floor = process_floor(full_map)
    traversible = get_traversible_area(full_map)
    destination = remove_small_objects(destination.astype(bool), min_size=30).astype(np.uint8)
    destination = dilation(destination, selem=disk(5))
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(destination)
    if len(centroids) > 1:
        centroid = centroids[1] # the first one is background
        waypoint = np.array([int(centroid[1]), int(centroid[0])])
        waypoint = get_nearest_nonzero_waypoint(np.logical_and(floor, traversible), waypoint)
        return waypoint
    else:
        return None


def angle_and_direction(a: np.ndarray, b: np.ndarray, turn_angle: float) -> Tuple:
    unit_a = a / (np.linalg.norm(a) + 1e-5)
    unit_b = b / (np.linalg.norm(b) + 1e-5)
    
    cross_product = np.cross(unit_a, unit_b)
    dot_product = np.dot(unit_a, unit_b)
    
    angle = np.arccos(dot_product)
    angle_degrees = np.degrees(angle)
    
    if cross_product > 0 and angle_degrees > (turn_angle / 2):
        direction = 3 # right
        print("turn left", angle_degrees)
    elif cross_product < 0 and angle_degrees > (turn_angle / 2):
        direction = 2 # left
        print("turn right", angle_degrees)
    elif cross_product == 0 and angle_degrees == 180:
        direction = 3
        print("turn left", angle_degrees)
    else:
        direction = 1 # forward
        print("go forward", angle_degrees)
    
    return angle_degrees, direction


def closest_point_within_threshold(points_array: np.ndarray, target_point: np.ndarray, threshold: float) -> int:
    """Find the point within the threshold distance that is closest to the target_point.

    Args:
        points_array (np.ndarray): An array of 2D points, where each point is a tuple
            (x, y).
        target_point (np.ndarray): The target 2D point (x, y).
        threshold (float): The maximum distance threshold.

    Returns:
        int: The index of the closest point within the threshold distance.
    """
    distances = np.sqrt((points_array[:, 0] - target_point[0]) ** 2 + (points_array[:, 1] - target_point[1]) ** 2)
    within_threshold = distances <= threshold

    if np.any(within_threshold):
        closest_index = np.argmin(distances)
        return int(closest_index)

    return -1
    
    
def collision_check(last_pose: np.ndarray, current_pose: np.ndarray,
                    resolution: float, map_shape: Sequence,
                    collision_threshold: float=0.20,
                    width: float=0.5, height: float=0.5, buf: float=0.0) -> np.ndarray:
    last_position, last_heading = get_agent_position(last_pose, resolution)
    x0, y0 = last_position
    current_position, _ = get_agent_position(current_pose, resolution)
    position_vector = current_position - last_position
    displacement = np.linalg.norm(position_vector)
    collision_map = np.zeros(map_shape)
    
    if displacement < collision_threshold * 100 / resolution:
        print("!!!!!!!!! COLLISION !!!!!!!!!")
        theta = np.deg2rad(last_heading)
        width_range = int(width * 100 / resolution)
        height_range = int(height * 100 / resolution)
        
        for i in range(height_range):
            for j in range(width_range):
                l1 = j + buf
                l2 = i - width_range // 2
                dy = l1 * np.cos(theta) + l2 * np.sin(theta) # change to ndarray coordinate
                dx = l1 * np.sin(theta) - l1 * np.cos(theta) # change to ndarray coordinate
                x1 = int(x0 - dx)
                y1 = int(y0 + dy)
                x1, y1 = threshold_poses([x1, y1], collision_map.shape)
                collision_map[x1, y1] = 1
    
    return collision_map
        
    
# def get_traversible_area(map: np.ndarray) -> np.ndarray:
#     objects = get_objects(map)
#     obstacles = get_obstacle(map)
#     traversible = 1 - (objects + obstacles)

#     return traversible


# def get_obstacle(map: np.ndarray, kernel_size: int=3) -> np.ndarray:
#     floor = map[4, ...] # channel is floor area
#     floor_mask = 1 - (floor != 0)
#     obstacle = map[0, ...] * floor_mask
#     obstacle = remove_small_objects(
#         obstacle.astype(bool), 
#         min_size=50, # you can try different minimum object size
#         connectivity=5)
#     selem = disk(kernel_size)
#     obstacle = closing(obstacle, selem)
    
#     return obstacle.astype(bool)


# def get_floor(map: np.ndarray, kernel_size: int=3) -> np.ndarray:
#     floor = map[4, ...]
#     selem = disk(kernel_size)
#     floor = closing(floor, selem)
#     floor = remove_small_objects(floor.astype(bool), min_size=200)
    
#     return floor.astype(bool)


# def process_floor(map: np.ndarray, kernel_size: int=3) -> np.ndarray:
#     floor = get_floor(map, kernel_size)
#     obstacles = get_obstacle(map, kernel_size)
#     objects = get_objects(map, kernel_size)
    
#     obstacles_mask = (obstacles == False)
#     objects_mask = (objects == False)
#     floor = floor * objects_mask * obstacles_mask
#     floor = remove_small_objects(floor.astype(bool), min_size=200)
    
#     return floor