import cv2
import torch
import numpy as np
import torch.nn.functional as F
from collections import Sequence
from skimage.morphology import remove_small_objects, closing, disk


def get_grid(pose, grid_size, device):
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
    floor = process_floor(map)
    explored_area = get_explored_area(map)
    contours, _ = cv2.findContours(explored_area.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image = np.zeros(map.shape[-2:], dtype=np.uint8)
    image = cv2.drawContours(image, contours, -1, (255, 255, 255), thickness=2)
    res = np.logical_and(floor, image)
    res = remove_small_objects(res.astype(bool), min_size=30)
    
    return res.astype(np.uint8)


def angle_between_vectors(vector1: np.ndarray, vector2: np.ndarray) -> np.ndarray:
    dot_product = np.dot(vector1, vector2)
    vector1_length = np.linalg.norm(vector1)
    vector2_length = np.linalg.norm(vector2)
    angle = np.arccos(dot_product / (vector1_length * vector2_length))
    
    return np.degrees(angle)


def angle_to_vector(angle: float):
    angle_rad = np.radians(angle)
    x = np.cos(angle_rad)
    y = np.sin(angle_rad)
    
    return np.array([x, y])


def change_to_Cartesian_coordinates(point: Sequence, x_shape):
    x, y = point
    
    return np.array([y, x_shape - x])


def array_coord_to_cartesian_coord(point: np.array, x_shape: int):
    x, y = point
    flipud_x = x_shape - x
    flipud_y = y
    
    return np.array([flipud_y, flipud_x])


def angle_and_direction(a: np.ndarray, b: np.ndarray, turn_angle: float):
    unit_a = a / np.linalg.norm(a)
    unit_b = b / np.linalg.norm(b)
    
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