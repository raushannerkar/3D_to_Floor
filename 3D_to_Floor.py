import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from sklearn.cluster import DBSCAN

def load_point_cloud(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    return pcd

def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
    pcd_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    return pcd_down

def project_to_2d(pcd):
    points = np.asarray(pcd.points)
    points_2d = points[:, :2]
    return points_2d

def extract_walls(points_2d, eps=0.5, min_samples=10):
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points_2d)
    labels = clustering.labels_
    walls = []
    for label in set(labels):
        if label == -1:
            continue
        cluster_points = points_2d[labels == label]
        polygon = Polygon(cluster_points).convex_hull
        walls.append(polygon)
    return walls

def visualize_floor_plan(walls):
    fig, ax = plt.subplots()
    for wall in walls:
        x, y = wall.exterior.xy
        ax.plot(x, y, color='black')
    plt.axis('equal')
    plt.show()

def save_floor_plan(walls, output_file):
    fig, ax = plt.subplots()
    for wall in walls:
        x, y = wall.exterior.xy
        ax.plot(x, y, color='black')
    plt.axis('equal')
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()

# Example usage
file_path = 'path/to/your/pointcloud.ply'
voxel_size = 0.05
pcd = load_point_cloud(file_path)
pcd_down = preprocess_point_cloud(pcd, voxel_size)
points_2d = project_to_2d(pcd_down)
walls = extract_walls(points_2d)
visualize_floor_plan(walls)
save_floor_plan(walls, 'floor_plan.png')
