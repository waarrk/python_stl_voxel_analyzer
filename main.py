import numpy as np
import trimesh
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def load_stl(file_path):
    # STLファイルを読み込む
    return trimesh.load(file_path)


def calculate_face_normals(mesh):
    try:
        face_normals = mesh.face_normals
    except Exception as e:
        print(f'Error in calculating face normals: {e}')
        face_normals = []
        for face in mesh.faces:
            vertices = mesh.vertices[face]
            normal = np.cross(vertices[1] - vertices[0],
                              vertices[2] - vertices[0])
            normal /= np.linalg.norm(normal)
            face_normals.append(normal)
        face_normals = np.array(face_normals)
    return face_normals


def has_space_below(vertices, mesh, kdtree, threshold=0.1):
    # 全ての頂点に対して下に空間があるかどうかをチェック
    space_below = []
    for vertex in vertices:
        _, idx = kdtree.query([vertex[0], vertex[1]])
        min_z = mesh.vertices[idx, 2]
        space_below.append((vertex[2] - min_z) > threshold)
    return np.array(space_below)


def calculate_support_volume(mesh, overhang_threshold_angle):
    # 表面法線を計算
    face_normals = calculate_face_normals(mesh)
    z_axis = np.array([0, 0, 1])

    # オーバーハング角度を計算
    dot_product = np.dot(face_normals, z_axis)
    dot_product = np.clip(dot_product, -1.0, 1.0)  # Avoid invalid values
    angles = np.degrees(np.arccos(dot_product))

    print(f'オーバーハング角度の最大値: {np.max(angles)}')
    print(f'angles 配列の長さ: {len(angles)}')
    print(f'mesh.faces 配列の長さ: {len(mesh.faces)}')

    # オーバーハング角度がしきい値を超える部分を特定
    try:
        overhang_faces = mesh.faces[angles > overhang_threshold_angle]
        print(f'オーバーハング面の数: {len(overhang_faces)}')
    except Exception as e:
        print(f'Error in selecting overhang faces: {e}')
        return 0.0, None, None

    # オーバーハング面の頂点を取得
    overhang_vertices = mesh.vertices[overhang_faces].reshape(-1, 3)
    print(f'オーバーハング頂点の数: {len(overhang_vertices)}')

    # KDTreeを構築して空間検索
    kdtree = cKDTree(mesh.vertices[:, :2])

    print(f'KDTreeによる最近傍点の検索: {kdtree.query([0, 0])}')

    # 下に空間がある頂点をフィルタリング
    space_below = has_space_below(overhang_vertices, mesh, kdtree)
    valid_vertices = overhang_vertices[space_below]

    print(f'下に空間があるオーバーハング頂点の数: {len(valid_vertices)}')

    overhang_vertices_ratio = len(
        valid_vertices) / len(overhang_vertices) * 100
    print(f'オーバーハング頂点の割合: {overhang_vertices_ratio:.2f}%')

    # サポート材の形状を計算
    support_meshes = []
    support_cylinders = []

    for vertex in valid_vertices:
        height = vertex[2]
        if height > 0 and np.isfinite(height):
            support_cylinder = trimesh.creation.cylinder(
                radius=0.5,
                height=height,
                sections=4,
                transform=trimesh.transformations.translation_matrix(
                    [vertex[0], vertex[1], height / 2])
            )
            support_meshes.append(support_cylinder)
            support_cylinders.append(support_cylinder)

    if support_meshes:
        support_mesh = trimesh.util.concatenate(support_meshes)
        support_volume = support_mesh.volume
    else:
        support_volume = 0.0

    return support_volume, overhang_faces, support_cylinders


def visualize(mesh, overhang_faces, support_cylinders):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # メッシュの可視化
    mesh_collection = Poly3DCollection(mesh.vertices[mesh.faces])
    mesh_collection.set_edgecolor('k')
    mesh_collection.set_facecolor('red')
    ax.add_collection3d(mesh_collection)

    # オーバーハング部分の可視化
    if overhang_faces is not None:
        overhang_collection = Poly3DCollection(mesh.vertices[overhang_faces])
        overhang_collection.set_facecolor('orange')
        ax.add_collection3d(overhang_collection)

    # サポート材の可視化
    for support in support_cylinders:
        support_collection = Poly3DCollection(support.vertices[support.faces])
        support_collection.set_facecolor('blue')
        support_collection.set_alpha(0.5)
        ax.add_collection3d(support_collection)

    min_bounds, max_bounds = mesh.bounds
    ax.auto_scale_xyz(min_bounds, max_bounds, max_bounds)

    # カメラ位置
    ax.view_init(elev=10, azim=110)

    # pngファイルとして保存
    plt.savefig('3DBenchy_supports.png', bbox_inches='tight')

    # plt.show()


def main(model_stl_file_path, overhang_threshold_angle):
    model_mesh = load_stl(model_stl_file_path)

    # モデルの体積を計算
    model_volume = model_mesh.volume

    # サポート材の体積を計算
    support_volume, overhang_faces, support_cylinders = calculate_support_volume(
        model_mesh, overhang_threshold_angle)

    print(f'モデルメッシュの頂点数: {len(model_mesh.vertices)}')
    print(f'モデルの体積: {model_volume} 立方単位')
    print(f'サポート材の推定体積: {support_volume} 立方単位')

    # 可視化
    visualize(model_mesh, overhang_faces, support_cylinders)


# 使用例
model_stl_file_path = '3DBenchy.stl'  # モデルのSTLファイルパス
overhang_threshold_angle = 45  # オーバーハングのしきい値角度

main(model_stl_file_path, overhang_threshold_angle)
