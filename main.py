import numpy as np
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def load_stl(file_path):
    # STLファイルを読み込む
    return trimesh.load(file_path)


def calculate_face_normals(mesh):
    try:
        face_normals = mesh.face_normals
    except Exception as e:
        print(f'表面法線の計算エラー: {e}')
        face_normals = []
        for face in mesh.faces:
            vertices = mesh.vertices[face]
            normal = np.cross(vertices[1] - vertices[0],
                              vertices[2] - vertices[0])
            normal /= np.linalg.norm(normal)
            face_normals.append(normal)
        face_normals = np.array(face_normals)
    return face_normals


def calculate_support_volume(mesh):
    # 表面法線を計算
    face_normals = calculate_face_normals(mesh)
    z_axis = np.array([0, 0, 1])

    # オーバーハング角度を計算
    dot_product = np.dot(face_normals, z_axis)
    dot_product = np.clip(dot_product, -1.0, 1.0)  # 無効な値を避ける
    angles = np.degrees(np.arccos(dot_product))

    print(f'オーバーハング角度の最大値: {np.max(angles)}')
    print(f'angles 配列の長さ: {len(angles)}')
    print(f'mesh.faces 配列の長さ: {len(mesh.faces)}')

    return angles


def preprocess_mesh(mesh):
    # メッシュの最小Z座標を取得し、モデルをZ軸方向にシフト
    min_z = mesh.bounds[0][2]
    mesh.apply_translation([0, 0, -min_z])


def visualize(mesh, angles):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # オーバーハング角度を0〜180度に正規化
    angles = np.clip(angles, 0, 180)

    # オーバーハング角度に基づいて色を設定
    colors = plt.cm.jet(np.linspace(0, 1, 18))
    face_colors = np.zeros((mesh.faces.shape[0], 4))

    for i in range(18):
        face_mask = (angles >= i * 10) & (angles < (i + 1) * 10)
        face_colors[face_mask] = colors[i]

    # メッシュの可視化
    mesh_collection = Poly3DCollection(
        mesh.vertices[mesh.faces], facecolors=face_colors)
    ax.add_collection3d(mesh_collection)

    min_bounds, max_bounds = mesh.bounds
    ax.auto_scale_xyz(min_bounds, max_bounds, max_bounds)

    # カメラ位置
    ax.view_init(elev=20, azim=120)

    # 色の凡例を追加
    mappable = plt.cm.ScalarMappable(cmap='jet')
    mappable.set_array(np.linspace(0, 180, 18))
    cbar = plt.colorbar(mappable, ax=ax, shrink=0.5, aspect=5)
    cbar.set_label('Overhang Angle (degrees)')
    cbar.set_ticks(np.linspace(0, 180, 10))
    cbar.set_ticklabels([f'{int(i*20)}°' for i in range(10)])

    # PNGファイルとして保存
    plt.savefig('3DBenchy_supports_with_legend.png', bbox_inches='tight')
    plt.show()


def main(model_stl_file_path):
    model_mesh = load_stl(model_stl_file_path)

    # 前処理：モデルをXY平面に一致させる
    preprocess_mesh(model_mesh)

    # モデルの体積を計算
    model_volume = model_mesh.volume

    # オーバーハング角度を計算
    angles = calculate_support_volume(model_mesh)

    print(f'モデルメッシュの頂点数: {len(model_mesh.vertices)}')
    print(f'モデルの体積: {model_volume} 立方単位')

    # 可視化
    visualize(model_mesh, angles)


# 使用例
model_stl_file_path = '3DBenchy.stl'  # モデルのSTLファイルパス

main(model_stl_file_path)
