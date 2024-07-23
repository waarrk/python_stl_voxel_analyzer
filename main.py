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


def create_supports(mesh, angles):
    support_prisms = []
    z_min = mesh.bounds[0][2]  # XY平面のZ座標
    for i, angle in enumerate(angles):
        if 120 <= angle <= 180:
            face = mesh.faces[i]
            vertices = mesh.vertices[face]

            # サポート材の頂点を作成
            support_vertices = np.vstack(
                (vertices, np.hstack((vertices[:, :2], np.full((3, 1), z_min)))))

            # サポート材の面を作成
            support_faces = [
                [0, 1, 2], [3, 4, 5],  # 上面と下面
                [0, 1, 4], [0, 4, 3],  # 側面
                [1, 2, 5], [1, 5, 4],
                [2, 0, 3], [2, 3, 5]
            ]
            support_prisms.append((support_vertices, support_faces))
    return support_prisms


def visualize(mesh, angles, support_prisms):
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

    # サポート材の可視化
    for vertices, faces in support_prisms:
        support_collection = Poly3DCollection(
            [vertices[face] for face in faces], facecolors='lightblue', alpha=0.5)
        ax.add_collection3d(support_collection)

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
    # plt.show()


def main(model_stl_file_path):
    model_mesh = load_stl(model_stl_file_path)

    # 前処理：モデルをXY平面に一致させる
    preprocess_mesh(model_mesh)

    # モデルの体積を計算
    model_volume = model_mesh.volume

    # オーバーハング角度を計算
    angles = calculate_support_volume(model_mesh)

    # サポート材を作成
    support_prisms = create_supports(model_mesh, angles)

    # サポート材の体積を計算
    support_volumes = [trimesh.Trimesh(vertices=vertices, faces=faces).volume
                       for vertices, faces in support_prisms]

    print(f'モデルメッシュの頂点数: {len(model_mesh.vertices)}')
    print(f'モデルの体積: {model_volume} 立方単位')
    print(f'サポート材の数: {len(support_prisms)}')
    print(f'サポート材の体積の合計: {np.sum(support_volumes)} 立方単位')

    # 可視化
    visualize(model_mesh, angles, support_prisms)


# 使用例
model_stl_file_path = '3DBenchy.stl'  # モデルのSTLファイルパス

main(model_stl_file_path)
