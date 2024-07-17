import numpy as np
import trimesh


def load_stl(file_path):
    # STLファイルを読み込む
    return trimesh.load(file_path)


def slice_mesh(mesh, slice_height):
    # メッシュをスライスする
    z_levels = np.arange(mesh.bounds[0][2], mesh.bounds[1][2], slice_height)
    slices = []

    for z in z_levels:
        plane_origin = [0, 0, z]
        plane_normal = [0, 0, 1]
        slice = mesh.section(plane_origin=plane_origin,
                             plane_normal=plane_normal)
        if slice is not None:
            slices.append(slice)

    return slices, z_levels


def calculate_slice_area(slice):
    # スライスの面積を計算する
    if slice is None:
        return 0
    slice_2D, _ = slice.to_planar()
    return slice_2D.area


def calculate_volume_from_slices(slices, slice_height):
    # スライスから体積を計算する
    total_volume = 0.0

    for i in range(len(slices) - 1):
        area1 = calculate_slice_area(slices[i])
        area2 = calculate_slice_area(slices[i + 1])
        average_area = (area1 + area2) / 2.0
        total_volume += average_area * slice_height

    return total_volume


def main(stl_file_path, slice_height):
    mesh = load_stl(stl_file_path)
    print(f'メッシュの頂点数: {len(mesh.vertices)}')

    slices, z_levels = slice_mesh(mesh, slice_height)
    print(f'スライス数: {len(slices)}')

    volume = calculate_volume_from_slices(slices, slice_height)
    print(f'推定体積: {volume} 立方単位')


# 使用例
stl_file_path = '3DBenchy.stl'
slice_height = 0.1  # スライス間隔を調整
main(stl_file_path, slice_height)
