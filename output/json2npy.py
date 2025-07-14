import json
import numpy as np


def get_perpendicular_vectors_batch(data, target_joint_idx, line_start_joint_idx, line_end_joint_idx):
    """
    批量计算所有帧中目标关节点到指定线段关节的垂直向量
    
    参数:
    data: 形状为 [frame, joint, 2] 的数组
    target_joint_idx: 目标关节点索引
    line_start_joint_idx: 线段起点关节点索引  
    line_end_joint_idx: 线段终点关节点索引
    
    返回:
    perpendicular_vectors: 形状为 [frame, 2] 的垂直向量数组
    """
    # 提取所有帧的目标点坐标
    P = data[:, target_joint_idx, :]  # [frame, 2]
    
    # 提取所有帧的线段起点和终点坐标
    A = data[:, line_start_joint_idx, :]  # [frame, 2]
    B = data[:, line_end_joint_idx, :]    # [frame, 2]
    
    # 计算线段向量 [frame, 2]
    AB = B - A
    
    # 计算从A到P的向量 [frame, 2]
    AP = P - A
    
    # 计算AP在AB上的投影长度 [frame]
    # 使用广播计算点积
    projection_length = np.sum(AP * AB, axis=1) / np.sum(AB * AB, axis=1)
    
    # 投影点坐标 [frame, 2]
    projection_point = A + projection_length[:, np.newaxis] * AB
    
    # 垂直向量（从投影点到P）[frame, 2]
    perpendicular_vectors = P - projection_point

    # 计算投影向量（从A到投影点）[frame, 2]
    projection_vectors = projection_point - A
    
    return perpendicular_vectors, projection_vectors

if __name__ == "__main__":
    # 读取json文件
    with open('output/results_Layup_Europe_V68_mirror0_0.json', 'r') as f:
        data = json.load(f)

    # 假设结构为：{"meta_info": ..., "instance_info": [ {frame_id, instances: [ {keypoints, ...}, ... ]}, ... ]}
    frames = data['instance_info']

    # 用于存储每个人的所有帧数据
    person_dict = {}

    for frame in frames:
        frame_id = frame['frame_id']
        for person_idx, instance in enumerate(frame['instances']):
            # 如果有id字段就用id，否则用person_idx（适合单人场景）
            person_id = instance.get('id', person_idx)
            keypoints = np.array(instance['keypoints'])  # shape: (num_keypoints, 3) 或 (num_keypoints, 2)
            if person_id not in person_dict:
                person_dict[person_id] = []
            person_dict[person_id].append((frame_id, keypoints))

    # 整理为矩阵
    # 每个人的矩阵 shape = (num_frames, num_keypoints, 3)
    person_matrices = {}
    for person_id, frames_data in person_dict.items():
        # 按帧排序
        frames_data.sort(key=lambda x: x[0])
        keypoints_list = [k for _, k in frames_data]
        person_matrices[person_id] = np.stack(keypoints_list, axis=0)  # shape: (num_frames, num_keypoints, 3)

    # 示例：输出每个人的关键点矩阵形状
    for person_id, mat in person_matrices.items():
        print(f'Person {person_id}: matrix shape {mat.shape}')
        # 声明一个shape和mat一样的ndarray，但是第二维的关节点数目是22
        new_shape = (mat.shape[0], 22, mat.shape[2]) if len(mat.shape) == 3 else (mat.shape[0], 22)
        new_array = np.zeros(new_shape, dtype=mat.dtype)
        new_array[:, 0, :] = mat[:, 19, :]
        hip2neck = mat[:, 18, :] - mat[:, 19, :]
        new_array[:, 1, :] = new_array[:, 0, :] + hip2neck * 0.155
        new_array[:, 2, :] = new_array[:, 0, :] + hip2neck * 0.39
        new_array[:, 3, :] = new_array[:, 0, :] + hip2neck * 0.66
        new_array[:, 4, :] = mat[:, 18, :]
        new_array[:, 5, :] = (mat[:, 17, :] + mat[:, 18, :]) / 2
        new_array[:, 7, :] = mat[:, 5, :]
        perpendicular_vectors_l, projection_vectors_l = get_perpendicular_vectors_batch(new_array, 7, 3, 4)
        new_array[:, 6, :] = new_array[:, 3, :] + projection_vectors_l * 0.66 + perpendicular_vectors_l * 0.33
        new_array[:, 8, :] = mat[:, 7, :]
        new_array[:, 9, :] = mat[:, 9, :]
        new_array[:, 11, :] = mat[:, 6, :]
        perpendicular_vectors_r, projection_vectors_r = get_perpendicular_vectors_batch(new_array, 11, 3, 4)
        new_array[:, 10, :] = new_array[:, 3, :] + projection_vectors_r * 0.66 + perpendicular_vectors_r * 0.33
        new_array[:, 12, :] = mat[:, 8, :]
        new_array[:, 13, :] = mat[:, 10, :]
        new_array[:, 14, :] = mat[:, 11, :]
        new_array[:, 15, :] = mat[:, 13, :]
        new_array[:, 16, :] = mat[:, 15, :]
        foot2back_l = mat[:, 24, :] - (mat[:, 20, :] + mat[:, 22, :]) / 2
        new_array[:, 17, :] = (mat[:, 20, :] + mat[:, 22, :]) / 2 + foot2back_l * 0.33
        new_array[:, 18, :] = mat[:, 12, :]
        new_array[:, 19, :] = mat[:, 14, :]
        new_array[:, 20, :] = mat[:, 16, :]
        foot2back_r = mat[:, 25, :] - (mat[:, 21, :] + mat[:, 23, :]) / 2
        new_array[:, 21, :] = (mat[:, 21, :] + mat[:, 23, :]) / 2 + foot2back_r * 0.33
        # 将new_array的最后一维y（即[..., 1]）取相反数，因为pose估计的y轴朝下
        new_array[..., 1] = -new_array[..., 1]
        print(f'新ndarray的shape: {new_array.shape}')
        # 如果需要保存为npy文件
        np.save("output/results_Layup_Europe_V68_mirror0_0.npy", new_array)