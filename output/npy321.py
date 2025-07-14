import numpy as np
from pykalman import KalmanFilter

def kalman_smooth(data):
    n_frames, n_joints, n_dim = data.shape
    smoothed = np.zeros_like(data)
    for j in range(n_joints):
        kf = KalmanFilter(initial_state_mean=data[0, j], n_dim_obs=n_dim)
        smoothed[:, j, :] = kf.smooth(data[:, j, :])[0]
    return smoothed
# 假设你的文件名分别为
file_paths = [
    "output/results_Layup_Europe_V68_mirror0_2.npy",
    "output/results_Layup_Europe_V68_mirror0_1.npy",
    "output/results_Layup_Europe_V68_mirror0_0.npy"
]

# 读取每个视角的npy文件
arrays = [kalman_smooth(np.load(path)) for path in file_paths]

# 检查shape是否一致
shapes = [arr.shape for arr in arrays]
print("各视角shape：", shapes)
assert all(shape == shapes[0] for shape in shapes), "三个视角的shape不一致！"

# 拼接成[view, frame, joints, 2]
multi_view_array = np.stack(arrays, axis=0)
print("拼接后shape：", multi_view_array.shape)  # [3, frame, joints, 2]
multi_view_array[:, :, :, 0] = multi_view_array[:, :, :, 0] / 1920
multi_view_array[:, :, :, 1] = multi_view_array[:, :, :, 1] / 1080
# 将最后两维(joints, 2)合并为一个维度，变成 [3, frame, joints*2]
multi_view_array = multi_view_array.reshape(multi_view_array.shape[0], multi_view_array.shape[1], -1)
print("reshape后shape：", multi_view_array.shape)  # [3, frame, joints*2]
np.save("output/results_Layup_Europe_V68_mirror0_ka.npy", multi_view_array)
print("已保存为 output/results_Layup_Europe_V68_mirror0_ka.npy")
