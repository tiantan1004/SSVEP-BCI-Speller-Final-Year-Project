import numpy as np

data = np.load('flash_eeg_segments.npz')

# 查看所有键
print(data.files)

# 查看各数组形状
for key in data.files:
    print(f"{key}: shape={data[key].shape}, dtype={data[key].dtype}")

# 取第 i 段的有效数据（去掉零填充）
for i in range(6):
    segment = data['flash_eeg'][i, :, :data['flash_lengths'][i]]  # (9, 实际长度)
    print(f"\n第{i}段: shape={segment.shape}, 目标='{data['flash_targets'][i]}'")