import torch
import random
import torchvision
import torchvision.transforms.v2.functional as TF

class Identity(object):
    def __init__(self):
        pass

    def __call__(self, batch):     
        return batch

class Transform(object):
    def __init__(self, p = 0.0, size = 300):
        self.p = p
        self.crop = torchvision.transforms.v2.RandomCrop(size=[size, size])

    def __call__(self, batch):
        keys = list(batch.keys())
        keys.remove('label')
        keys.remove('name')

        if 'aerial' in keys:
            batch['aerial'] = self.crop(batch['aerial'])

        if random.random() < self.p:
            for key in keys:
                batch[key] = TF.horizontal_flip(batch[key])
        
        if random.random() < self.p:
            for key in keys:
                batch[key] = TF.vertical_flip(batch[key])

        if random.random() < self.p:
            for key in keys:
                batch[key] = TF.rotate(batch[key], 90)
        
        return batch
    
class TransformMAE(object):
    def __init__(self, p = 0.5, size = 224, s2_size = 0):
        self.p = p
        self.s2 = False
        self.size = size
        self.s2_size = s2_size
        if s2_size > 0:
            self.s2 = True

    def _safe_resize(self, tensor, target_size):
        """安全地resize张量，处理不同的输入格式"""
        if tensor is None:
            return None
            
        # 检查张量的维度和形状
        if tensor.dim() == 3:  # (C, H, W)
            return TF.resize(tensor, [target_size, target_size], antialias=True)
        # elif tensor.dim() == 4:  # (T, C, H, W) 时序数据
        #     # 对每个时间步分别进行resize
        #     resized_frames = []
        #     for t in range(tensor.shape[0]):
        #         frame = tensor[t]  # (C, H, W)
        #         resized_frame = TF.resize(frame, [target_size, target_size], antialias=True)
        #         resized_frames.append(resized_frame)
        #     return torch.stack(resized_frames, dim=0)  # (T, C, H, W)
        # elif tensor.dim() == 5:  # (B, T, C, H, W) 批次时序数据
        #     # 对每个batch和每个时间步分别进行resize
        #     batch_size, time_steps = tensor.shape[0], tensor.shape[1]
        #     resized_data = []
        #     for b in range(batch_size):
        #         batch_frames = []
        #         for t in range(time_steps):
        #             frame = tensor[b, t]  # (C, H, W)
        #             resized_frame = TF.resize(frame, [target_size, target_size], antialias=True)
        #             batch_frames.append(resized_frame)
        #         resized_data.append(torch.stack(batch_frames, dim=0))  # (T, C, H, W)
        #     return torch.stack(resized_data, dim=0)  # (B, T, C, H, W)
        else:
            # 不支持的张量维度，打印调试信息
            # print(f"警告: 不支持的张量维度 {tensor.dim()}, 形状: {tensor.shape}")
            return tensor

    def __call__(self, batch):
        keys = list(batch.keys())
        if 'label' in keys:
            keys.remove('label')
        if 'name' in keys:
            keys.remove('name')

        for key in keys:
            try:
                if self.s2 and key in ['s2-4season-median', 's2-median']:
                        target_size = self.s2_size
                else:
                    target_size = self.size
                
                # 安全地处理resize
                batch[key] = self._safe_resize(batch[key], target_size)
                
            except Exception as e:
                print(f"变换失败 - 模态: {key}, 错误: {e}")
                if batch[key] is not None:
                    print(f"  张量形状: {batch[key].shape}")
                    print(f"  张量类型: {type(batch[key])}")
                    print(f"  张量数据类型: {batch[key].dtype if torch.is_tensor(batch[key]) else 'N/A'}")
                # 如果变换失败，保持原始数据
                continue

        return batch
    