import torch.utils.data as data
import numpy as np
from PIL import Image
import os
import torch
import torchvision.transforms as transforms


def get_data_loader(root_dir, mode):
    if mode == 'train':
        dl = data.DataLoader(TrajDataset(os.path.join(root_dir, mode)), shuffle=True)
    else:
        dl = data.DataLoader(TrajDataset(os.path.join(root_dir, mode)), shuffle=False)
    return dl


class TrajDataset(data.Dataset):
    def __init__(self, data_path):
        super(TrajDataset, self).__init__()
        self.img_paths = [os.path.join(data_path, filename) for filename in os.listdir(data_path)
                          if filename.endswith('.png')]

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = Image.open(img_path).convert('L')
        speed = np.load(img_path.replace('.png', '_speed.npy'))
        direction = np.load(img_path.replace('.png', '_direction.npy'))
        transition_view = np.load(img_path.replace('.png', '_transition.npy')).astype('float32')

        transition_view[transition_view > 0] = 1.0
        transition_view = torch.from_numpy(transition_view)

        w, h = img.size
        unit_w = int(w / 4)
        point_img = img.crop((0, 0, unit_w, h))
        line_img = img.crop((unit_w, 0, unit_w * 2, h))
        centerline_img = img.crop((unit_w * 2, 0, unit_w * 3, h))
        region_img = img.crop((unit_w * 3, 0, unit_w * 4, h))

        # normalization & to torch data structure
        spatial_features = []
        img_transform = transforms.ToTensor()
        spatial_features.append(img_transform(point_img))
        line_img = img_transform(line_img)
        spatial_features.append(line_img)
        speed = torch.from_numpy(((speed / 34 - 0.5) * 2.0).astype('float32'))
        spatial_features.append(speed.permute(2, 0, 1))
        direction = direction.astype('float32')
        summed = np.sum(direction, axis=2, keepdims=True)
        direction = torch.from_numpy(np.divide(direction, summed, out=np.zeros_like(direction), where=summed != 0))
        for i in range(direction.shape[2]):
            spatial_features.append(direction[:, :, i:i+1].permute(2, 0, 1))
        spatial_view = torch.cat(tuple(spatial_features), 0)
        centerline_img = img_transform(centerline_img)
        region_img = img_transform(region_img)

        return {
            'img_path': img_path,
            'spatial_view': spatial_view, 'transition_view': transition_view,
            'centerline': centerline_img, 'region': region_img
        }

    def __len__(self):
        return len(self.img_paths)
