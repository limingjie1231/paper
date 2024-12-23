import torch
import torch_scatter
import torch.nn as nn
import numpy as np
import spconv.pytorch as spconv


class voxelization(nn.Module):
    def __init__(self, coors_range_xyz, spatial_shape, scale_list):
        super(voxelization, self).__init__()
        self.spatial_shape = spatial_shape
        self.scale_list = scale_list + [1]
        self.coors_range_xyz = coors_range_xyz

    @staticmethod
    def sparse_quantize(pc, coors_range, spatial_shape):
        idx = spatial_shape * (pc - coors_range[0]) / (coors_range[1] - coors_range[0])
        rounded_data = torch.round(idx)
        floor_data = torch.floor(idx)
        result = rounded_data - floor_data
        floor_data[result == 0] += 1
        return torch.cat((rounded_data.unsqueeze(0), floor_data.unsqueeze(0)), dim=0)

    def forward(self, data_dict):
        pc = data_dict['points'][:, :3]

        for idx, scale in enumerate(self.scale_list):
            xidx = self.sparse_quantize(pc[:, 0], self.coors_range_xyz[0], np.ceil(self.spatial_shape[0] / scale))
            yidx = self.sparse_quantize(pc[:, 1], self.coors_range_xyz[1], np.ceil(self.spatial_shape[1] / scale))
            zidx = self.sparse_quantize(pc[:, 2], self.coors_range_xyz[2], np.ceil(self.spatial_shape[2] / scale))

            bxyz_indx = torch.stack([data_dict['batch_idx'], xidx[0], yidx[0], zidx[0]], dim=-1).long()
            unq, unq_inv, unq_cnt = torch.unique(bxyz_indx, return_inverse=True, return_counts=True, dim=0)
            unq = torch.cat([unq[:, 0:1], unq[:, [3, 2, 1]]], dim=1)
            data_dict['scale_{}'.format(scale)] = {
                'full_coors': bxyz_indx,
                'xidx': xidx,
                'yidx': yidx,
                'zidx': zidx,
                'coors_inv': unq_inv,
                'coors': unq.type(torch.int32)
            }
        return data_dict


class voxel_3d_generator(nn.Module):
    def __init__(self, in_channels, out_channels, coors_range_xyz, spatial_shape):
        super(voxel_3d_generator, self).__init__()
        self.spatial_shape = spatial_shape
        self.coors_range_xyz = coors_range_xyz
        self.PPmodel = nn.Sequential(
            nn.Linear(in_channels + 6 +3*3, out_channels),
            nn.ReLU(True),
            nn.Linear(out_channels, out_channels)
        )

    def prepare_input(self, point, grid_ind,scale_1,batch_idx ,inv_idx):
        bxyz_indx_1 = torch.stack([batch_idx, scale_1['xidx'][1], scale_1['yidx'][0], scale_1['zidx'][0]], dim=-1).long()
        bxyz_indx_2 = torch.stack([batch_idx, scale_1['xidx'][0], scale_1['yidx'][1], scale_1['zidx'][0]], dim=-1).long()
        bxyz_indx_3 = torch.stack([batch_idx, scale_1['xidx'][0], scale_1['yidx'][0], scale_1['zidx'][1]], dim=-1).long()
        _, unq_inv_1, _ = torch.unique(bxyz_indx_1, return_inverse=True, return_counts=True, dim=0)
        _, unq_inv_2, _ = torch.unique(bxyz_indx_2, return_inverse=True, return_counts=True, dim=0)
        _, unq_inv_3, _ = torch.unique(bxyz_indx_3, return_inverse=True, return_counts=True, dim=0)

        pc_mean = torch_scatter.scatter_mean(point[:, :3], inv_idx, dim=0)[inv_idx]
        nor_pc = point[:, :3] - pc_mean
        pc_mean_1 = torch_scatter.scatter_mean(point[:, :3], unq_inv_1, dim=0)[unq_inv_1]
        nor_pc_1 = point[:, :3] - pc_mean_1
        pc_mean_2 = torch_scatter.scatter_mean(point[:, :3], unq_inv_2, dim=0)[unq_inv_2]
        nor_pc_2 = point[:, :3] - pc_mean_2
        pc_mean_3 = torch_scatter.scatter_mean(point[:, :3], unq_inv_3, dim=0)[unq_inv_3]
        nor_pc_3 = point[:, :3] - pc_mean_3

        coors_range_xyz = torch.Tensor(self.coors_range_xyz)
        cur_grid_size = torch.Tensor(self.spatial_shape)
        crop_range = coors_range_xyz[:, 1] - coors_range_xyz[:, 0]
        intervals = (crop_range / cur_grid_size).to(point.device)

        voxel_centers = grid_ind * intervals + coors_range_xyz[:, 0].to(point.device)
        center_to_point = point[:, :3] - voxel_centers

        voxel_centers_1 = bxyz_indx_1[:, 1:] * intervals + coors_range_xyz[:, 0].to(point.device)
        center_to_point_1 = point[:, :3] - voxel_centers_1
        norm1 = torch.norm(center_to_point_1, dim=1)
        voxel_centers_2 = bxyz_indx_2[:, 1:] * intervals + coors_range_xyz[:, 0].to(point.device)
        center_to_point_2 = point[:, :3] - voxel_centers_2
        norm2 = torch.norm(center_to_point_2, dim=1)
        voxel_centers_3 = bxyz_indx_3[:, 1:] * intervals + coors_range_xyz[:, 0].to(point.device)
        center_to_point_3 = point[:, :3] - voxel_centers_3
        norm3 = torch.norm(center_to_point_3, dim=1)

        pc_feature = torch.cat((point, nor_pc, center_to_point,nor_pc_1/norm1.view(-1,1),nor_pc_2/norm2.view(-1,1),nor_pc_3/ norm3.view(-1,1)), dim=1)
        return pc_feature

    def forward(self, data_dict):
        pt_fea = self.prepare_input(
            data_dict['points'],
            data_dict['scale_1']['full_coors'][:, 1:],
            data_dict['scale_1'],
            data_dict['batch_idx'],
            data_dict['scale_1']['coors_inv']
        )
        pt_fea = self.PPmodel(pt_fea)

        features = torch_scatter.scatter_mean(pt_fea, data_dict['scale_1']['coors_inv'], dim=0)
        data_dict['sparse_tensor'] = spconv.SparseConvTensor(
            features=features,
            indices=data_dict['scale_1']['coors'].int(),
            spatial_shape=np.int32(self.spatial_shape)[::-1].tolist(),
            batch_size=data_dict['batch_size']
        )

        data_dict['coors'] = data_dict['scale_1']['coors']
        data_dict['coors_inv'] = data_dict['scale_1']['coors_inv']
        data_dict['full_coors'] = data_dict['scale_1']['full_coors']

        return data_dict