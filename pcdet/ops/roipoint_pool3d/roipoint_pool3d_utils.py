import torch
import torch.nn as nn
from torch.autograd import Function

from ...utils import box_utils
from . import roipoint_pool3d_cuda


class RoIPointPool3d(nn.Module):
    def __init__(self, num_sampled_points=512, pool_extra_width=1.0):
        super().__init__()
        self.num_sampled_points = num_sampled_points
        self.pool_extra_width = pool_extra_width

    def forward(self, points, point_features, boxes3d):
        """
        Args:
            points: (B, N, 3)
            point_features: (B, N, C)
            boxes3d: (B, M, 7), [x, y, z, dx, dy, dz, heading]

        Returns:
            pooled_features: (B, M, 512, 3 + C)
            pooled_empty_flag: (B, M)
        """
        return RoIPointPool3dFunction.apply(
            points, point_features, boxes3d, self.pool_extra_width, self.num_sampled_points
        )


class RoIPointPool3dFunction(Function):
    @staticmethod
    def forward(ctx, points, point_features, boxes3d, pool_extra_width, num_sampled_points=512):
        """
        Args:
            ctx:
            points: (B, N, 3)
            point_features: (B, N, C)
            boxes3d: (B, num_boxes, 7), [x, y, z, dx, dy, dz, heading]
            pool_extra_width:
            num_sampled_points:

        Returns:
            pooled_features: (B, num_boxes, 512, 3 + C)
            pooled_empty_flag: (B, num_boxes)
            pts_idx: (B, num_boxes, 512)
        """
        assert points.shape.__len__() == 3 and points.shape[2] == 3
        batch_size, boxes_num, feature_len = points.shape[0], boxes3d.shape[1], point_features.shape[2]
        pooled_boxes3d = box_utils.enlarge_box3d(boxes3d.view(-1, 7), pool_extra_width).view(batch_size, -1, 7)

        pooled_features = point_features.new_zeros((batch_size, boxes_num, num_sampled_points, 3 + feature_len))
        pooled_empty_flag = point_features.new_zeros((batch_size, boxes_num)).int()
        pts_idx = point_features.new_zeros((batch_size, boxes_num, num_sampled_points)).int()

        roipoint_pool3d_cuda.forward(
            points.contiguous(), pooled_boxes3d.contiguous(),
            point_features.contiguous(), pooled_features, pts_idx, pooled_empty_flag
        )

        ctx.save_for_backward(
                point_features, pooled_empty_flag, pts_idx)
        ctx.mark_non_differentiable(
                pooled_empty_flag, pts_idx)
        
            
        return pooled_features, pooled_empty_flag, pts_idx

    @staticmethod
    def backward(ctx, pooled_features_grad, pooled_empty_flag, pts_idx):
        point_features, pooled_empty_flag, pts_idx = ctx.saved_tensors
        batch_size = point_features.size(0)
        npoint = point_features.size(1)
        feature_len = point_features.size(2)

        # filter empty boxes
        # pooled_features_grad = pooled_features_grad[~pooled_empty_flag]
        # pts_idx = pts_idx[~pooled_empty_flag]
        
        batch_point_features_grad = torch.zeros_like(point_features)
        xyz_features_grad = point_features.new_zeros((batch_size, npoint, 3))
        
        for batch_mask in range(0, batch_size):
            # expand points index to max(3, feature_len)
            pts_idx_expanded = pts_idx[batch_mask].view(-1).long().unsqueeze(1).expand(-1, max(3, feature_len))
            xyz_pooled_features_grad_viewed = pooled_features_grad[batch_mask, :, :, :3].reshape(-1, 3)
            pooled_features_grad_viewed = pooled_features_grad[batch_mask, :, :, 3:].reshape(-1, feature_len)

            # scatter points gradient
            batch_point_features_grad[batch_mask].scatter_add_(0, pts_idx_expanded[:, :feature_len], pooled_features_grad_viewed)
            xyz_features_grad[batch_mask].scatter_add_(0, pts_idx_expanded[:, :3], xyz_pooled_features_grad_viewed)
        
        return xyz_features_grad, batch_point_features_grad, None, None, None


if __name__ == '__main__':
    pass
