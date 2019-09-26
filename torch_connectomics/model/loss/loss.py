from __future__ import print_function, division

import torch
torch.cuda.manual_seed_all(2)
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# 0. main loss functions

class DiceLoss(nn.Module):
    """DICE loss.
    """

    def __init__(self, size_average=True, reduce=True, smooth=100.0):
        super(DiceLoss, self).__init__(size_average, reduce)
        self.smooth = smooth
        self.reduce = reduce

    def dice_loss(self, input, target):
        loss = 0.

        for index in range(input.size()[0]):
            iflat = input[index].view(-1)
            tflat = target[index].view(-1)
            intersection = (iflat * tflat).sum()

            loss += 1 - ((2. * intersection + self.smooth) / 
                    ( (iflat**2).sum() + (tflat**2).sum() + self.smooth))

        # size_average=True for the dice loss
        return loss / float(input.size()[0])

    def dice_loss_batch(self, input, target):

        iflat = input.view(-1)
        tflat = target.view(-1)
        intersection = (iflat * tflat).sum()

        loss = 1 - ((2. * intersection + self.smooth) / 
               ( (iflat**2).sum() + (tflat**2).sum() + self.smooth))
        return loss

    def forward(self, input, target):
        #_assert_no_grad(target)
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

        if self.reduce:
            loss = self.dice_loss(input, target)
        else:    
            loss = self.dice_loss_batch(input, target)
        return loss

class WeightedMSE(nn.Module):
    """Weighted mean-squared error.
    """

    def __init__(self):
        super().__init__()

    def weighted_mse_loss(self, input, target, weight):
        s1 = torch.prod(torch.tensor(input.size()[2:]).float())
        s2 = input.size()[0]
        norm_term = (s1 * s2).cuda()
        return torch.sum(weight * (input - target) ** 2) / norm_term

    def forward(self, input, target, weight):
        #_assert_no_grad(target)
        return self.weighted_mse_loss(input, target, weight)  

class WeightedBCE(nn.Module):
    """Weighted binary cross-entropy.
    """
    def __init__(self, size_average=True, reduce=True):
        super().__init__()
        self.size_average = size_average
        self.reduce = reduce

    def forward(self, input, target, weight):
        #_assert_no_grad(target)
        #print(torch.max(weight),torch.min(weight))
        assert(torch.min(weight)>=0),'weight out of range'
        return F.binary_cross_entropy(input, target, weight, reduction='mean')

#. 1. Regularization

class BinaryReg(nn.Module):
    """Regularization for encouraging the outputs to be binary.
    """
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, input):
        diff = input - 0.5
        diff = torch.clamp(torch.abs(diff), min=1e-2)
        loss = 1.0 / diff.sum()
        return self.alpha * loss
    
class AuxMSELoss(nn.Module):
    """
    get the WeightedMSELoss of the seg_out + WeightedMSELoss of the aux_out
    """
    def __init__(self):
        super(AuxMSELoss, self).__init__()
        self.mse_loss = WeightedMSE()


    def forward(self, inputs, targets, weights, **kwargs):
        aux_out, seg_out = inputs
        seg_loss = self.mse_loss(seg_out, targets,weights)
        aux_targets = self._scale_target(targets, (aux_out.size(2),aux_out.size(3), aux_out.size(4)))
        aux_weights = self._scale_target(weights, (aux_out.size(2),aux_out.size(3), aux_out.size(4)))
        aux_loss = self.mse_loss(aux_out, aux_targets,aux_weights)
        loss =  seg_loss
        loss = loss + 0.5 * aux_loss
        return loss

    @staticmethod
    def _scale_target(targets_, scaled_size):
        targets = targets_.clone().float()
        targets = F.interpolate(targets, size=scaled_size, mode='trilinear',align_corners=True)
        return targets.float()

class AuxBCELoss(nn.Module):
    """
    get WeightedBCELoss of the seg_out + WeightedBCELoss of the aux_out
    """
    def __init__(self):
        super(AuxBCELoss, self).__init__()
        self.bce_loss = WeightedBCE()


    def forward(self, inputs, targets, weights, **kwargs):
        aux_out, seg_out = inputs
        seg_loss = self.bce_loss(seg_out, targets,weights)
        aux_targets = self._scale_target(targets, (aux_out.size(2),aux_out.size(3), aux_out.size(4)))
        aux_weights = self._scale_target(weights, (aux_out.size(2),aux_out.size(3), aux_out.size(4)))
        aux_loss = self.bce_loss(aux_out, aux_targets,aux_weights)
        loss =  seg_loss
        loss = loss + 0.5 * aux_loss
        return loss

    @staticmethod
    def _scale_target(targets_, scaled_size):
        targets = targets_.clone().float()
        targets = F.interpolate(targets, size=scaled_size, mode='trilinear',align_corners=True)
        return targets.float()


class OhemMSELoss(nn.Module):
    """Online hard negative mining, select the top n weighted MSE Loss
    """
    def __init__(self, thresh, min_kept, reduction='elementwise_mean'):
        super(OhemMSELoss, self).__init__()
        self.thresh = thresh
        self.min_kept = min_kept

        self.reduction = reduction

        self.mse_loss = OhemWeightedMSE()


    def forward(self, predict, target, weight, **kwargs):

        loss_matrix = self.mse_loss(predict, target, weight).contiguous().view(-1, )

        sort_loss_matrix, sort_indices = loss_matrix.sort()
        start_index = max(sort_loss_matrix.numel()-self.min_kept, 0)
        print('threshold:',sort_loss_matrix[start_index])
        select_loss_matrix = sort_loss_matrix[start_index:]
        
        if self.reduction == 'sum':
            return select_loss_matrix.sum()
        elif self.reduction == 'elementwise_mean':
            return select_loss_matrix.mean()
        else:
            raise NotImplementedError('Reduction Error!')
            
class OhemMSELoss2(nn.Module):
    """Online hard negative mining, select the top n MSE Loss
    """
    def __init__(self, thresh, min_kept, reduction='elementwise_mean'):
        super(OhemMSELoss2, self).__init__()
        self.thresh = thresh
        self.min_kept = min_kept

        self.reduction = reduction

        self.mse_loss = OhemMSE()
        self.weighted_mse_loss = OhemWeightedMSE()


    def forward(self, predict, target, weight, **kwargs):

        loss_matrix = self.mse_loss(predict, target).contiguous().view(-1, )

        sort_loss_matrix, sort_indices = loss_matrix.contiguous().sort()
        
        start_index = max(sort_loss_matrix.numel() - self.min_kept, 0)
        threshold = sort_loss_matrix[start_index]
        
        print('threshold:', threshold)
        
        weighted_loss_matrix = self.weighted_mse_loss(predict, target, weight).contiguous().view(-1, )
        sort_weighted_loss_matrix = weighted_loss_matrix[sort_indices]
        select_loss_matrix = sort_weighted_loss_matrix[sort_loss_matrix > threshold]
        
        if self.reduction == 'sum':
            return select_loss_matrix.sum()
        elif self.reduction == 'elementwise_mean':
            return select_loss_matrix.mean()
        else:
            raise NotImplementedError('Reduction Error!')


class AuxOhemMSELoss(nn.Module):
    """Weighted mse on auxiliary output + online hard negative mining on final output
    """
    def __init__(self, thresh, min_kept):
        super(AuxOhemMSELoss, self).__init__()
        self.mse_loss = WeightedMSE()
        #Online hard negative mining, select the top n MSE Loss
        self.ohem_mse_loss = OhemMSELoss2(thresh,min_kept,reduction='elementwise_mean')

    def forward(self, inputs, targets, weights, **kwargs):
        aux_out, seg_out = inputs
        seg_loss = self.ohem_mse_loss(seg_out, targets, weights)
        aux_targets = self._scale_target(targets, (aux_out.size(2),aux_out.size(3), aux_out.size(4)))
        aux_weights = self._scale_target(weights, (aux_out.size(2),aux_out.size(3), aux_out.size(4)))
        aux_loss = self.mse_loss(aux_out, aux_targets, aux_weights)
        loss = 1 * seg_loss
        loss = loss + 0.5 * aux_loss
        return loss

    @staticmethod
    def _scale_target(targets_, scaled_size):
        targets = targets_.clone().float()
        targets = F.interpolate(targets, size=scaled_size, mode='trilinear',align_corners=True)
        return targets.float()

    
class AuxOhemMSELoss2(nn.Module):
    """Weighted mse on final output + online hard negative mining on auxiliary output
    """
    def __init__(self, thresh, min_kept):
        super(AuxOhemMSELoss2, self).__init__()
        self.mse_loss = WeightedMSE()
        self.ohem_mse_loss = OhemMSELoss2(thresh,min_kept,reduction='elementwise_mean')

    def forward(self, inputs, targets, weights, **kwargs):
        aux_out, seg_out = inputs
        seg_loss = self.mse_loss(seg_out, targets, weights)
        aux_targets = self._scale_target(targets, (aux_out.size(2),aux_out.size(3), aux_out.size(4)))
        aux_weights = self._scale_target(weights, (aux_out.size(2),aux_out.size(3), aux_out.size(4)))
        aux_loss = self.ohem_mse_loss(aux_out, aux_targets, aux_weights)
        loss = 1 * seg_loss
        loss = loss + 0.5 * aux_loss
        return loss

    @staticmethod
    def _scale_target(targets_, scaled_size):
        targets = targets_.clone().float()
        targets = F.interpolate(targets, size=scaled_size, mode='trilinear',align_corners=True)
        return targets.float()
    
class OhemBCELoss(nn.Module):
    """online hard negative mining
    """
    def __init__(self, thresh, min_kept, reduction = 'elementwise_mean'):
        super(OhemBCELoss, self).__init__()
        self.thresh = thresh
        self.min_kept = min_kept

        self.reduction = reduction

        self.bce_loss = OhemWeightedBCE()
        self.ignore_label = -100

    def forward(self, predict, target, weight, **kwargs):

        mask = target.contiguous().view(-1,)!= self.ignore_label  #binary mask
        sort_prob, sort_indices = predict.contiguous().view(-1,)[mask].contiguous().sort()
        min_threshold = sort_prob[min(self.min_kept, sort_prob.numel() - 1)]# the number of element
        threshold = max(min_threshold, self.thresh)
        loss_matirx = self.bce_loss(predict, target, weight).contiguous().view(-1,)
        print(loss_matirx.size())
        print(mask.size())
        sort_loss_matirx = loss_matirx[mask][sort_indices]
        select_loss_matrix = sort_loss_matirx[sort_prob < threshold]
        if self.reduction == 'sum':
            return select_loss_matrix.sum()
        elif self.reduction == 'elementwise_mean':
            return select_loss_matrix.mean()
        else:
            raise NotImplementedError('Reduction Error!')


class AuxOhemBCELoss(nn.Module):
    """Weighted bce on Auxiliary Output + online hard negative mining on final output
    """
    def __init__(self, thresh, min_kept):
        super(AuxOhemBCELoss, self).__init__()
        self.bce_loss = WeightedBCE()
        self.ohem_bce_loss = OhemBCELoss(thresh,min_kept,reduction='elementwise_mean')

    def forward(self, inputs, targets, weights, **kwargs):
        aux_out, seg_out = inputs
        seg_loss = self.ohem_bce_loss(seg_out, targets, weights)
        aux_targets = self._scale_target(targets, (aux_out.size(2),aux_out.size(3), aux_out.size(4)))
        aux_weights = self._scale_target(weights, (aux_out.size(2),aux_out.size(3), aux_out.size(4)))
        aux_loss = self.bce_loss(aux_out, aux_targets, aux_weights)
        loss = 1 * seg_loss
        loss = loss + 0.5 * aux_loss
        return loss

    @staticmethod
    def _scale_target(targets_, scaled_size):
        targets = targets_.clone().float()
        targets = F.interpolate(targets, size=scaled_size, mode='trilinear',align_corners=True)
        return targets.float()
    
    
class OhemWeightedBCE(nn.Module):
    """Weighted binary cross-entropy for OhemBCELoss.
       Return a matrix rather than a number
    """
    def __init__(self, size_average=True, reduce=True):
        super().__init__()
        self.size_average = size_average
        self.reduce = reduce

    def forward(self, input, target, weight):
        #_assert_no_grad(target)
        return F.binary_cross_entropy(input, target, weight, reduction='none')

class OhemMSE(nn.Module):
    """mean-squared error for OHEM.
       Return a matrix rather than a number
    """
    def __init__(self):
        super().__init__()

    def ohem_mse_loss(self, input, target):
        s1 = torch.prod(torch.tensor(input.size()[2:]).float())
        s2 = input.size()[0]
        norm_term = (s1 * s2).cuda()
        return (input - target) ** 2/ norm_term

    def forward(self, input, target):
        # _assert_no_grad(target)
        return self.ohem_mse_loss(input, target)

class OhemWeightedMSE(nn.Module):
    """Weighted mean-squared error for OHEM.
       Return a matrix rather than a number
    """
    def __init__(self):
        super().__init__()

    def ohem_weighted_mse_loss(self, input, target, weight):
        s1 = torch.prod(torch.tensor(input.size()[2:]).float())
        s2 = input.size()[0]
        norm_term = (s1 * s2).cuda()
        return weight * (input - target) ** 2/norm_term

    def forward(self, input, target, weight):
        # _assert_no_grad(target)
        return self.ohem_weighted_mse_loss(input, target, weight)

class DiscriminativeLoss(nn.Module):
    """
        Discriminative Loss
    """
    def __init__(self, device, delta_var=0.5, delta_dist=1.5,
                 norm=2, alpha=1.0, beta=1.0, gamma=0.0001,
                 usegpu=True, size_average=True):
        super(DiscriminativeLoss, self).__init__()
        self.delta_var = delta_var
        self.delta_dist = delta_dist
        self.norm = norm
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.usegpu = usegpu
        self.device = device
        assert self.norm in [1, 2]

    def forward(self, input, target, n_clusters):
        return self._discriminative_loss(input, target, n_clusters)

    def _discriminative_loss(self, input, target, n_clusters):
        # pdb.set_trace()
        bs, n_features, point_num = input.size()
        max_n_clusters = target.size(1)


        input = input.contiguous().view(bs, n_features, point_num)
        target = target.contiguous().view(bs, max_n_clusters, point_num)

        c_means, bs_list = self._cluster_means(input, target, n_clusters)
        l_var = self._variance_term(input, target, c_means, n_clusters, bs_list)
        l_dist = self._distance_term(c_means, n_clusters, bs_list)
        l_reg = self._regularization_term(c_means, n_clusters, bs_list)

        loss = self.alpha * l_var + self.beta * l_dist + self.gamma * l_reg

        return loss

    def _cluster_means(self, input, target, n_clusters):
        bs, n_features, n_loc = input.size()
        max_n_clusters = target.size(1)

        # bs, n_features, max_n_clusters, n_loc
        input = input.unsqueeze(2).expand(bs, n_features, max_n_clusters, n_loc)
        # bs, 1, max_n_clusters, n_loc
        target = target.unsqueeze(1)
        # bs, n_features, max_n_clusters, n_loc
        input = input * target  # bs, n_features, max_n_clusters, n_loc

        means = []
        bs_list = []
        for i in range(bs):
            class_index = torch.nonzero(target[i, 0].sum(1) > 0).view(-1)
            #print(class_index) # 0,1
            
            if len(class_index)==2:
                # n_features, n_clusters, n_loc
                # input_sample = input[i, :, :n_clusters[i]]
                input_sample = input[i, :, class_index]
                # 1, n_clusters, n_loc,
                target_sample = target[i, :, class_index]

                # n_features, n_cluster
                mean_sample = input_sample.sum(2) / target_sample.sum(2) # 36,2

                # padding
                n_pad_clusters = max_n_clusters - n_clusters[i]
                assert n_pad_clusters >= 0
                if n_pad_clusters > 0:
                    pad_sample = torch.zeros(n_features, n_pad_clusters)
                    pad_sample = Variable(pad_sample)
                    if self.usegpu:
                        pad_sample = pad_sample.cuda()
                    mean_sample = torch.cat((mean_sample, pad_sample), dim=1)

                means.append(mean_sample) 
                bs_list.append(i) # ignore the batch which only have one class

        # bs, n_features, max_n_clusters
        means = torch.stack(means)
        
        return means, bs_list

    def _variance_term(self, input, target, c_means, n_clusters, bs_list):
        input = input[bs_list]
        target = target[bs_list]
        bs, n_features, n_loc = input.size()
        max_n_clusters = target.size(1)

        # bs, n_features, max_n_clusters, n_loc
        c_means = c_means.unsqueeze(3).expand(bs, n_features, max_n_clusters, n_loc)
        #print(
        # bs, n_features, max_n_clusters, n_loc
        input = input.unsqueeze(2).expand(bs, n_features, max_n_clusters, n_loc)
        # bs, max_n_clusters, n_loc
        var = (torch.clamp(torch.norm((input - c_means), self.norm, 1) -
                           self.delta_var, min=0) ** 2) * target   ###

        var_term = 0
        for i in range(bs):
            class_index = torch.nonzero(target[i].sum(1) > 0).view(-1)
            # n_clusters, n_loc
            # var_sample = var[i, :n_clusters[i]]
            var_sample = var[i, class_index]
            # n_clusters, n_loc
            # target_sample = target[i, :n_clusters[i]]
            target_sample = target[i, class_index]
            # n_clusters
            c_var = var_sample.sum(1) / target_sample.sum(1)
            var_term += c_var.sum() / n_clusters[i]
        var_term /= bs

        return var_term

    def _distance_term(self, c_means, n_clusters, bs_list):
        bs, n_features, max_n_clusters = c_means.size()

        dist_term = 0
        for i in range(bs):
            #i = bs_list[j]
            if n_clusters[i] <= 1:
                continue
            # n_features, n_clusters
            mean_sample = c_means[i, :, :n_clusters[i]]

            # n_features, n_clusters, n_clusters
            means_a = mean_sample.unsqueeze(2).expand(n_features, n_clusters[i], n_clusters[i])
            means_b = means_a.permute(0, 2, 1)
            diff = means_a - means_b

            margin = 2 * self.delta_dist * (1.0 - torch.eye(n_clusters[i]))
            margin = Variable(margin).to(self.device)
            # if self.usegpu:
            #     margin = margin.cuda()
            c_dist = torch.sum(torch.clamp(margin - torch.norm(diff, self.norm, 0), min=0) ** 2)
            dist_term += c_dist / (2 * n_clusters[i] * (n_clusters[i] - 1))
        dist_term /= bs

        return dist_term

    def _regularization_term(self, c_means, n_clusters, bs_list):
        bs, n_features, max_n_clusters = c_means.size()

        reg_term = 0
        for i in range(bs):
            #i = bs_list[j]
            # n_features, n_clusters
            mean_sample = c_means[i, :, :n_clusters[i]]
            reg_term += torch.mean(torch.norm(mean_sample, self.norm, 0))
        reg_term /= bs

        return reg_term
    
if __name__ == "__main__":
    targ = torch.empty(8, 2, 100).random_(2).cuda()
    inputs = torch.randn(8,36,100).cuda()
    #inputs = aux,seg
    crit = DiscriminativeLoss(device='cuda',usegpu=False)
    loss = crit(inputs,targ,[2]*8)
    print(loss)