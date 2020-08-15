import math

import torch

from .utils import TensorList, operation, optimization


class FactorizedConvProblem(optimization.L2Problem):
    def __init__(self, training_samples: TensorList, y: TensorList,
                 use_attention: bool, filter_reg: torch.Tensor, projection_reg,
                 sample_weights: TensorList, projection_activation,
                 att_activation, response_activation, cfg):

        self.training_samples = training_samples
        self.y = y
        self.sample_weights = sample_weights
        self.use_attetion = use_attention
        self.filter_reg = filter_reg
        self.projection_reg = projection_reg
        self.projection_activation = projection_activation
        self.att_activation = att_activation
        self.response_activation = response_activation
        self.cfg = cfg

        if self.use_attetion:
            self.diag_M = self.filter_reg.concat(projection_reg).concat(
                projection_reg).concat(projection_reg)
        else:
            self.diag_M = self.filter_reg.concat(projection_reg)

    def __call__(self, x: TensorList):

        if self.use_attetion:
            filter = x[:1]
            fc2 = x[1:2]
            fc1 = x[2:3]
            P = x[3:4]
        else:
            filter = x[:len(x) // 2]  # w2 in paper
            P = x[len(x) // 2:]  # w1 in paper
        cfg = self.cfg

        # Compression module
        compressed_samples = operation.conv1x1(self.training_samples, P).apply(
            self.projection_activation)

        # Attention module
        if self.use_attetion:
            if cfg["channel_attention"]:
                global_average = operation.adaptive_avg_pool2d(
                    compressed_samples, 1)
                temp_variables = operation.conv1x1(global_average, fc1).apply(
                    self.att_activation)
                channel_attention = operation.sigmoid(
                    operation.conv1x1(temp_variables, fc2))
            else:
                channel_attention = TensorList([
                    torch.zeros(compressed_samples[0].size(0),
                                compressed_samples[0].size(1), 1, 1).cuda()
                ])

            if cfg["spatial_attention"] == 'none':
                spatial_attention = TensorList([
                    torch.zeros(compressed_samples[0].size(0), 1,
                                compressed_samples[0].size(2),
                                compressed_samples[0].size(3)).cuda()
                ])
            elif cfg["spatial_attention"] == 'pool':
                spatial_attention = operation.spatial_attention(
                    compressed_samples, dim=1, keepdim=True)
            else:
                raise NotImplementedError('No spatial attention Implemented')

            compressed_samples = operation.matmul(compressed_samples, spatial_attention) + \
                                 operation.matmul(compressed_samples, channel_attention)

        # Filter module
        residuals = operation.conv2d(compressed_samples, filter,
                                     mode='same').apply(
                                         self.response_activation)
        residuals = residuals - self.y
        residuals = self.sample_weights.sqrt().view(-1, 1, 1, 1) * residuals

        residuals.extend(self.filter_reg.apply(math.sqrt) * filter)
        if self.use_attetion:
            residuals.extend(self.projection_reg.apply(math.sqrt) * fc2)
            residuals.extend(self.projection_reg.apply(math.sqrt) * fc1)
        residuals.extend(self.projection_reg.apply(math.sqrt) * P)

        return residuals

    def ip_input(self, a: TensorList, b: TensorList):

        if self.use_attetion:
            a_filter = a[:1]
            a_f2 = a[1:2]
            a_f1 = a[2:3]
            a_P = a[3:]
            b_filter = b[:1]
            b_f2 = b[1:2]
            b_f1 = b[2:3]
            b_P = b[3:]

            ip_out = operation.conv2d(a_filter, b_filter).view(-1)
            ip_out += operation.conv2d(a_f2.view(1, -1, 1, 1),
                                       b_f2.view(1, -1, 1, 1)).view(-1)
            ip_out += operation.conv2d(a_f1.view(1, -1, 1, 1),
                                       b_f1.view(1, -1, 1, 1)).view(-1)
            ip_out += operation.conv2d(a_P.view(1, -1, 1, 1),
                                       b_P.view(1, -1, 1, 1)).view(-1)

            return ip_out.concat(ip_out.clone()).concat(ip_out.clone()).concat(
                ip_out.clone())

        else:
            num = len(a) // 2  # Number of filters
            a_filter = a[:num]
            b_filter = b[:num]
            a_P = a[num:]
            b_P = b[num:]

            # Filter inner product
            # ip_out = a_filter.reshape(-1) @ b_filter.reshape(-1)
            ip_out = operation.conv2d(a_filter, b_filter).view(-1)

            # Add projection matrix part
            # ip_out += a_P.reshape(-1) @ b_P.reshape(-1)
            ip_out += operation.conv2d(a_P.view(1, -1, 1, 1),
                                       b_P.view(1, -1, 1, 1)).view(-1)

            # Have independent inner products for each filter
            return ip_out.concat(ip_out.clone())

    def M1(self, x: TensorList):
        # factorized convolution
        return x / self.diag_M


class ConvProblem(optimization.L2Problem):
    def __init__(self, training_samples: TensorList, y: TensorList,
                 filter_reg: torch.Tensor, sample_weights: TensorList,
                 response_activation):
        self.training_samples = training_samples
        self.y = y
        self.filter_reg = filter_reg
        self.sample_weights = sample_weights
        self.response_activation = response_activation

    def __call__(self, x: TensorList):
        """
        Compute residuals
        :param x: [filters]
        :return: [data_terms, filter_regularizations]
        """
        # Do convolution and compute residuals
        residuals = operation.conv2d(self.training_samples, x,
                                     mode='same').apply(
                                         self.response_activation)
        residuals = residuals - self.y
        residuals = self.sample_weights.sqrt().view(-1, 1, 1, 1) * residuals

        # Add regularization for projection matrix
        residuals.extend(self.filter_reg.apply(math.sqrt) * x)

        return residuals

    def ip_input(self, a: TensorList, b: TensorList):
        # return a.reshape(-1) @ b.reshape(-1)
        # return (a * b).sum()
        return operation.conv2d(a, b).view(-1)
