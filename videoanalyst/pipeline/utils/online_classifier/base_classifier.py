import math

import numpy as np

import torch
import torch.nn
import torch.nn.functional as F

from .optim import ConvProblem, FactorizedConvProblem
from .utils import TensorList, augmentation, dcf, fourier, operation
from .utils.optimization import (ConjugateGradient, GaussNewtonCG,
                                 GradientDescent, GradientDescentL2, NewtonCG)
from .utils.plotting import show_tensor


class BaseClassifier(object):
    def __init__(self, model, cfg):
        self.model = model
        self.cfg = cfg
        self.device = torch.device("cuda")

    def initialize(self, image, state):
        cfg = self.cfg
        # Initialize some stuff
        self.frame_num = 1

        self.img_sample_sz = torch.round(
            torch.Tensor([cfg["x_size"], cfg["x_size"]]))
        raw_fea_size = cfg["raw_fea_size"]
        self.output_sz = torch.Tensor((cfg["x_size"], cfg["x_size"]))
        self.feature_sz = TensorList(
            [torch.Tensor([raw_fea_size, raw_fea_size])])
        self.kernel_size = TensorList([cfg["z_kernel_size"]])

        # Get all geometry information
        self.pos = torch.Tensor(
            [state[1] + (state[3] - 1) / 2, state[0] + (state[2] - 1) / 2])
        self.target_sz = torch.Tensor([state[3], state[2]])

        wc_z = self.target_sz[1] + cfg["context_amount"] * sum(self.target_sz)
        hc_z = self.target_sz[0] + cfg["context_amount"] * sum(self.target_sz)
        s_z = np.sqrt(wc_z * hc_z)
        self.scale = cfg["z_size"] / s_z

        # hinge mask
        self.use_attention_layer = cfg["use_attention_layer"] and (
            cfg["channel_attention"] or cfg["spatial_attention"] != 'none')

        # Optimization options
        self.precond_learning_rate = cfg["precond_learning_rate"]
        self.CG_forgetting_rate = cfg["CG_forgetting_rate"]
        if not self.CG_forgetting_rate or self.precond_learning_rate >= 1:
            self.direction_forget_factor = 0
        else:
            self.direction_forget_factor = (
                1 - self.precond_learning_rate)**self.CG_forgetting_rate

        # Initialize some learning things
        self.init_learning()

        # Extract and transform sample
        train_x, aug_imgs = self.generate_init_samples(image)
        self.aug_imgs = aug_imgs

        # Initialize projection matrix
        self.init_projection_matrix(train_x)

        # Generate label function
        init_y = self.init_label_function(train_x)

        # Init memory
        self.init_memory(train_x)

        # Init attention layer
        if self.use_attention_layer:
            self.init_attention_layer(train_x)

        # Init optimizer and do initial optimization
        self.init_optimization(train_x, init_y)

    def init_optimization(self, train_x, init_y):
        # Initialize filter
        cfg = self.cfg
        filter_init_method = cfg["filter_init_method"]
        self.filter = TensorList([
            x.new_zeros(1, cdim, sz[0], sz[1]) for x, cdim, sz in zip(
                train_x, self.compressed_dim, self.kernel_size)
        ])
        if filter_init_method == 'zeros':
            pass
        elif filter_init_method == 'randn':
            for f in self.filter:
                f.normal_(0, 1 / f.numel())
        else:
            raise ValueError('Unknown "filter_init_method"')

        # Get parameters
        self.update_projection_matrix = cfg["update_projection_matrix"] and cfg[
            "use_projection_matrix"]
        optimizer = cfg["optimizer"]
        # Setup factorized joint optimization
        plot_show = cfg["online_debug_show"]
        if self.update_projection_matrix:
            self.projection_reg = TensorList([cfg["projection_reg"]])

            self.joint_problem = FactorizedConvProblem(
                self.init_training_samples,
                init_y,
                self.use_attention_layer,
                self.filter_reg,
                self.projection_reg,
                self.init_sample_weights,
                self.projection_activation,
                self.att_activation,
                self.response_activation,
                cfg=self.cfg)

            # Variable containing both filter and projection matrix
            if self.use_attention_layer:
                joint_var = self.filter.concat(self.channel_att_fc2).concat(
                    self.channel_att_fc1).concat(self.projection_matrix)
            else:
                joint_var = self.filter.concat(self.projection_matrix)

            # Initialize optimizer
            if optimizer == 'GaussNewtonCG':
                self.joint_optimizer = GaussNewtonCG(
                    self.joint_problem,
                    joint_var,
                    plotting=plot_show,
                    analyze=cfg["analyze_convergence"],
                    fig_num=(12, 13, 14))
            elif optimizer == 'GradientDescentL2':
                self.joint_optimizer = GradientDescentL2(
                    self.joint_problem,
                    joint_var,
                    cfg["optimizer_step_length"],
                    cfg["optimizer_momentum"],
                    plotting=plot_show,
                    debug=cfg["analyze_convergence"],
                    fig_num=(12, 13))
            elif optimizer == 'NewtonCG':
                self.joint_optimizer = NewtonCG(
                    self.joint_problem,
                    joint_var,
                    plotting=plot_show,
                    analyze=cfg["analyze_convergence"],
                    fig_num=(12, 13, 14))
            elif optimizer == 'GradientDescent':
                self.joint_optimizer = GradientDescent(
                    self.joint_problem,
                    joint_var,
                    cfg["optimizer_step_length"],
                    cfg["optimizer_momentum"],
                    plotting=plot_show,
                    debug=cfg["analyze_convergence"],
                    fig_num=(12, 13))

            # Do joint optimization
            if isinstance(cfg["init_CG_iter"], (list, tuple)):
                self.joint_optimizer.run(cfg["init_CG_iter"])
            else:
                self.joint_optimizer.run(
                    cfg["init_CG_iter"] // cfg["init_GN_iter"],
                    cfg["init_GN_iter"])

            if cfg["analyze_convergence"]:
                opt_name = 'CG' if cfg["CG_optimizer"] else 'GD'
                for val_name, values in zip(['loss', 'gradient'], [
                        self.joint_optimizer.losses,
                        self.joint_optimizer.gradient_mags
                ]):
                    val_str = ' '.join(
                        ['{:.8e}'.format(v.item()) for v in values])
                    file_name = '{}_{}.txt'.format(opt_name, val_name)
                    with open(file_name, 'a') as f:
                        f.write(val_str + '\n')
                raise RuntimeError('Exiting')

        # Re-project samples with the new projection/attention layer
        compressed_samples = self.project_sample(self.init_training_samples,
                                                 self.projection_matrix)
        for train_samp, init_samp in zip(self.training_samples,
                                         compressed_samples):
            train_samp[:init_samp.shape[0], ...] = init_samp

        # Initialize optimizer
        self.conv_problem = ConvProblem(self.training_samples, self.y,
                                        self.filter_reg, self.sample_weights,
                                        self.response_activation)

        if optimizer == 'GaussNewtonCG':
            self.filter_optimizer = ConjugateGradient(
                self.conv_problem,
                self.filter,
                fletcher_reeves=cfg["fletcher_reeves"],
                direction_forget_factor=self.direction_forget_factor,
                debug=plot_show,
                fig_num=(12, 13))
        elif optimizer == 'GradientDescent':
            self.filter_optimizer = GradientDescent(
                self.conv_problem,
                self.filter,
                cfg["optimizer_step_length"],
                cfg["optimizer_momentum"],
                debug=plot_show,
                fig_num=12)

        # Transfer losses from previous optimization
        if self.update_projection_matrix:
            self.filter_optimizer.residuals = self.joint_optimizer.residuals
            self.filter_optimizer.losses = self.joint_optimizer.losses

        if not self.update_projection_matrix:
            self.filter_optimizer.run(cfg["init_CG_iter"])

        # Post optimization
        self.filter_optimizer.run(cfg["post_init_CG_iter"])

        # Free memory
        del self.init_training_samples
        if self.use_projection_matrix:
            del self.joint_problem, self.joint_optimizer

    def track(self, scale_idx=None):
        cfg = self.cfg

        self.frame_num += 1

        # Get infer sample without shift
        self.sample_pos = self.pos.round()
        self.feat_x = self.extract_processed_sample(scale_idx=scale_idx)

        # Compute scores
        scores_raw = self.apply_filter(self.feat_x)
        s, flag = self.localize_target(scores_raw)

        if cfg["online_debug_show"]:
            show_tensor(s,
                        5,
                        title='Classification Max score = {:.2f}'.format(
                            torch.max(s).item()))

        return flag, s

    def update(self, state, scale_z, flag):
        cfg = self.cfg

        if flag != 'not_found':
            self.pos = torch.Tensor([
                state[1] + (state[3] - 1) / 2., state[0] + (state[2] - 1) / 2.
            ])
            self.target_sz = torch.Tensor([state[3], state[2]])

        # Check flags and set learning rate if hard negative
        update_flag = flag not in ['not_found', 'uncertain']
        hard_negative = (flag == 'hard_negative')
        learning_rate = cfg[
            "hard_negative_learning_rate"] if hard_negative else cfg[
                "precond_learning_rate"]

        if update_flag:
            # Create label for sample
            train_y = self.get_label_function(self.sample_pos, scale_z,
                                              self.target_sz)
            train_x = TensorList([x for x in self.feat_x])

            # Update memory
            self.update_memory(train_x, train_y, learning_rate)

        # Train filter
        if hard_negative:
            self.filter_optimizer.run(cfg["hard_negative_CG_iter"])
        elif (self.frame_num - 1) % cfg["train_skipping"] == 0:
            self.filter_optimizer.run(cfg["CG_iter"])

    def apply_filter(self, sample_x: TensorList):
        return operation.conv2d(sample_x, self.filter, mode='same')

    def localize_target(self, scores_raw):
        cfg = self.cfg
        sf_weighted = fourier.cfft2(scores_raw) / (scores_raw.size(2) *
                                                   scores_raw.size(3))
        for i, (sz, ksz) in enumerate(zip(self.feature_sz, self.kernel_size)):
            sf_weighted[i] = fourier.shift_fs(
                sf_weighted[i],
                math.pi * (1 - torch.Tensor([ksz[0] % 2, ksz[1] % 2]) / sz))

        scores_fs = fourier.sum_fs(sf_weighted)
        scores = fourier.sample_fs(scores_fs, None).squeeze()

        if cfg["advanced_localization"]:
            return self.localize_advanced(scores)

        # Shift the score output for visualization purposes
        if cfg["online_debug_show"]:
            sz = scores.shape[-2:]
            scores = torch.cat(
                [scores[..., sz[0] // 2:, :], scores[..., :sz[0] // 2, :]], -2)
            scores = torch.cat(
                [scores[..., :, sz[1] // 2:], scores[..., :, :sz[1] // 2]], -1)

        return scores, None

    def localize_advanced(self, scores):
        """Does the advanced localization with hard negative detection and target not found."""
        cfg = self.cfg
        sz = scores.shape[-2:]

        # Shift scores back
        scores = torch.cat([
            scores[..., (sz[0] + 1) // 2:, :], scores[..., :(sz[0] + 1) // 2, :]
        ], -2)
        scores = torch.cat([
            scores[..., :, (sz[1] + 1) // 2:], scores[..., :, :(sz[1] + 1) // 2]
        ], -1)

        # Find maximum
        max_score1, max_disp1 = dcf.max2d(scores)
        max_disp1 = max_disp1.float().cpu().view(-1)
        target_disp1 = max_disp1 - self.output_sz // 2

        if max_score1.item() < cfg["target_not_found_threshold"]:
            return scores, 'not_found'

        # Mask out target neighborhood
        target_neigh_sz = cfg[
            "target_neighborhood_scale"] * self.target_sz * self.scale
        tneigh_top = max(
            round(max_disp1[0].item() - target_neigh_sz[0].item() / 2), 0)
        tneigh_bottom = min(
            round(max_disp1[0].item() + target_neigh_sz[0].item() / 2 + 1),
            sz[0])
        tneigh_left = max(
            round(max_disp1[1].item() - target_neigh_sz[1].item() / 2), 0)
        tneigh_right = min(
            round(max_disp1[1].item() + target_neigh_sz[1].item() / 2 + 1),
            sz[1])
        scores_masked = scores.clone()
        scores_masked[..., tneigh_top:tneigh_bottom, tneigh_left:
                      tneigh_right] = 0

        # Find new maximum
        max_score2, max_disp2 = dcf.max2d(scores_masked)
        max_disp2 = max_disp2.float().cpu().view(-1)
        target_disp2 = max_disp2 - self.output_sz // 2

        # Handle the different cases
        if max_score2 > cfg["distractor_threshold"] * max_score1:
            disp_norm1 = torch.sqrt(torch.sum(target_disp1**2))
            disp_norm2 = torch.sqrt(torch.sum(target_disp2**2))
            disp_threshold = cfg["displacement_scale"] * math.sqrt(
                sz[0] * sz[1]) / 2

            if disp_norm2 > disp_threshold and disp_norm1 < disp_threshold:
                return scores, 'hard_negative'
            if disp_norm2 < disp_threshold and disp_norm1 > disp_threshold:
                return scores, 'hard_negative'
            if disp_norm2 > disp_threshold and disp_norm1 > disp_threshold:
                return scores, 'uncertain'

            # If also the distractor is close, return with highest score
            return scores, 'uncertain'

        if max_score2 > cfg[
                "hard_negative_threshold"] * max_score1 and max_score2 > cfg[
                    "target_not_found_threshold"]:
            return scores, 'hard_negative'

        return scores, None

    def extract_sample(self, im: torch.Tensor):

        with torch.no_grad():
            self.model.instance(im.to(self.device))
        feat = TensorList([self.model.cf])

        return feat

    def extract_transformed_samples(self, im: torch.Tensor, transforms):
        ims_augmented = torch.cat([T(im) for T in transforms])
        feature_map = self.extract_sample(ims_augmented)

        return feature_map, ims_augmented

    def project_sample(self, x: TensorList, proj_matrix=None):
        cfg = self.cfg
        # Apply projection matrix
        if proj_matrix is None:
            proj_matrix = self.projection_matrix

        compressed_samples = operation.conv2d(x, proj_matrix).apply(
            self.projection_activation)

        if self.use_attention_layer:
            if cfg["channel_attention"]:
                global_average = operation.adaptive_avg_pool2d(
                    compressed_samples, 1)
                temp_variables = operation.conv1x1(global_average,
                                                   self.channel_att_fc1).apply(
                                                       self.att_activation)
                channel_attention = operation.sigmoid(
                    operation.conv1x1(temp_variables, self.channel_att_fc2))
            else:
                channel_attention = TensorList([
                    torch.zeros(compressed_samples[0].size(0),
                                compressed_samples[0].size(1), 1, 1).cuda()
                ])

            if cfg["spatial_attention"] == 'pool':
                spatial_attention = operation.spatial_attention(
                    compressed_samples, dim=1, keepdim=True)
            else:
                spatial_attention = TensorList([
                    torch.zeros(compressed_samples[0].size(0), 1,
                                compressed_samples[0].size(2),
                                compressed_samples[0].size(3)).cuda()
                ])

            compressed_samples = operation.matmul(compressed_samples, spatial_attention) + \
                                 operation.matmul(compressed_samples, channel_attention)

        return compressed_samples

    def extract_processed_sample(self, im: torch.Tensor = None, scale_idx=None):

        with torch.no_grad():
            if scale_idx is not None:
                x = TensorList([self.model.cf[scale_idx].unsqueeze(0)])
            else:
                x = TensorList([self.model.cf])
        feat = self.project_sample(x)

        return feat

    def init_learning(self):
        cfg = self.cfg
        # Filter regularization
        self.filter_reg = TensorList([cfg["filter_reg"]])

        # Activation function after the projection matrix (phi_1 in the paper)
        projection_activation = cfg["projection_activation"]
        if isinstance(projection_activation, tuple):
            projection_activation, act_param = projection_activation

        if projection_activation == 'none':
            self.projection_activation = lambda x: x
        elif projection_activation == 'relu':
            self.projection_activation = torch.nn.ReLU(inplace=True)
        elif projection_activation == 'elu':
            self.projection_activation = torch.nn.ELU(inplace=True)
        elif projection_activation == 'mlu':
            self.projection_activation = lambda x: F.elu(
                F.leaky_relu(x, 1 / act_param), act_param)
        else:
            raise ValueError('Unknown activation')

        # Activation function for attention layer
        att_activation = cfg["att_activation"]
        if isinstance(att_activation, tuple):
            att_activation, act_param = att_activation
        if att_activation == 'none':
            self.att_activation = lambda x: x
        elif att_activation == 'relu':
            self.att_activation = torch.nn.ReLU(inplace=True)
        elif att_activation == 'elu':
            self.att_activation = torch.nn.ELU(inplace=True)
        elif att_activation == 'mlu':
            self.att_activation = lambda x: F.elu(
                F.leaky_relu(x, 1 / act_param), act_param)
        else:
            raise ValueError('Unknown activation')

        # Activation function after the output scores (phi_2 in the paper)
        response_activation = cfg["reponse_activation"]
        if isinstance(response_activation, tuple):
            response_activation, act_param = response_activation

        if response_activation == 'none':
            self.response_activation = lambda x: x
        elif response_activation == 'relu':
            self.response_activation = torch.nn.ReLU(inplace=True)
        elif response_activation == 'elu':
            self.response_activation = torch.nn.ELU(inplace=True)
        elif response_activation == 'mlu':
            self.response_activation = lambda x: F.elu(
                F.leaky_relu(x, 1 / act_param), act_param)
        else:
            raise ValueError('Unknown activation')

    def generate_init_samples(self, im: torch.Tensor) -> TensorList:
        """Generate augmented initial samples."""

        # Compute augmentation size
        cfg = self.cfg
        aug_expansion_factor = cfg["augmentation_expansion_factor"]
        aug_expansion_sz = self.img_sample_sz.clone()
        aug_output_sz = None
        if aug_expansion_factor is not None and aug_expansion_factor != 1:
            aug_expansion_sz = (self.img_sample_sz *
                                aug_expansion_factor).long()
            aug_expansion_sz += (aug_expansion_sz -
                                 self.img_sample_sz.long()) % 2
            aug_expansion_sz = aug_expansion_sz.float()
            aug_output_sz = self.img_sample_sz.long().tolist()

        # Random shift operator
        self.get_rand_shift = lambda: None
        self.random_shift_factor = cfg["augmentation_shift_factor"]
        if self.random_shift_factor > 0:
            self.get_rand_shift = lambda: (
                (torch.rand(2) - 0.5) * self.img_sample_sz * self.
                random_shift_factor).long().tolist()

        # Create transofmations
        self.transforms = [augmentation.Identity(aug_output_sz)]
        if cfg["augmentation_shift"]:
            self.transforms.extend([
                augmentation.Translation(shift, aug_output_sz)
                for shift in cfg["augmentation_shift"]
            ])
        if cfg["augmentation_relativeshift"]:
            get_absolute = lambda shift: (torch.Tensor(shift) * self.
                                          img_sample_sz / 2).long().tolist()
            self.transforms.extend([
                augmentation.Translation(get_absolute(shift), aug_output_sz)
                for shift in cfg["augmentation_relativeshift"]
            ])
        if cfg["augmentation_fliplr"]:
            self.transforms.append(
                augmentation.FlipHorizontal(aug_output_sz,
                                            self.get_rand_shift()))
        if cfg["augmentation_blur"]:
            self.transforms.extend([
                augmentation.Blur(sigma, aug_output_sz, self.get_rand_shift())
                for sigma in cfg["augmentation_blur"]
            ])
        if cfg["augmentation_scale"]:
            self.transforms.extend([
                augmentation.Scale(scale_factor, aug_output_sz,
                                   self.get_rand_shift())
                for scale_factor in cfg["augmentation_scale"]
            ])
        if cfg["augmentation_rotate"]:
            self.transforms.extend([
                augmentation.Rotate(angle, aug_output_sz, self.get_rand_shift())
                for angle in cfg["augmentation_rotate"]
            ])

        # Generate initial samples
        init_samples, aug_imgs = self.extract_transformed_samples(
            im, self.transforms)

        # Remove augmented samples for those that shall not have
        for i, use_aug in enumerate(TensorList([cfg["use_augmentation"]])):
            if not use_aug:
                init_samples[i] = init_samples[i][0:1, ...]

        # Add dropout samples
        if cfg["augmentation_dropout"]:
            num, prob = cfg["augmentation_dropout"]
            self.transforms.extend(self.transforms[:1] * num)
            for i, use_aug in enumerate(TensorList([cfg["use_augmentation"]])):
                if use_aug:
                    init_samples[i] = torch.cat([
                        init_samples[i],
                        F.dropout2d(init_samples[i][0:1, ...].expand(
                            num, -1, -1, -1),
                                    p=prob,
                                    training=True)
                    ])

        return init_samples, aug_imgs

    def init_projection_matrix(self, x):
        # Set if using projection matrix
        cfg = self.cfg
        self.use_projection_matrix = cfg["use_projection_matrix"]

        if self.use_projection_matrix:
            self.compressed_dim = TensorList([cfg["compressed_dim"]])
            proj_init_method = cfg["proj_init_method"]
            if proj_init_method == 'pca':
                x_mat = TensorList([
                    e.permute(1, 0, 2, 3).reshape(e.shape[1], -1).clone()
                    for e in x
                ])
                x_mat -= x_mat.mean(dim=1, keepdim=True)
                cov_x = x_mat @ x_mat.t()
                self.projection_matrix = TensorList([
                    None if cdim is None else torch.svd(C)[0]
                    [:, :cdim].t().unsqueeze(-1).unsqueeze(-1).clone()
                    for C, cdim in zip(cov_x, self.compressed_dim)
                ])
            elif proj_init_method == 'randn':
                self.projection_matrix = TensorList([
                    None if cdim is None else ex.new_zeros(
                        cdim, ex.shape[1], 1, 1).normal_(
                            0, 1 / math.sqrt(ex.shape[1]))
                    for ex, cdim in zip(x, self.compressed_dim)
                ])
        else:
            self.compressed_dim = x.size(1)
            self.projection_matrix = TensorList([None] * len(x))

    def init_attention_layer(self, x):
        cfg = self.cfg

        self.compressed_dim = TensorList([cfg["compressed_dim"]])
        self.channel_att_fc1 = TensorList([
            None if cdim is None else ex.new_zeros(
                cdim // 2, cdim, 1, 1).normal_(0, 1 / math.sqrt(cdim))
            for ex, cdim in zip(x, self.compressed_dim)
        ])
        self.channel_att_fc2 = TensorList([
            None if cdim is None else ex.new_zeros(
                cdim, cdim // 2, 1, 1).normal_(0, 1 / math.sqrt(cdim))
            for ex, cdim in zip(x, self.compressed_dim)
        ])

    def init_label_function(self, train_x):
        cfg = self.cfg
        # Allocate label function
        self.y = TensorList([
            x.new_zeros(cfg["sample_memory_size"], 1, x.shape[2], x.shape[3])
            for x in train_x
        ])

        # Output sigma factor
        output_sigma_factor = cfg["output_sigma_factor"]

        self.sigma = (
            self.feature_sz / self.img_sample_sz * self.target_sz *
            self.scale).prod().sqrt() * output_sigma_factor * torch.ones(2)

        # Generate label functions
        target_center_norm = (self.pos - self.pos.round()) / self.img_sample_sz
        for y, sig, sz, ksz, x in zip(self.y, self.sigma, self.feature_sz,
                                      self.kernel_size, train_x):
            center_pos = sz * target_center_norm + 0.5 * torch.Tensor(
                [(ksz[0] + 1) % 2, (ksz[1] + 1) % 2])
            for i, T in enumerate(self.transforms[:x.shape[0]]):
                sample_center = center_pos + torch.Tensor(
                    T.shift) / self.img_sample_sz * sz
                #y[i, 0, ...] = dcf.label_function_spatial(sz, sig, sample_center, scale)
                y[i, 0, ...] = dcf.label_function_spatial(
                    sz, sig, sample_center)

        # Return only the ones to use for initial training
        return TensorList(
            [y[:x.shape[0], ...] for y, x in zip(self.y, train_x)])

    def init_memory(self, train_x):
        cfg = self.cfg
        # Initialize first-frame training samples
        self.num_init_samples = train_x.size(0)
        self.init_sample_weights = TensorList(
            [x.new_ones(1) / x.shape[0] for x in train_x])
        self.init_training_samples = train_x

        # Sample counters and weights
        self.num_stored_samples = self.num_init_samples.copy()
        self.previous_replace_ind = [None] * len(self.num_stored_samples)
        self.sample_weights = TensorList(
            [x.new_zeros(cfg["sample_memory_size"]) for x in train_x])
        for sw, init_sw, num in zip(self.sample_weights,
                                    self.init_sample_weights,
                                    self.num_init_samples):
            sw[:num] = init_sw

        # Initialize memory
        self.training_samples = TensorList([
            x.new_zeros(cfg["sample_memory_size"], cdim, x.shape[2], x.shape[3])
            for x, cdim in zip(train_x, self.compressed_dim)
        ])

    def update_memory(self,
                      sample_x: TensorList,
                      sample_y: TensorList,
                      learning_rate=None):
        replace_ind = self.update_sample_weights(self.sample_weights,
                                                 self.previous_replace_ind,
                                                 self.num_stored_samples,
                                                 self.num_init_samples,
                                                 learning_rate)
        self.previous_replace_ind = replace_ind
        for train_samp, x, ind in zip(self.training_samples, sample_x,
                                      replace_ind):
            train_samp[ind:ind + 1, ...] = x
        for y_memory, y, ind in zip(self.y, sample_y, replace_ind):
            y_memory[ind:ind + 1, ...] = y
        self.num_stored_samples += 1

    def update_sample_weights(self,
                              sample_weights,
                              previous_replace_ind,
                              num_stored_samples,
                              num_init_samples,
                              learning_rate=None):
        # Update weights and get index to replace in memory
        cfg = self.cfg
        replace_ind = []
        for sw, prev_ind, num_samp, num_init in zip(sample_weights,
                                                    previous_replace_ind,
                                                    num_stored_samples,
                                                    num_init_samples):
            lr = learning_rate
            # if lr is None:
            #     lr = fpar.learning_rate

            init_samp_weight = cfg["init_samples_minimum_weight"]
            if init_samp_weight == 0:
                init_samp_weight = None
            s_ind = 0 if init_samp_weight is None else num_init

            if num_samp == 0 or lr == 1:
                sw[:] = 0
                sw[0] = 1
                r_ind = 0
            else:
                # Get index to replace
                _, r_ind = torch.min(sw[s_ind:], 0)
                r_ind = r_ind.item() + s_ind

                # Update weights
                if prev_ind is None:
                    sw /= 1 - lr
                    sw[r_ind] = lr
                else:
                    sw[r_ind] = sw[prev_ind] / (1 - lr)

            sw /= sw.sum()
            if init_samp_weight is not None and sw[:num_init].sum(
            ) < init_samp_weight:
                sw /= init_samp_weight + sw[num_init:].sum()
                sw[:num_init] = init_samp_weight / num_init

            replace_ind.append(r_ind)

        return replace_ind

    def get_label_function(self, sample_pos, scale_z, target_sz=torch.ones(2)):
        # Generate label function
        train_y = TensorList()
        target_center_norm = (self.pos - sample_pos) / self.img_sample_sz
        # target_center_norm = (self.pos - sample_pos) * scale_z / self.img_sample_sz
        target_sz[0] / target_sz[1]

        for sig, sz, ksz in zip(self.sigma, self.feature_sz, self.kernel_size):
            center = sz * target_center_norm + 0.5 * torch.Tensor(
                [(ksz[0] + 1) % 2, (ksz[1] + 1) % 2])
            train_y.append(dcf.label_function_spatial(sz, sig, center))

        return train_y

    def transform_score(self, scores_raw):
        cfg = self.cfg

        sf_weighted = fourier.cfft2(scores_raw) / (scores_raw.size(2) *
                                                   scores_raw.size(3))
        for i, (sz, ksz) in enumerate(zip(self.feature_sz, self.kernel_size)):
            sf_weighted[i] = fourier.shift_fs(
                sf_weighted[i],
                math.pi * (1 - torch.Tensor([ksz[0] % 2, ksz[1] % 2]) / sz))

        scores_fs = fourier.sum_fs(sf_weighted)
        scores = fourier.sample_fs(scores_fs, self.output_sz).squeeze()

        if cfg["online_debug_show"]:
            sz = scores.shape[-2:]
            scores = torch.cat(
                [scores[..., sz[0] // 2:, :], scores[..., :sz[0] // 2, :]], -2)
            scores = torch.cat(
                [scores[..., :, sz[1] // 2:], scores[..., :, :sz[1] // 2]], -1)

        return scores
