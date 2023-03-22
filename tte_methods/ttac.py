import os
import torch
import torch.nn as nn
import torch.optim as optim

from utils.dataloader import ImageNet_val_subset_data

class TTAC(nn.Module):
    """EATA adapts a model by entropy minimization during testing.
    Once EATAed, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, args, class_num=1000):
        super().__init__()

        self.class_num = class_num

        self.model = model
        self.ext = torch.nn.Sequential(*(list(model.children())[:-1]))
        self.classifier = model.fc
        
        self.ext_mean, self.ext_cov, self.ext_mean_categories, self.ext_cov_categories = offline(self.ext, self.classifier, args)
        
        self.optimizer = optim.SGD(self.ext.parameters(), lr=0.000707107, momentum=0.9)

        # sample_predict_ema_logit = torch.zeros(len(target_dataset_adapt), class_num, dtype=torch.float)
        # sample_alpha = torch.ones(len(target_dataset_adapt), dtype=torch.float)
        # ema_alpha = 0.9

        self.ema_ext_mu = self.ext_mean_categories.clone()
        self.ema_ext_cov = self.ext_cov_categories.clone()
        self.ema_ext_total_mu = torch.zeros(2048).cuda()
        self.ema_ext_total_cov = torch.zeros(2048, 2048).cuda()

        self.class_ema_length = 64
        self.ema_n = torch.ones(class_num).cuda() * self.class_ema_length
        self.ema_total_n = 0.

        self.loss_scale = 0.05

        bias = self.ext_cov.max().item() / 30.
        self.template_ext_cov = torch.eye(2048).cuda() * bias
        
    def forward(self, te_inputs):
        outputs = self.forward_and_adapt_ttac(te_inputs)
        return outputs

    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt_ttac(self, te_inputs, class_num=1000):
        
        self.model.train()
        self.optimizer.zero_grad()

        ####### feature alignment ###########
        loss = 0.
        inputs = te_inputs.cuda()

        feat_ext = self.ext(inputs).squeeze(-1).squeeze(-1)
        logit = self.classifier(feat_ext)
        softmax_logit = logit.softmax(dim=-1)
        pro, pseudo_label = softmax_logit.max(dim=-1)
        pseudo_label_mask = (pro > 0.9)
        feat_ext2 = feat_ext[pseudo_label_mask]
        pseudo_label2 = pseudo_label[pseudo_label_mask].cuda()

        # Gaussian Mixture Distribution Alignment
        b, d = feat_ext2.shape
        feat_ext2_categories = torch.zeros(class_num, b, d).cuda() # K, N, D
        feat_ext2_categories.scatter_add_(dim=0, index=pseudo_label2[None, :, None].expand(-1, -1, d), src=feat_ext2[None, :, :])

        num_categories = torch.zeros(class_num, b, dtype=torch.int).cuda() # K, N
        num_categories.scatter_add_(dim=0, index=pseudo_label2[None, :], src=torch.ones_like(pseudo_label2[None, :], dtype=torch.int))

        self.ema_n += num_categories.sum(dim=1) # K
        alpha = torch.where(self.ema_n > self.class_ema_length, torch.ones(class_num, dtype=torch.float).cuda() / self.class_ema_length, 1. / (self.ema_n + 1e-10))

        delta_pre = (feat_ext2_categories - self.ema_ext_mu[:, None, :]) * num_categories[:, :, None] # K, N, D
        delta = alpha[:, None] * delta_pre.sum(dim=1) # K, D
        ext_mu_categories = self.ema_ext_mu + delta
        ext_sigma_categories = self.ema_ext_cov + alpha[:, None] * ((delta_pre ** 2).sum(dim=1) - num_categories.sum(dim=1)[:, None] * self.ema_ext_cov) - delta ** 2
        
        for label in pseudo_label2.unique():
            if self.ema_n[label] > self.class_ema_length:
                bias = self.ext_cov.max().item() / 30.
                self.template_ext_cov = torch.eye(2048).cuda() * bias
                source_domain = torch.distributions.MultivariateNormal(self.ext_mean_categories[label, :], torch.diag_embed(self.ext_cov_categories[label, :]) + self.template_ext_cov)
                target_domain = torch.distributions.MultivariateNormal(ext_mu_categories[label, :], torch.diag_embed(ext_sigma_categories[label, :]) + self.template_ext_cov)
                loss += (torch.distributions.kl_divergence(source_domain, target_domain) + torch.distributions.kl_divergence(target_domain, source_domain)) * self.loss_scale
        with torch.no_grad():
            self.ema_ext_mu = ext_mu_categories.detach()
            self.ema_ext_cov = ext_sigma_categories.detach()

        # Gaussian Distribution Alignment
        b = feat_ext.shape[0]
        self.ema_total_n += b
        alpha = 1. / 1280 if self.ema_total_n > 1280 else 1. / self.ema_total_n
        delta = alpha * (feat_ext - self.ema_ext_total_mu).sum(dim=0)
        tmp_mu = self.ema_ext_total_mu + delta
        tmp_cov = self.ema_ext_total_cov + alpha * ((feat_ext - self.ema_ext_total_mu).t() @ (feat_ext - self.ema_ext_total_mu) - b * self.ema_ext_total_cov) - delta[:, None] @ delta[None, :]

        with torch.no_grad():
            self.ema_ext_total_mu = tmp_mu.detach()
            self.ema_ext_total_cov = tmp_cov.detach()
        source_domain = torch.distributions.MultivariateNormal(self.ext_mean, self.ext_cov + self.template_ext_cov)
        target_domain = torch.distributions.MultivariateNormal(tmp_mu, tmp_cov + self.template_ext_cov)
        loss += (torch.distributions.kl_divergence(source_domain, target_domain) + torch.distributions.kl_divergence(target_domain, source_domain)) * self.loss_scale

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        #### Test ####
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(te_inputs.cuda())
        self.model.train()
        return outputs

def offline(ext, classifier, args, num_classes=1000):
    if os.path.exists('offline.pth'):
        data = torch.load('offline.pth')
        return data
    
    trloader = ImageNet_val_subset_data(data_dir=args.imagenet_path, 
                                            batch_size=args.batch_size, shuffle=args.shuffle, subset_size=-1)
    ext.eval()

    feat_ext_mean = torch.zeros(2048).cuda()
    feat_ext_variance = torch.zeros(2048, 2048).cuda()

    feat_ext_mean_categories = torch.zeros(num_classes, 2048).cuda() # K, D
    feat_ext_variance_categories = torch.zeros(num_classes, 2048).cuda()

    ema_n = torch.zeros(num_classes).cuda()
    ema_total_n = 0

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(trloader):
            feat = ext(inputs.cuda()).squeeze(-1).squeeze(-1) # N, D
            b, d = feat.shape
            labels = classifier(feat).argmax(dim=-1)

            feat_ext_categories = torch.zeros(num_classes, b, d).cuda()
            feat_ext_categories.scatter_add_(dim=0, index=labels[None, :, None].expand(-1, -1, d), src=feat[None, :, :])
            
            num_categories = torch.zeros(num_classes, b, dtype=torch.int).cuda()
            num_categories.scatter_add_(dim=0, index=labels[None, :], src=torch.ones_like(labels[None, :], dtype=torch.int))
            ema_n += num_categories.sum(dim=1)
            alpha_categories = 1 / (ema_n + 1e-10)  # K
            delta_pre = (feat_ext_categories - feat_ext_mean_categories[:, None, :]) * num_categories[:, :, None] # K, N, D
            delta = alpha_categories[:, None] * delta_pre.sum(dim=1) # K, D
            feat_ext_mean_categories += delta
            feat_ext_variance_categories += alpha_categories[:, None] * ((delta_pre ** 2).sum(dim=1) - num_categories.sum(dim=1)[:, None] * feat_ext_variance_categories) \
                                          - delta ** 2
            
            ema_total_n += b
            alpha = 1 / (ema_total_n + 1e-10)
            delta_pre = feat - feat_ext_mean[None, :] # b, d
            delta = alpha * (delta_pre).sum(dim=0)
            feat_ext_mean += delta
            feat_ext_variance += alpha * (delta_pre.t() @ delta_pre - b * feat_ext_variance) - delta[:, None] @ delta[None, :]
            print('offline process rate: %.2f%%\r' % ((batch_idx + 1) / len(trloader) * 100.), end='')


    torch.save((feat_ext_mean, feat_ext_variance, feat_ext_mean_categories, feat_ext_variance_categories), 'offline.pth')
    return feat_ext_mean, feat_ext_variance, feat_ext_mean_categories, feat_ext_variance_categories

