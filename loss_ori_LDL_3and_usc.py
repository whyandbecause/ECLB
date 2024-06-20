import torch
import torch.nn as nn
import torch.nn.functional as F


def CrossEntropy(outputs, targets):
    log_softmax_outputs = F.log_softmax(outputs, dim=1)
    softmax_targets = F.softmax(targets, dim=1)

    return -(log_softmax_outputs*softmax_targets).sum(dim=1).mean()


class UscLoss(nn.Module):
    def __init__(self, batch_size, device='cuda', temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature).to(device)) 
        # self.register_buffer("negatives_mask", (
        #     ~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(device)).float())  

    def forward(self, emb_i, emb_j): 
        #z_i = F.normalize(emb_i, dim=1)  # (bs, dim)  --->  (bs, dim)
        #z_j = F.normalize(emb_j, dim=1)  # (bs, dim)  --->  (bs, dim)
        negatives_mask = (~torch.eye(emb_i.shape[0] * 2, emb_i.shape[0] * 2, dtype=bool).cuda()).float()
        representations = torch.cat([emb_i, emb_j], dim=0)  # repre: (2*bs, dim)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)  # simi_mat: (2*bs, 2*bs)

        sim_ij = torch.diag(similarity_matrix, emb_i.shape[0])  # bs
        sim_ji = torch.diag(similarity_matrix, -emb_i.shape[0])  # bs
        positives = torch.cat([sim_ij, sim_ji], dim=0)  # 2*bs

        nominator = torch.exp(positives / self.temperature)  # 2*bs
        denominator = negatives_mask * torch.exp(similarity_matrix / self.temperature)  # 2*bs, 2*bs

        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))  # 2*bs
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss


class LabelLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super(LabelLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]  #   2
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        #   256 x 512
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature   #   256 x   512
            anchor_count = contrast_count   #   2
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        #   print (anchor_dot_contrast.size())  256 x 256

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        #mask = mask.repeat(anchor_count, contrast_count)

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1)+1e-5)
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07, alpha=0.7):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.alpha = alpha
    def forward(self, features, labels=None, mask=None, D=None, L=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]  #   2
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        #   256 x 512
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature   #   256 x   512
            anchor_count = contrast_count   #   2
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        #   print (anchor_dot_contrast.size())  256 x 256

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        #--------------ldl----------------
        label_mask = torch.scatter(torch.ones_like(logits), 1, torch.arange(batch_size * anchor_count).view(-1, 1).to(device), 0)
        label_logits = torch.exp(-logits) * label_mask
        #A_hat = torch.pow(torch.sum(label_logits, dim=1), -0.5) * torch.eye(label_logits.shape[0]).to(device)
        #P = torch.mm(torch.mm(A_hat, label_logits), A_hat)
        A_hat = torch.inverse(torch.sum(label_logits, dim=1) * torch.eye(label_logits.shape[0]).to(device))
        P = torch.eye(label_logits.shape[0]).to(device) - torch.mm(A_hat, label_logits)
        D = torch.cat((D, D), dim=0)
        L = torch.cat((L, L), dim=0)
        D_hat = self.alpha * torch.mm(P, D) + (1-self.alpha) * L
        D_hat = F.softmax(D_hat, dim=1)
        #--------------------------------------

        # tile mask
        #mask = mask.repeat(anchor_count, contrast_count)

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1)+1e-5)
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss, D_hat


class TwoCropTransform:
    def __init__(self, transform1, transform2):
        self.transform1 = transform1
        self.transform2 = transform2

    def __call__(self, x):
        return [self.transform1(x), self.transform2(x)]

'''
1                       1
    1                       1
        1
            1
                1
                    1




'''