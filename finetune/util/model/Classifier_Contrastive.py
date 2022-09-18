from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.roberta.modeling_roberta import RobertaConfig, RobertaForSequenceClassification
import torch.nn as nn
import torch
import torch.nn.functional as F

class Classifier_Contrastive(nn.Module):
    def __init__(self, model_name="roberta", num_labels=2, pretrained_model_path=None):
        super().__init__()
        if model_name == 'roberta':
            model_config = RobertaConfig.from_pretrained(pretrained_model_name_or_path="roberta-base",
                                                         hidden_dropout_prob=0.1, num_labels=num_labels)
            self.model = RobertaForSequenceClassification.from_pretrained("roberta-base", config=model_config)
            self.model.roberta.load_state_dict(torch.load(pretrained_model_path))
            
            
        self.sup_con_loss_fct = SupConLoss(temperature=0.1)
        self.ce_loss_fct = SmoothCrossEntropyLoss(smoothing=0.0)
        
        
    def forward(self, input_ids, attention_mask, labels=None, mode='train'):      
        outputs = self.model.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]

        features = sequence_output[:,0,:].unsqueeze(1)
        features_for_classification = sequence_output

        if mode=='train':
            for i in [0, 0.06, 0.12, 0.18,0.24, ]:
                set_dropout_mf(self.model.roberta, i)
                
                pos_feature = self.model.roberta(input_ids=input_ids, attention_mask=attention_mask)[0][:,0,:].unsqueeze(1)

                features = torch.cat([features, pos_feature], dim=1)
            sup_con_loss = self.sup_con_loss_fct(F.normalize(features, p=2, dim=2), labels)
        else:
            sup_con_loss = None
        
        logits = self.model.classifier(features_for_classification)
        cls_loss = self.ce_loss_fct(logits, labels)
        
        if mode=='train':
            loss = sup_con_loss  +  cls_loss * 0.1
        else:
            loss = cls_loss
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

def set_dropout_mf(
    model:nn, 
    w
    ):
    """Alters the dropouts in the embeddings.
    """
    # ------ set hidden dropout -------#
    if hasattr(model, 'module'):
        model.module.embeddings.dropout.p = w
        for i in model.module.encoder.layer:
            i.attention.self.dropout.p = w
            i.attention.output.dropout.p = w
            i.output.dropout.p = w        
    else:
        model.embeddings.dropout.p = w
        for i in model.encoder.layer:
            i.attention.self.dropout.p = w
            i.attention.output.dropout.p = w
            i.output.dropout.p = w
        
    return model

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
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

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0) # (bsz*view, hidden)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count # view
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach() # (bsz*view, bsz*view)

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count) # (bsz*view, bsz*view)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        
       
        # logits_mask -> tensor([
        #     [0., 1., 1., 1.],
        #     [1., 0., 1., 1.],
        #     [1., 1., 0., 1.],
        #     [1., 1., 1., 0.]])
        mask = mask * logits_mask
        


        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss        
    
    
from torch.nn.modules.loss import _WeightedLoss    
class SmoothCrossEntropyLoss(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    def k_one_hot(self, targets:torch.Tensor, n_classes:int, smoothing=0.0):
        with torch.no_grad():
            targets = torch.empty(size=(targets.size(0), n_classes),
                                  device=targets.device) \
                                  .fill_(smoothing /(n_classes-1)) \
                                  .scatter_(1, targets.data.unsqueeze(1), 1.-smoothing)
        return targets

    def reduce_loss(self, loss):
        return loss.mean() if self.reduction == 'mean' else loss.sum() \
        if self.reduction == 'sum' else loss

    def forward(self, inputs, targets):
        assert 0 <= self.smoothing < 1

        targets = self.k_one_hot(targets, inputs.size(-1), self.smoothing)
        log_preds = F.log_softmax(inputs, -1)

        if self.weight is not None:
            log_preds = log_preds * self.weight.unsqueeze(0)

        return self.reduce_loss(-(targets * log_preds).sum(dim=-1))