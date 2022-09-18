from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.roberta.modeling_roberta import RobertaConfig, RobertaForSequenceClassification
import torch.nn as nn
import torch
from torch.nn import BCEWithLogitsLoss

class Classifier(nn.Module):
    def __init__(self, model_name="roberta", num_labels=2):
        super().__init__()
        if model_name == 'roberta':
            model_config = RobertaConfig.from_pretrained(pretrained_model_name_or_path="roberta-base",
                                                         hidden_dropout_prob=0.1, num_labels=num_labels)
            self.model = RobertaForSequenceClassification.from_pretrained("roberta-base", config=model_config)

    def forward(self, input_ids, attention_mask, labels=None):
        
        outputs = self.model.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        logits = self.model.classifier(sequence_output)
        
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits, labels)
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )