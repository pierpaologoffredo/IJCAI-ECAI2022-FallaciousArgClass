import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.utils.checkpoint
from torch.nn import CrossEntropyLoss
from transformers import LongformerPreTrainedModel, LongformerModel
from transformers.models.longformer.modeling_longformer import LongformerClassificationHead
from transformers.utils import logging

logger = logging.get_logger(__name__)


class LongformerPolitics(LongformerPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        
        self.longformer = LongformerModel(config, add_pooling_layer=False)
        self.classifier = LongformerClassificationHead(config)

        self.init_weights()

        self.alpha = 0.5 # default alpha in Bert is 0.1
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        global_attention_mask=None,
        head_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,

        argComp = None,
        argRel = None,
        argText = None,

        span=None
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if global_attention_mask is None:
            logger.info("Initializing global attention on CLS token...")
            global_attention_mask = torch.zeros_like(input_ids)
            # global attention on cls token
            global_attention_mask[:, 0] = 1
            
        sen_outputs = self.longformer(
            input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            head_mask=head_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        # Main dialogue
        pooled_sequence_output = sen_outputs[0]
        sen_logits = self.classifier(pooled_sequence_output)

        # Snippet
        span_input_ids, span_attention_mask = [*span.values()]
        span_outputs = self.longformer(
            span_input_ids,
            attention_mask=span_attention_mask,
        )
        pooled_span_output = span_outputs[0]
        span_logits = self.classifier(pooled_span_output)

        # Argumentation Component
        if argComp is not None:
            argComp_input_ids, argComp_attention_mask = [*argComp.values()]
            argComp_outputs = self.longformer(
                argComp_input_ids,
                attention_mask=argComp_attention_mask,
            )
            pooled_argComp_output = argComp_outputs[0]
            argComp_logits = self.classifier(pooled_argComp_output)
        
        # Argumentation Relation
        if argRel is not None:
            argRel_input_ids, argRel_attention_mask = [*argRel.values()]
            argRel_outputs = self.longformer(
                argRel_input_ids,
                attention_mask=argRel_attention_mask,
            )
            pooled_argRel_output = argRel_outputs[0]
            argRel_logits = self.classifier(pooled_argRel_output)

        # Argumentation Text
        if argText is not None:
            argText_input_ids, argText_attention_mask = [*argText.values()]
            argText_outputs = self.longformer(
                argText_input_ids,
                attention_mask=argText_attention_mask,
            )
            pooled_argText_output = argText_outputs[0]
            argText_logits = self.classifier(pooled_argText_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            sen_loss = loss_fct(sen_logits.view(-1, self.num_labels), labels.view(-1))
            span_loss = loss_fct(span_logits.view(-1, self.num_labels), labels.view(-1))

            if argComp is None and argRel is None and argText is None:
                joint_loss = self.alpha * ((sen_loss + span_loss))/2
                
            if argComp is not None and argRel is None and argText is None:
                argComp_loss = loss_fct(argComp_logits.view(-1, self.num_labels), labels.view(-1))
                joint_loss = self.alpha * ((sen_loss + span_loss + argComp_loss))/3
                
            if argRel is not None and argComp is None and argText is None:
                argRel_loss = loss_fct(argRel_logits.view(-1, self.num_labels), labels.view(-1))
                joint_loss = self.alpha * ((sen_loss + span_loss + argRel_loss))/3
                
            if argText is not None and argComp is None and argRel is None:
                argText_loss = loss_fct(argText_logits.view(-1, self.num_labels), labels.view(-1))
                joint_loss = self.alpha * ((sen_loss + span_loss + argText_loss))/3
                
            if argComp is not None and argRel is not None and argText is None:
                argComp_loss = loss_fct(argComp_logits.view(-1, self.num_labels), labels.view(-1))
                argRel_loss = loss_fct(argRel_logits.view(-1, self.num_labels), labels.view(-1))
                joint_loss = self.alpha * ((sen_loss + span_loss + argRel_loss + argComp_loss))/4
                
            if argComp is not None and argRel is None and argText is not None:
                argComp_loss = loss_fct(argComp_logits.view(-1, self.num_labels), labels.view(-1))
                argText_loss = loss_fct(argText_logits.view(-1, self.num_labels), labels.view(-1))
                joint_loss = self.alpha * ((sen_loss + span_loss + argComp_loss + argText_loss))/4
                
            if argComp is None and argRel is not None and argText is not None:
                argRel_loss = loss_fct(argRel_logits.view(-1, self.num_labels), labels.view(-1))
                argText_loss = loss_fct(argText_logits.view(-1, self.num_labels), labels.view(-1))
                joint_loss = self.alpha * ((sen_loss + span_loss + argRel_loss + argText_loss))/4
                
            if argComp is not None and argRel is not None and argText is not None:
                argComp_loss = loss_fct(argComp_logits.view(-1, self.num_labels), labels.view(-1))
                argRel_loss = loss_fct(argRel_logits.view(-1, self.num_labels), labels.view(-1))
                argText_loss = loss_fct(argText_logits.view(-1, self.num_labels), labels.view(-1))
                joint_loss = self.alpha * ((sen_loss + span_loss + argRel_loss + argComp_loss + argText_loss))/5
        else: 
            joint_loss = None

        return (joint_loss, sen_logits, span_logits)