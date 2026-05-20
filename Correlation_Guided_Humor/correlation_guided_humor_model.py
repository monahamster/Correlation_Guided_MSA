import logging
import math
import os
import sys
from typing import Optional, Tuple
from modules.transformer import TransformerEncoder

import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn import CrossEntropyLoss, MSELoss
from einops import rearrange, repeat
from einops.layers.torch import Reduce
from torch.nn import L1Loss, MSELoss
from torch.autograd import Function
from math import pi, log
from functools import wraps
from MoE import MoE, MLP
from torch import nn, einsum
import torch.nn.functional as F


from transformers import BertPreTrainedModel
from transformers.models.bert.modeling_bert import BertEmbeddings, BertEncoder, BertPooler
from transformers.activations import gelu, gelu_new
from transformers import BertConfig
import numpy as np

import torch.optim as optim
from itertools import chain

from transformers.modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)

# from global_configs_class import ACOUSTIC_DIM, VISUAL_DIM, TEXT_DIM, DEVICE# *

logger = logging.getLogger(__name__)

_CONFIG_FOR_DOC = "BertConfig"
_TOKENIZER_FOR_DOC = "BertTokenizer"

BERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "bert-base-uncased",
    "bert-base-cased",
    # See all BERT models at https://huggingface.co/models?filter=bert
]
# ori_output : output
class GLoMo_BertModel(BertPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.config = config
        self.config.output_hidden_states=True
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.d_l = args.d_l
        self.gran_t = args.gran_t
        self.linear1 = nn.Linear(in_features=args.TEXT_DIM, out_features=self.d_l)
        self.proj_l = nn.Conv1d(args.TEXT_DIM, self.d_l, kernel_size=3, stride=1, padding=1, bias=False)
        nn.init.xavier_uniform_(self.proj_l.weight)
        self.avgmaxpooling_t = nn.AdaptiveMaxPool1d(self.gran_t)
        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        r"""
    Return:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (:obj:`torch.FloatTensor`: of shape :obj:`(batch_size, hidden_size)`):
            Last layer hidden-state of the first token of the sequence (classification token)
            further processed by a Linear layer and a Tanh activation function. The Linear
            layer weights are trained from the next sentence prediction (classification)
            objective during pre-training.

            This output is usually *not* a good summary
            of the semantic content of the input, you're often better with averaging or pooling
            the sequence of hidden-states for the whole input sequence.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(
                input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape, device
        )

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            (
                encoder_batch_size,
                encoder_sequence_length,
                _,
            ) = encoder_hidden_states.size()
            encoder_hidden_shape = (
                encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(
                    encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(
                encoder_attention_mask
            )
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(
            head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )
        
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        last_sequence_output = encoder_outputs[0]# 36*60*768:bsz*msl*dim
        last2_sequence_output = encoder_outputs.hidden_states[-2]
        attn_outputs_t = self.linear1(last_sequence_output[:,0])
        last12_fine_output = torch.concat((last_sequence_output, last2_sequence_output), dim=1)# concat on the msl dim, bsz*msl*dim
        outputs = last12_fine_output.transpose(1,2)# bsz*dim*msl
        outputs_t = self.proj_l(outputs)# 
        fine_output = self.avgmaxpooling_t(outputs_t)
        
        return attn_outputs_t, fine_output

    
class Audio_Video_network(nn.Module): 
    def __init__(self, modality_dim, grans, args = None):
        super(Audio_Video_network, self).__init__()
        self.num_heads = args.num_heads
        self.layers = args.layers
        self.d_l = args.d_l
        self.relu_dropout = args.relu_dropout
        self.res_dropout = args.res_dropout
        self.embed_dropout = args.embed_dropout
        self.attn_dropout = args.attn_dropout
        self.modality_dim = modality_dim # for 
        self.grans = grans
        self.projs = nn.Conv1d(self.modality_dim, self.d_l, kernel_size=3, stride=1, padding=1, bias=False)
        nn.init.xavier_uniform_(self.projs.weight)
        self.avgmaxpoolings_c = nn.AdaptiveMaxPool1d(1)
        self.avgmaxpoolings_f = nn.AdaptiveMaxPool1d(self.grans)
        
        self.encoder = TransformerEncoder(embed_dim=self.d_l,
                                  num_heads= self.num_heads,
                                  layers=self.layers,
                                  attn_dropout= self.attn_dropout,
                                  relu_dropout=self.relu_dropout,   
                                  res_dropout= self.res_dropout,    
                                  embed_dropout=self.embed_dropout,  
                                  attn_mask= False)                 
    def forward(self,feas):
        # the modality dimension can not be divided by num_heads, so first use Conv1d to change the dimensions
        feas = feas.transpose(1, 2)
        feas = self.projs(feas) # , self.modality_dim
        feas = feas.permute(2, 0, 1)
        outputs = self.encoder(feas)# output: [src_len, batch, modality] # [4, 60, 36, 64]
        outputs = outputs.permute(1,2,0)
        coarsed = self.avgmaxpoolings_c(outputs)
        coarsed = coarsed.view(-1,self.d_l)#.squeeze()
        fine_output = self.avgmaxpoolings_f(outputs)
        return coarsed, fine_output


# Attention implementation     
class Attention(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.3):# layers,0.1
        super(Attention, self).__init__()
        self.dim = dim
        self.scale = dim ** -0.5
        self.attention_mlp = nn.Sequential()
        self.attention_mlp.add_module('attention_mlp', nn.Linear(in_features=dim*3, out_features=hidden_dim))
        self.attention_mlp.add_module('attention_mlp_dropout', nn.Dropout(dropout))
        self.attention_mlp.add_module('attention_mlp_activation', nn.ReLU())
        self.fc_att = nn.Linear(hidden_dim, 3)

    def forward(self, feas1, feas2, feas3):
        multi_hidden1 = torch.cat([feas1, feas2, feas3], dim=1) # [bsz, 768*2]
        attention = self.attention_mlp(multi_hidden1) # [bsz, 64]  
        attention = self.fc_att(attention)# [bsz, 2]
        attention = torch.unsqueeze(attention, 2) * self.scale # [bsz, 2, 1]
        attention = attention.softmax(dim = 1)
        multi_hidden2 = torch.stack([feas1, feas2, feas3], dim=2) # [bsz, 768, 2]
        fused_feat = torch.matmul(multi_hidden2, attention) # 
        fused_feat = fused_feat.squeeze() # [bsz, 64]
        fused_feat = fused_feat.view(-1,self.dim)
        return attention, fused_feat


class GLoMo(BertPreTrainedModel):
    def __init__(self, config, args = None):
        super().__init__(config)
        self.num_labels = args.num_labels# here is 1
        self.d_l = args.d_l
        self.gran_t = args.gran_t
        self.gran_a = args.gran_a
        self.gran_v = args.gran_v
        self.bert = GLoMo_BertModel(config, args) #.d_l
        self.dropout = nn.Dropout(args.dropout_prob)
        self.activation = nn.ReLU()
        self.attn_dropout = args.attn_dropout  
        self.num_heads = args.num_heads
        self.relu_dropout = args.relu_dropout
        self.res_dropout = args.res_dropout
        self.embed_dropout = args.embed_dropout
        self.audio_network = Audio_Video_network(args.ACOUSTIC_DIM, args.gran_a, args)
        self.video_network = Audio_Video_network(args.VISUAL_DIM, args.gran_v, args)
        self.moe_t = MoE(
            input_size=args.d_l * self.gran_t,
            output_size=args.d_l,
            num_experts=args.experts_t,
            hidden_size=args.d_l,
            model=MLP,
            k=args.k,
            use_reliability=getattr(args, "use_moe_reliability", False),
            reliability_dim=1,
            reliability_lambda=getattr(args, "moe_reliability_lambda", 0.1),
        )
        self.moe_a = MoE(
            input_size=args.d_l * self.gran_a,
            output_size=args.d_l,
            num_experts=args.experts_a,
            hidden_size=args.d_l,
            model=MLP,
            k=args.k,
            use_reliability=getattr(args, "use_moe_reliability", False),
            reliability_dim=1,
            reliability_lambda=getattr(args, "moe_reliability_lambda", 0.1),
        )
        self.moe_v = MoE(
            input_size=args.d_l * self.gran_v,
            output_size=args.d_l,
            num_experts=args.experts_v,
            hidden_size=args.d_l,
            model=MLP,
            k=args.k,
            use_reliability=getattr(args, "use_moe_reliability", False),
            reliability_dim=1,
            reliability_lambda=getattr(args, "moe_reliability_lambda", 0.1),
        )
        ## multimodal fusion
        self.fc_ts = nn.Linear(in_features=self.d_l*2, out_features=self.d_l)
        self.fc_as = nn.Linear(in_features=self.d_l*2, out_features=self.d_l)
        self.fc_vs = nn.Linear(in_features=self.d_l*2, out_features=self.d_l)
        self.fc_all = nn.Linear(in_features=self.d_l*2, out_features=self.d_l)
        self.attn_cs = Attention(self.d_l, self.d_l)
        self.attn_fs = Attention(self.d_l, self.d_l)
        self.classifier = nn.Linear(in_features=self.d_l*2, out_features= self.num_labels)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_l, nhead=self.num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2) # num_layers 

        # Correlation-guided fusion scaling (CorMulT-style)
        self.use_correlation = getattr(args, "use_correlation", False)
        self.use_fusion_correlation = getattr(args, "use_fusion_correlation", self.use_correlation)
        self.use_moe_reliability = getattr(args, "use_moe_reliability", False)
        self.corr_alpha = getattr(args, "corr_alpha", 1.0)
        self.corr_model = None
        corr_model_path = getattr(args, "corr_model_path", None)
        if (self.use_fusion_correlation or self.use_moe_reliability) and corr_model_path:
            corr_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Correlation_Pretraining", "modality_correlation"))
            if corr_dir not in sys.path:
                sys.path.append(corr_dir)
            try:
                from correlation_models import CorrelationModel
                self.corr_model = CorrelationModel(
                    text_in_dim=args.TEXT_DIM,
                    audio_in_dim=args.ACOUSTIC_DIM,
                    vision_in_dim=args.VISUAL_DIM,
                    d_model=128,
                    num_layers=3,
                    num_heads=4,
                    dim_feedforward=256,
                    dropout=0.1,
                    out_dim=64,
                )
                state = torch.load(corr_model_path, map_location="cpu")
                self.corr_model.load_state_dict(state, strict=True)
                self.corr_model.eval()
                for p in self.corr_model.parameters():
                    p.requires_grad = False
            except Exception as exc:
                logger.warning("Failed to load correlation model: %s", exc)
                self.corr_model = None

        self.init_weights()
        self.init_custom()

    def init_custom(self):
        for conv in (self.audio_network.projs, self.video_network.projs):
            nn.init.kaiming_uniform_(conv.weight, a=math.sqrt(5))
        for encoder in (self.audio_network.encoder, self.video_network.encoder):
            for layer in encoder.layers:
                nn.init.xavier_uniform_(layer.self_attn.in_proj_weight)
                if layer.self_attn.in_proj_bias is not None:
                    nn.init.constant_(layer.self_attn.in_proj_bias, 0.0)
                nn.init.xavier_uniform_(layer.self_attn.out_proj.weight)
                if layer.self_attn.out_proj.bias is not None:
                    nn.init.constant_(layer.self_attn.out_proj.bias, 0.0)
                nn.init.xavier_uniform_(layer.fc1.weight)
                nn.init.constant_(layer.fc1.bias, 0.0)
                nn.init.xavier_uniform_(layer.fc2.weight)
                nn.init.constant_(layer.fc2.bias, 0.0)
                for ln in layer.layer_norms:
                    nn.init.constant_(ln.bias, 0.0)
                    nn.init.constant_(ln.weight, 1.0)
            if encoder.normalize:
                nn.init.constant_(encoder.layer_norm.bias, 0.0)
                nn.init.constant_(encoder.layer_norm.weight, 1.0)
        for moe in (self.moe_t, self.moe_a, self.moe_v):
            nn.init.xavier_uniform_(moe.w_gate)
            nn.init.xavier_uniform_(moe.w_noise)
            if hasattr(moe, "mean"):
                nn.init.constant_(moe.mean, 0.1)
            if hasattr(moe, "std"):
                nn.init.constant_(moe.std, 1.0)

    def _corr_scales(self, input_ids, acoustic, visual, token_type_ids=None, position_ids=None):
        if not self.use_fusion_correlation or self.corr_model is None:
            return None
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        with torch.no_grad():
            text_emb = self.bert.embeddings(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
            )
            F_T, F_A, F_V = self.corr_model(text_emb, acoustic, visual)
            F_T_mean = F_T.mean(dim=1)
            F_A_mean = F_A.mean(dim=1)
            F_V_mean = F_V.mean(dim=1)
            cor_ta = F.cosine_similarity(F_T_mean, F_A_mean, dim=-1)
            cor_tv = F.cosine_similarity(F_T_mean, F_V_mean, dim=-1)
            cor_av = F.cosine_similarity(F_A_mean, F_V_mean, dim=-1)
            cor_ta = (cor_ta + 1.0) / 2.0
            cor_tv = (cor_tv + 1.0) / 2.0
            cor_av = (cor_av + 1.0) / 2.0
            r_t = (cor_ta + cor_tv) / 2.0
            r_a = (cor_ta + cor_av) / 2.0
            r_v = (cor_tv + cor_av) / 2.0
            scale_t = 1.0 + self.corr_alpha * (r_t - 0.5)
            scale_a = 1.0 + self.corr_alpha * (r_a - 0.5)
            scale_v = 1.0 + self.corr_alpha * (r_v - 0.5)
        return scale_t, scale_a, scale_v

    def forward(
        self,
        input_ids,
        visual,
        acoustic,
        label_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
    ):

        coarsed_t, finegrained_t = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        coarsed_a, finegrained_a = self.audio_network(acoustic)
        coarsed_v, finegrained_v = self.video_network(visual)
        moe_scales = None
        if self.use_moe_reliability:
            moe_scales = self._corr_scales(
                input_ids=input_ids,
                acoustic=acoustic,
                visual=visual,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
            )
        if moe_scales is not None:
            r_t, r_a, r_v = moe_scales
        else:
            r_t = r_a = r_v = None
        fine_ts, moe_loss_ts = self.moe_t(torch.flatten(finegrained_t, start_dim=1), reliability=r_t)
        fine_as, moe_loss_as = self.moe_a(torch.flatten(finegrained_a, start_dim=1), reliability=r_a)
        fine_vs, moe_loss_vs = self.moe_v(torch.flatten(finegrained_v, start_dim=1), reliability=r_v)
        
   
        ## Coarsed_guided fusion module##
        # first use default MLP or transformerencoders
        all_feas = torch.stack((coarsed_t, fine_ts, coarsed_a, fine_as, coarsed_v, fine_vs), dim=0)
        # multimodal fusion modules-->
        h = self.transformer_encoder(all_feas)
        if self.use_fusion_correlation:
            scales = self._corr_scales(
                input_ids=input_ids,
                acoustic=acoustic,
                visual=visual,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
            )
            if scales is not None:
                scale_t, scale_a, scale_v = scales
                h[0] = h[0] * scale_t.view(-1, 1)
                h[1] = h[1] * scale_t.view(-1, 1)
                h[2] = h[2] * scale_a.view(-1, 1)
                h[3] = h[3] * scale_a.view(-1, 1)
                h[4] = h[4] * scale_v.view(-1, 1)
                h[5] = h[5] * scale_v.view(-1, 1)
        attn_weights_cs, attn_cs = self.attn_cs(h[0], h[2], h[4])
        attn_weights_fs, attn_fs = self.attn_fs(h[1], h[3], h[5])
        fea_cfs = self.fc_all(torch.cat((attn_cs,attn_fs),dim=1))
        
        fea_modality_t = self.fc_ts(torch.cat((h[0], h[1]), dim=1))
        fea_modality_a = self.fc_as(torch.cat((h[2], h[3]), dim=1))
        fea_modality_v = self.fc_vs(torch.cat((h[4], h[5]), dim=1))
        feas_cf_all = torch.stack([fea_modality_t, fea_modality_a, fea_modality_v], dim=2) # [bsz, 768, 2]
        feas_cf_all = torch.matmul(feas_cf_all, attn_weights_cs) 
        feas_cf_all = feas_cf_all.squeeze() # [bsz, 64]
        feas_cf_all = feas_cf_all.view(-1,fea_modality_t.shape[1])
        
        outputs = torch.cat((fea_cfs, feas_cf_all), dim=1)
        logits = self.classifier(outputs)
        
        moes = moe_loss_ts + moe_loss_as + moe_loss_vs
        all_losses = moes
        
        return all_losses, logits, outputs


    def test(self,
        input_ids,
        visual,
        acoustic,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,):

        coarsed_t, finegrained_t = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,)

        coarsed_a, finegrained_a = self.audio_network(acoustic)
        coarsed_v, finegrained_v = self.video_network(visual)
        moe_scales = None
        if self.use_moe_reliability:
            moe_scales = self._corr_scales(
                input_ids=input_ids,
                acoustic=acoustic,
                visual=visual,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
            )
        if moe_scales is not None:
            r_t, r_a, r_v = moe_scales
        else:
            r_t = r_a = r_v = None
        fine_ts, moe_loss_ts = self.moe_t(torch.flatten(finegrained_t, start_dim=1), reliability=r_t)
        fine_as, moe_loss_as = self.moe_a(torch.flatten(finegrained_a, start_dim=1), reliability=r_a)
        fine_vs, moe_loss_vs = self.moe_v(torch.flatten(finegrained_v, start_dim=1), reliability=r_v)

        all_feas = torch.stack((coarsed_t, fine_ts, coarsed_a, fine_as, coarsed_v, fine_vs), dim=0)
        h = self.transformer_encoder(all_feas)
        if self.use_fusion_correlation:
            scales = self._corr_scales(
                input_ids=input_ids,
                acoustic=acoustic,
                visual=visual,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
            )
            if scales is not None:
                scale_t, scale_a, scale_v = scales
                h[0] = h[0] * scale_t.view(-1, 1)
                h[1] = h[1] * scale_t.view(-1, 1)
                h[2] = h[2] * scale_a.view(-1, 1)
                h[3] = h[3] * scale_a.view(-1, 1)
                h[4] = h[4] * scale_v.view(-1, 1)
                h[5] = h[5] * scale_v.view(-1, 1)
        
        attn_weights_cs, attn_cs = self.attn_cs(h[0], h[2], h[4])
        attn_weights_fs, attn_fs = self.attn_fs(h[1], h[3], h[5])
        fea_cfs = self.fc_all(torch.cat((attn_cs,attn_fs),dim=1))
        
        fea_modality_t = self.fc_ts(torch.cat((h[0], h[1]), dim=1))
        fea_modality_a = self.fc_as(torch.cat((h[2], h[3]), dim=1))
        fea_modality_v = self.fc_vs(torch.cat((h[4], h[5]), dim=1))
        feas_cf_all = torch.stack([fea_modality_t, fea_modality_a, fea_modality_v], dim=2) # [bsz, 768, 2]
        feas_cf_all = torch.matmul(feas_cf_all, attn_weights_cs) 
        
        feas_cf_all = feas_cf_all.squeeze() # [bsz, 64]
        feas_cf_all = feas_cf_all.view(-1,fea_modality_t.shape[1])
        outputs = torch.cat((fea_cfs, feas_cf_all), dim=1)
        logits = self.classifier(outputs)
        
        return logits, outputs




