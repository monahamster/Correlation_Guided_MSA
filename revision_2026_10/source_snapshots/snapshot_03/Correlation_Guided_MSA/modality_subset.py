"""Retrained modality subsets with the original parameter layout and alignment."""
import torch
import torch.nn.functional as F

NAMES = {'t': 'text', 'a': 'audio', 'v': 'vision'}

def subset_signals(model, corr_text, corr_audio, corr_visual, corr_mask):
    active = model.modalities
    if len(active) == 1:
        return None
    if len(active) != 2 or model.corr_model is None:
        raise ValueError('Subset guidance requires two modalities and a frozen encoder')
    inputs = dict(t=corr_text, a=corr_audio, v=corr_visual)
    if corr_mask is None or any(inputs[m] is None for m in active):
        raise ValueError('Available modalities require aligned correlation inputs')
    valid = corr_mask.bool()
    if not valid.any(1).all():
        raise ValueError('Each sample must contain valid aligned words')
    encoded = {}
    branch = model.corr_model
    with torch.no_grad():
        for m in active:
            name = NAMES[m]
            if inputs[m].shape[:2] != valid.shape:
                raise ValueError('Correlation input/mask shape mismatch')
            z = branch.pos_encoder(getattr(branch, name + '_fc')(inputs[m]))
            z = getattr(branch, name + '_encoder')(z, src_key_padding_mask=~valid)
            encoded[m] = getattr(branch, name + '_out_fc')(z)
        local = F.cosine_similarity(encoded[active[0]], encoded[active[1]], dim=-1).clamp(-1, 1)
        raw = local.masked_fill(~valid, 0).sum(1) / valid.sum(1)
        c = (raw + 1) / 2
        scale = 1 + model.corr_alpha * (c - .5)
    result = {'valid_mask': valid}
    for pair in ['ta', 'tv', 'av']:
        result['raw_cosine_' + pair] = raw if pair == active else None
        result['c_' + pair] = c if pair == active else None
    for m in 'tav':
        result['q_' + m] = c if m in active else None
        result['s_' + m] = scale if m in active else None
    return result


def masked_attention(module, features, active):
    hidden = module.attention_mlp(torch.cat(features, dim=1))
    logits = module.fc_att(hidden).unsqueeze(2) * module.scale
    available = torch.tensor([m in active for m in 'tav'], device=logits.device, dtype=torch.bool)
    weights = logits.masked_fill(~available[None, :, None], float('-inf')).softmax(dim=1)
    fused = torch.matmul(torch.stack(features, dim=2), weights).view(-1, module.dim)
    return weights, fused


def subset_forward(model, input_ids, visual, acoustic, attention_mask=None,
                   token_type_ids=None, position_ids=None, head_mask=None,
                   inputs_embeds=None, output_attentions=None, output_hidden_states=None,
                   corr_text=None, corr_audio=None, corr_visual=None, corr_mask=None):
    active = model.modalities
    if active == 'tav':
        raise ValueError('Full model must use its original forward path')
    corr = subset_signals(model, corr_text, corr_audio, corr_visual, corr_mask)
    batch = input_ids.shape[0]
    zero = model.regressor.weight.new_zeros((batch, model.d_l))
    coarse, fine, losses = {}, {}, []
    for m in active:
        if m == 't':
            co, fi = model.bert(input_ids, attention_mask=attention_mask,
                token_type_ids=token_type_ids, position_ids=position_ids,
                head_mask=head_mask, inputs_embeds=inputs_embeds,
                output_attentions=output_attentions, output_hidden_states=output_hidden_states)
        else:
            network = model.audio_network if m == 'a' else model.video_network
            co, fi = network(acoustic if m == 'a' else visual)
        fi = torch.nan_to_num(fi, nan=0.0, posinf=0.0, neginf=0.0)
        reliability = corr['s_' + m] if corr else None
        fi, loss = getattr(model, 'moe_' + m)(fi.flatten(start_dim=1), reliability=reliability)
        coarse[m], fine[m] = co, fi
        losses.append(loss)
    tokens = torch.stack([x for m in active for x in (coarse[m], fine[m])], dim=0)
    joint = model.transformer_encoder(tokens)
    coarse, fine = {}, {}
    for i, m in enumerate(active):
        scale = corr['s_' + m].view(-1, 1) if corr else 1.0
        coarse[m], fine[m] = joint[2*i] * scale, joint[2*i+1] * scale
    wc, fc = masked_attention(model.attn_cs, [coarse.get(m, zero) for m in 'tav'], active)
    wf, ff = masked_attention(model.attn_fs, [fine.get(m, zero) for m in 'tav'], active)
    cfs = model.fc_all(torch.cat((fc, ff), dim=1))
    per_modality = [getattr(model, 'fc_' + m + 's')(torch.cat((coarse[m], fine[m]), dim=1)) if m in active else zero for m in 'tav']
    cf_all = torch.matmul(torch.stack(per_modality, dim=2), wc).view(batch, model.d_l)
    outputs = torch.cat((cfs, cf_all), dim=1)
    reg, cls = model.regressor(outputs), model.classifier(outputs)
    analysis = dict(corr or {})
    analysis.update(repr=outputs, reg_logits=reg, cls_logits=cls,
                    attention_coarse=wc, attention_fine=wf, modalities=active)
    return sum(losses), reg, cls, outputs, analysis
