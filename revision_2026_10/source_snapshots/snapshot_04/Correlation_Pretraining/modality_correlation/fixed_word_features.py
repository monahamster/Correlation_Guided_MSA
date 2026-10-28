"""Deterministic word-level inputs for the frozen correlation branch.

This module is deliberately independent of the downstream BERT path.  It
uses the raw word-level acoustic/visual arrays in the released PKL, and a
separate frozen local BERT to build contextual word embeddings.  Both
correlation pretraining and downstream guidance consume this exact cache.
"""
import hashlib
import json
import os
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import BertModel, BertTokenizer

PIPELINE_VERSION = "fixed_contextual_word_v1"


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sample_id(meta):
    if isinstance(meta, np.ndarray):
        meta = meta.tolist()
    return "_".join(map(str, meta)) if isinstance(meta, (tuple, list)) else str(meta)


def _tokenize_words(words, tokenizer, max_seq_length):
    """Tokenize and truncate exactly at wordpiece level, retaining word map."""
    pieces, owners = [], []
    limit = max_seq_length - 2
    for wi, word in enumerate(words):
        sub = tokenizer.tokenize(word) or [tokenizer.unk_token]
        remaining = limit - len(pieces)
        if remaining <= 0:
            break
        sub = sub[:remaining]
        pieces.extend(sub)
        owners.extend([wi] * len(sub))
    ids = tokenizer.convert_tokens_to_ids([tokenizer.cls_token] + pieces + [tokenizer.sep_token])
    return ids, owners


def cache_path(cache_root, dataset, split):
    return Path(cache_root) / dataset / (split + "." + PIPELINE_VERSION + ".pt")


def build_or_load_cache(pkl_path, dataset, split, bert_path, cache_root, max_seq_length,
                        batch_size=16, device=None):
    """Return a versioned cache; rebuild on metadata mismatch rather than reuse."""
    out = cache_path(cache_root, dataset, split)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "pipeline_version": PIPELINE_VERSION, "dataset": dataset, "split": split,
        "pkl_path": os.path.abspath(pkl_path), "pkl_sha256": _sha256(pkl_path),
        "bert_path": os.path.abspath(bert_path), "bert_config_sha256": _sha256(os.path.join(bert_path, "config.json")),
        "tokenizer": "BertTokenizer", "max_seq_length": int(max_seq_length),
        "text_layer": "last_hidden_state", "pooling": "mean_subwords", "special_tokens": "[CLS],[SEP] excluded",
    }
    if out.exists():
        cached = torch.load(out, map_location="cpu")
        if cached.get("metadata") == meta:
            return cached, out, False

    with open(pkl_path, "rb") as f:
        samples = pickle.load(f)[split]
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertModel.from_pretrained(bert_path).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    records = []
    for start in range(0, len(samples), batch_size):
        chunk = samples[start:start + batch_size]
        tokenized = [_tokenize_words(x[0][0], tokenizer, max_seq_length) for x in chunk]
        longest = max(len(ids) for ids, _ in tokenized)
        ids = torch.full((len(chunk), longest), tokenizer.pad_token_id, dtype=torch.long, device=device)
        attn = torch.zeros_like(ids)
        for i, (row, _) in enumerate(tokenized):
            ids[i, :len(row)] = torch.tensor(row, device=device)
            attn[i, :len(row)] = 1
        with torch.no_grad():
            hidden = model(input_ids=ids, attention_mask=attn).last_hidden_state.cpu()
        for i, ((words, visual, acoustic), _label, seg) in enumerate(chunk):
            _ids, owners = tokenized[i]
            kept_words = []
            for wi in owners:
                if not kept_words or kept_words[-1] != wi:
                    kept_words.append(wi)
            text_rows = []
            for wi in kept_words:
                positions = [j + 1 for j, owner in enumerate(owners) if owner == wi]
                text_rows.append(hidden[i, positions].mean(0))
            if not text_rows:
                raise ValueError("empty correlation sequence for sample {}".format(sample_id(seg)))
            records.append({
                "sample_id": sample_id(seg), "text": torch.stack(text_rows).float(),
                "audio": torch.as_tensor(np.asarray(acoustic)[kept_words], dtype=torch.float32),
                "visual": torch.as_tensor(np.asarray(visual)[kept_words], dtype=torch.float32),
                "length": len(kept_words), "kept_word_indices": kept_words,
            })
    cache = {"metadata": meta, "records": records}
    torch.save(cache, out)
    (out.with_suffix(".json")).write_text(json.dumps({**meta, "n_samples": len(records)}, indent=2))
    return cache, out, True


def collate_word_records(records):
    max_len = max(r["length"] for r in records)
    def pad(name):
        dim = records[0][name].shape[-1]
        value = torch.zeros(len(records), max_len, dim, dtype=torch.float32)
        for i, r in enumerate(records): value[i, :r["length"]] = r[name]
        return value
    mask = torch.zeros(len(records), max_len, dtype=torch.bool)
    for i, r in enumerate(records): mask[i, :r["length"]] = True
    return {"text": pad("text"), "audio": pad("audio"), "visual": pad("visual"),
            "mask": mask, "sample_id": [r["sample_id"] for r in records]}


class CachedTripletDataset(Dataset):
    """Positive aligned record plus deterministic cross-sample negatives."""
    def __init__(self, cache, negative_offset=1):
        self.records = cache["records"]; self.offset = int(negative_offset)
    def __len__(self): return len(self.records)
    def __getitem__(self, idx):
        return self.records[idx], self.records[(idx + self.offset) % len(self.records)]


def collate_triplets(batch):
    pos, neg = zip(*batch)
    return collate_word_records(pos), collate_word_records(neg)
