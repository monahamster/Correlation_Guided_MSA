import os
import pickle
import random
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import BertModel, BertTokenizer


def _load_glomo_pkl(pkl_path: str):
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def _words_to_bert_embeddings(
    words: List[str],
    tokenizer: BertTokenizer,
    model: BertModel,
    device: torch.device,
):
    tokens = []
    word_to_subword = []
    for word in words:
        sub = tokenizer.tokenize(word)
        if not sub:
            sub = [tokenizer.unk_token]
        start = len(tokens)
        tokens.extend(sub)
        end = len(tokens)
        word_to_subword.append((start, end))

    tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state.squeeze(0)

    # Shift by 1 to account for CLS.
    word_embeddings = []
    for start, end in word_to_subword:
        sub_emb = hidden[start + 1 : end + 1]
        if sub_emb.numel() == 0:
            sub_emb = hidden[0:1]
        word_embeddings.append(sub_emb.mean(dim=0))

    return torch.stack(word_embeddings, dim=0)


class GLoMoPKLDataset(Dataset):
    def __init__(
        self,
        pkl_path: str,
        split: str,
        bert_path: str,
        cache_dir: str = None,
        build_cache: bool = False,
        for_correlation: bool = False,
        perturbation_ratio: float = 0.0,
        noise_std: float = 0.05,
        strategy_weights: List[float] = None,
    ):
        self.data = _load_glomo_pkl(pkl_path)
        self.samples = self.data[split]
        self.for_correlation = for_correlation
        self.perturbation_ratio = perturbation_ratio
        self.noise_std = noise_std
        self.strategy_weights = strategy_weights or [1 / 3, 1 / 3, 1 / 3]
        self.strategies = ["A", "B", "C"]

        self.bert_path = bert_path
        self.cache_dir = cache_dir
        self.cache_path = None
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            self.cache_path = os.path.join(cache_dir, f"{os.path.basename(pkl_path)}.{split}.text.pt")

        self.text_cache = None
        if self.cache_path and os.path.exists(self.cache_path):
            self.text_cache = torch.load(self.cache_path, map_location="cpu")
        elif build_cache:
            self.text_cache = self._build_text_cache()
            torch.save(self.text_cache, self.cache_path)

    def _build_text_cache(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        model = BertModel.from_pretrained(self.bert_path)
        model.to(device)
        model.eval()

        cache = []
        for (words, _visual, _acoustic), _label, _seg in self.samples:
            emb = _words_to_bert_embeddings(words, tokenizer, model, device)
            cache.append(emb.cpu())
        return cache

    def __len__(self):
        return len(self.samples)

    def _get_text(self, index, words):
        if self.text_cache is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            tokenizer = BertTokenizer.from_pretrained(self.bert_path)
            model = BertModel.from_pretrained(self.bert_path)
            model.to(device)
            model.eval()
            emb = _words_to_bert_embeddings(words, tokenizer, model, device).cpu()
            return emb
        return self.text_cache[index]

    def apply_perturbation(self, index, text, audio, vision, label):
        chosen_strategy = random.choices(self.strategies, weights=self.strategy_weights, k=1)[0]
        text_neg = text.clone()
        audio_neg = audio.clone()
        vision_neg = vision.clone()
        final_label = label.clone()

        if chosen_strategy == "A":
            chosen_modality = random.choice(["T", "A", "V"])
            rand_idx = random.randint(0, len(self.samples) - 1)
            while rand_idx == index:
                rand_idx = random.randint(0, len(self.samples) - 1)
            (words_r, visual_r, acoustic_r), label_r, _seg_r = self.samples[rand_idx]
            other_text = self._get_text(rand_idx, words_r)
            other_audio = torch.tensor(acoustic_r, dtype=torch.float)
            other_vision = torch.tensor(visual_r, dtype=torch.float)
            if chosen_modality == "T":
                text_neg = other_text.clone()
            elif chosen_modality == "A":
                audio_neg = other_audio.clone()
            else:
                vision_neg = other_vision.clone()
            final_label = 0.5 * (label + torch.tensor(label_r, dtype=torch.float))
        elif chosen_strategy == "B":
            text_neg = self.shift_sequence(text_neg)
            audio_neg = self.shift_sequence(audio_neg)
            vision_neg = self.shift_sequence(vision_neg)
        else:
            audio_neg = audio_neg + torch.randn_like(audio_neg) * self.noise_std
            vision_neg = vision_neg + torch.randn_like(vision_neg) * self.noise_std
            if text_neg.size(0) > 0:
                idx_word = random.randint(0, text_neg.size(0) - 1)
                text_neg[idx_word] = torch.randn_like(text_neg[idx_word]) * self.noise_std

        return text_neg, audio_neg, vision_neg, final_label

    @staticmethod
    def shift_sequence(seq):
        if seq.size(0) > 1:
            shifted = torch.zeros_like(seq)
            shifted[:-1] = seq[1:]
            return shifted
        return seq

    def generate_negative_sample(self, index, text_pos, audio_pos, vision_pos):
        text_neg = text_pos.clone()
        audio_neg = audio_pos.clone()
        vision_neg = vision_pos.clone()

        strategies = ["A", "B", "C"]
        chosen_strategy = random.choice(strategies) if random.random() < 0.75 else "D"
        if chosen_strategy == "D":
            chosen_strategy = random.choice(strategies)

        if chosen_strategy == "A":
            chosen_modality = random.choice(["T", "A", "V"])
            rand_idx = random.randint(0, len(self.samples) - 1)
            while rand_idx == index:
                rand_idx = random.randint(0, len(self.samples) - 1)
            (words_r, visual_r, acoustic_r), _label_r, _seg_r = self.samples[rand_idx]
            other_text = self._get_text(rand_idx, words_r)
            other_audio = torch.tensor(acoustic_r, dtype=torch.float)
            other_vision = torch.tensor(visual_r, dtype=torch.float)
            if chosen_modality == "T":
                text_neg = other_text.clone()
            elif chosen_modality == "A":
                audio_neg = other_audio.clone()
            else:
                vision_neg = other_vision.clone()
        elif chosen_strategy == "B":
            text_neg = self.shift_sequence(text_neg)
            audio_neg = self.shift_sequence(audio_neg)
            vision_neg = self.shift_sequence(vision_neg)
        else:
            audio_neg = audio_neg + torch.randn_like(audio_neg) * self.noise_std
            vision_neg = vision_neg + torch.randn_like(vision_neg) * self.noise_std
            if text_neg.size(0) > 0:
                rand_word_idx = random.randint(0, text_neg.size(0) - 1)
                text_neg[rand_word_idx] = torch.randn_like(text_neg[rand_word_idx]) * self.noise_std

        return text_neg, audio_neg, vision_neg

    def __getitem__(self, index):
        (words, visual, acoustic), label, seg = self.samples[index]
        text = self._get_text(index, words)
        audio = torch.tensor(acoustic, dtype=torch.float)
        vision = torch.tensor(visual, dtype=torch.float)
        label = torch.tensor(label, dtype=torch.float)

        if random.random() < self.perturbation_ratio:
            text, audio, vision, label = self.apply_perturbation(index, text, audio, vision, label)

        if not self.for_correlation:
            return ((seg, text, audio, vision), label, (seg,))

        text_pos = text.clone()
        audio_pos = audio.clone()
        vision_pos = vision.clone()
        text_neg, audio_neg, vision_neg = self.generate_negative_sample(index, text_pos, audio_pos, vision_pos)

        return (
            (seg, text_pos, audio_pos, vision_pos),
            (text_neg, audio_neg, vision_neg),
            label,
            (seg,),
        )
