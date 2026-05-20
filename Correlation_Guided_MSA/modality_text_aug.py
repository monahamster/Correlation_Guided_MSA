from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np


@dataclass
class ModalityTextAugmentor:
    audio_thresholds: Dict[str, Tuple[float, float]]
    visual_thresholds: Dict[str, Tuple[float, float]]
    audio_means: Dict[str, float]
    audio_stds: Dict[str, float]
    visual_means: Dict[str, float]
    visual_stds: Dict[str, float]
    use_audio_desc: bool = True
    use_visual_desc: bool = True
    visual_desc_version: str = "v1"

    @staticmethod
    def _safe_array(x):
        arr = np.asarray(x, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D modality array, got shape {arr.shape}")
        return arr

    @staticmethod
    def _compute_audio_stats(acoustic: np.ndarray) -> Dict[str, float]:
        acoustic = ModalityTextAugmentor._safe_array(acoustic)
        energy = float(np.mean(np.linalg.norm(acoustic, axis=1)))
        variation = float(np.mean(np.std(acoustic, axis=0)))
        dynamics = float(np.mean(np.linalg.norm(np.diff(acoustic, axis=0), axis=1))) if acoustic.shape[0] > 1 else 0.0
        stability = 1.0 / (1.0 + dynamics)
        return {
            "energy": energy,
            "pitch_variation": variation,
            "vocal_dynamics": dynamics,
            "stability": stability,
        }

    @staticmethod
    def _compute_visual_stats(visual: np.ndarray) -> Dict[str, float]:
        visual = ModalityTextAugmentor._safe_array(visual)
        motion = float(np.mean(np.linalg.norm(visual, axis=1)))
        variation = float(np.mean(np.std(visual, axis=0)))
        tension = float(np.mean(np.linalg.norm(np.diff(visual, axis=0), axis=1))) if visual.shape[0] > 1 else 0.0
        return {
            "facial_motion": motion,
            "expression_variation": variation,
            "visual_tension": tension,
        }

    @staticmethod
    def _quantile_bounds(values: Iterable[float]) -> Tuple[float, float]:
        values = np.asarray(list(values), dtype=np.float32)
        return float(np.quantile(values, 1.0 / 3.0)), float(np.quantile(values, 2.0 / 3.0))

    @classmethod
    def from_training_examples(
        cls,
        train_examples,
        use_audio_desc: bool = True,
        use_visual_desc: bool = True,
        visual_desc_version: str = "v1",
    ) -> "ModalityTextAugmentor":
        audio_stats = {"energy": [], "pitch_variation": [], "vocal_dynamics": [], "stability": []}
        visual_stats = {"facial_motion": [], "expression_variation": [], "visual_tension": []}

        for (words, visual, acoustic), _, _ in train_examples:
            a_stats = cls._compute_audio_stats(acoustic)
            v_stats = cls._compute_visual_stats(visual)
            for k, v in a_stats.items():
                audio_stats[k].append(v)
            for k, v in v_stats.items():
                visual_stats[k].append(v)

        audio_thresholds = {k: cls._quantile_bounds(vs) for k, vs in audio_stats.items()}
        audio_means = {k: float(np.mean(vs)) for k, vs in audio_stats.items()}
        audio_stds = {k: float(np.std(vs) + 1e-6) for k, vs in audio_stats.items()}
        visual_thresholds = {k: cls._quantile_bounds(vs) for k, vs in visual_stats.items()}
        visual_means = {k: float(np.mean(vs)) for k, vs in visual_stats.items()}
        visual_stds = {k: float(np.std(vs) + 1e-6) for k, vs in visual_stats.items()}
        return cls(
            audio_thresholds=audio_thresholds,
            visual_thresholds=visual_thresholds,
            audio_means=audio_means,
            audio_stds=audio_stds,
            visual_means=visual_means,
            visual_stds=visual_stds,
            use_audio_desc=use_audio_desc,
            use_visual_desc=use_visual_desc,
            visual_desc_version=visual_desc_version,
        )

    @staticmethod
    def _bucketize(value: float, bounds: Tuple[float, float]) -> str:
        low, high = bounds
        if value < low:
            return "low"
        if value > high:
            return "high"
        return "medium"

    def _audio_phrase(self, name: str, bucket: str) -> str:
        phrase_map = {
            "energy": {
                "low": "calm",
                "medium": "moderate in energy",
                "high": "energetic",
            },
            "pitch_variation": {
                "low": "limited pitch variation",
                "medium": "moderate pitch variation",
                "high": "noticeable pitch variation",
            },
            "vocal_dynamics": {
                "low": "slow vocal changes",
                "medium": "moderate vocal dynamics",
                "high": "strong vocal dynamics",
            },
            "stability": {
                "low": "less steady delivery",
                "medium": "moderately steady delivery",
                "high": "steady delivery",
            },
        }
        return phrase_map[name][bucket]

    def _audio_description(self, acoustic: np.ndarray) -> str:
        stats = self._compute_audio_stats(acoustic)
        z_scores = {}
        for name, value in stats.items():
            z_scores[name] = (value - self.audio_means[name]) / self.audio_stds[name]

        salient = sorted(z_scores.items(), key=lambda kv: abs(kv[1]), reverse=True)
        # Suppress audio description for acoustically neutral samples.
        if not salient or abs(salient[0][1]) < 0.6:
            return ""

        selected = [salient[0][0]]
        if len(salient) > 1 and abs(salient[1][1]) >= 0.6:
            selected.append(salient[1][0])

        phrases = []
        for name in selected:
            bucket = self._bucketize(stats[name], self.audio_thresholds[name])
            phrases.append(self._audio_phrase(name, bucket))

        if len(phrases) == 1:
            return f"[AUDIO] The speaker sounds {phrases[0]}."
        return f"[AUDIO] The speaker sounds {phrases[0]} with {phrases[1]}."

    def _visual_phrase(self, name: str, bucket: str) -> str:
        phrase_map = {
            "facial_motion": {
                "low": "subtle facial movement",
                "medium": "moderate facial movement",
                "high": "noticeable facial movement",
            },
            "expression_variation": {
                "low": "limited expression variation",
                "medium": "moderate expression variation",
                "high": "strong expression variation",
            },
            "visual_tension": {
                "low": "relaxed visual cues",
                "medium": "moderate visual tension",
                "high": "tense visual cues",
            },
        }
        return phrase_map[name][bucket]

    def _visual_description_v1(self, visual: np.ndarray) -> str:
        stats = self._compute_visual_stats(visual)
        motion = self._bucketize(stats["facial_motion"], self.visual_thresholds["facial_motion"])
        expr = self._bucketize(stats["expression_variation"], self.visual_thresholds["expression_variation"])
        tension = self._bucketize(stats["visual_tension"], self.visual_thresholds["visual_tension"])
        return (
            f"[VISUAL] facial motion {motion} expression variation {expr} "
            f"visual tension {tension}"
        )

    def _visual_description_v2(self, visual: np.ndarray) -> str:
        stats = self._compute_visual_stats(visual)
        z_scores = {}
        for name, value in stats.items():
            z_scores[name] = (value - self.visual_means[name]) / self.visual_stds[name]

        salient = sorted(z_scores.items(), key=lambda kv: abs(kv[1]), reverse=True)
        # Suppress visual description for visually neutral samples.
        if not salient or abs(salient[0][1]) < 0.6:
            return ""

        selected = [salient[0][0]]
        if len(salient) > 1 and abs(salient[1][1]) >= 0.6:
            selected.append(salient[1][0])

        phrases = []
        for name in selected:
            bucket = self._bucketize(stats[name], self.visual_thresholds[name])
            phrases.append(self._visual_phrase(name, bucket))

        if len(phrases) == 1:
            return f"[VISUAL] The speaker shows {phrases[0]}."
        return f"[VISUAL] The speaker shows {phrases[0]} with {phrases[1]}."

    def _visual_description(self, visual: np.ndarray) -> str:
        if self.visual_desc_version == "v2":
            return self._visual_description_v2(visual)
        return self._visual_description_v1(visual)

    def augment_example(self, example):
        (words, visual, acoustic), label, meta = example
        words = list(words)
        visual = self._safe_array(visual)
        acoustic = self._safe_array(acoustic)

        desc_tokens: List[str] = []
        if self.use_audio_desc:
            desc_tokens.extend(self._audio_description(acoustic).split())
        if self.use_visual_desc:
            desc_tokens.extend(self._visual_description(visual).split())

        if not desc_tokens:
            return example

        visual_pad = np.zeros((len(desc_tokens), visual.shape[1]), dtype=visual.dtype)
        acoustic_pad = np.zeros((len(desc_tokens), acoustic.shape[1]), dtype=acoustic.dtype)
        aug_words = words + desc_tokens
        aug_visual = np.concatenate([visual, visual_pad], axis=0)
        aug_acoustic = np.concatenate([acoustic, acoustic_pad], axis=0)
        return ((aug_words, aug_visual, aug_acoustic), label, meta)

    def augment_split(self, examples):
        return [self.augment_example(example) for example in examples]
