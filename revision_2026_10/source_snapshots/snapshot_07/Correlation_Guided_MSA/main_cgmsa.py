from __future__ import absolute_import, division, print_function

import argparse
import math
import platform
from formal_protocol import (VERSION, preserve_rng, seeded_loader, strict_cache,
    backbone_digest, tensor_digest, verify_inputs, verify_source, file_sha256, verify_environment)
import csv
import json
import os
import random
import pickle
import shlex
import sys
import numpy as np
from typing import *
import time
from pathlib import Path
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score

# Honor external CUDA_VISIBLE_DEVICES, default to GPU 7 if unset
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from torch.nn import CrossEntropyLoss, L1Loss, MSELoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef
from transformers import BertTokenizer, XLNetTokenizer, get_cosine_schedule_with_warmup, BertConfig
from transformers.optimization import AdamW

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}, visible GPUs: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
if torch.cuda.is_available():
    try:
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
    except Exception:
        pass


from correlation_guided_msa_model import GLoMo
from modality_text_aug import ModalityTextAugmentor

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str,
                    choices=["mosi", "mosei"], default="mosi")
parser.add_argument("--max_seq_length", type=int, default=60)
parser.add_argument("--train_batch_size", type=int, default=64)
parser.add_argument("--dev_batch_size", type=int, default=128)
parser.add_argument("--test_batch_size", type=int, default=128)
parser.add_argument("--n_epochs", type=int, default=100)
parser.add_argument("--dropout_prob", type=float, default=0.3)
parser.add_argument(
    "--model",
    type=str,
    choices=["bert-base-uncased", "T5-base", "CoCo-LM"],
    default="bert-base-uncased",
)
parser.add_argument("--learning_rate", type=float, default=None)
parser.add_argument("--gradient_accumulation_step", type=int, default=1)
parser.add_argument("--d_l", type=int, default=None)
parser.add_argument("--seed", type=int, default=5576)
parser.add_argument("--gran_t", type=int, default=3)
parser.add_argument("--gran_a", type=int, default=3)
parser.add_argument("--gran_v", type=int, default=3)
parser.add_argument("--TEXT_DIM", type=int, default=768)
parser.add_argument("--ACOUSTIC_DIM", type=int, default=74)
parser.add_argument("--VISUAL_DIM", type=int, default=35)
parser.add_argument("--experts_t", type=int, default=3)
parser.add_argument("--experts_a", type=int, default=3)
parser.add_argument("--experts_v", type=int, default=3)
parser.add_argument("--experts_all", type=int, default=2)
parser.add_argument("--k", type=int, default=2)
parser.add_argument("--k_all", type=int, default=2)
parser.add_argument("--attn_dropout", type=float, default=0.5)
parser.add_argument("--num_heads", type=int, default=16) 
parser.add_argument("--relu_dropout", type=float, default=0.3)
parser.add_argument("--res_dropout", type=float, default=0.3)
parser.add_argument("--embed_dropout", type=float, default=0.2)
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--schedule', default=[80, 100], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument("--layers", type=int, default=2)
parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
parser.add_argument("--load", type=int, default=0)
parser.add_argument("--paired_init_save", type=str, default=None)
parser.add_argument("--paired_init_load", type=str, default=None)
parser.add_argument("--test", type=int, default=0)
parser.add_argument("--model_path", type=str, default='correlation_guided_msa.pth')
parser.add_argument(
    "--test_audio_noise_std",
    type=float,
    default=0.0,
    help=(
        "Test-time acoustic corruption strength. Gaussian noise is scaled by "
        "each segment's acoustic-feature standard deviation."
    ),
)
parser.add_argument(
    "--test_visual_mask_ratio",
    type=float,
    default=0.0,
    help="Test-time probability of masking each non-padding visual frame.",
)
parser.add_argument(
    "--test_corruption_seed",
    type=int,
    default=20260831,
    help="Fixed RNG seed used only for deterministic test-time corruptions.",
)
parser.add_argument(
    "--export_analysis",
    action="store_true",
    help="Export per-example representations during --test; disabled by default to limit disk use.",
)
parser.add_argument(
    "--profile_inference",
    action="store_true",
    help="Profile single-sample model-forward latency and peak GPU memory during --test.",
)
parser.add_argument("--profile_warmup", type=int, default=20)
parser.add_argument("--profile_repeats", type=int, default=100)
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')
parser.add_argument("--cls_task", type=str, choices=["binary", "seven"], default="seven")
parser.add_argument("--reg_loss_weight", type=float, default=1.0)
parser.add_argument("--cls_loss_weight", type=float, default=1.0)
parser.add_argument("--use_correlation", action="store_true")
parser.add_argument("--use_fusion_correlation", action="store_true")
parser.add_argument("--corr_model_path", type=str, default=None)
parser.add_argument("--corr_cache_root", type=str, default=None,
                    help="Versioned fixed word-level cache used only by the frozen correlation branch.")
parser.add_argument(
    "--corr_text_encoder_path",
    type=str,
    default="../BERT_EN/",
    help="Frozen BERT checkpoint used to generate contextual wordpiece features for correlation guidance.",
)
parser.add_argument("--corr_alpha", type=float, default=1.0)
parser.add_argument(
    "--correlation_control",
    choices=["none", "neutral", "shuffle", "constant"],
    default="none",
    help=(
        "Test-time correlation control: neutral removes correlation modulation; "
        "shuffle cyclically assigns each sample another sample's scales."
    ),
)
parser.add_argument("--use_moe_reliability", action="store_true")
parser.add_argument("--moe_reliability_lambda", type=float, default=0.1)
parser.add_argument("--drop_text", action="store_true")
parser.add_argument("--drop_audio", action="store_true")
parser.add_argument("--drop_visual", action="store_true")
parser.add_argument("--analysis_output_dir", type=str, default="analysis/outputs")
parser.add_argument("--analysis_tag", type=str, default="")
parser.add_argument("--experiment_root", type=str, default="../experiments")
parser.add_argument("--experiment_tag", type=str, default="")
parser.add_argument("--log_path", type=str, default="")
parser.add_argument("--use_modality_text_aug", action="store_true")
parser.add_argument("--use_audio_desc", action="store_true")
parser.add_argument("--use_visual_desc", action="store_true")
parser.add_argument("--visual_desc_version", type=str, choices=["v1", "v2"], default="v1")
parser.add_argument(
    "--save_best_by",
    type=str,
    choices=["acc2", "acc2_non_zero", "f1", "f1_non_zero", "corr", "mae", "valid_mse", "valid_mae"],
    default="valid_mae",
    help=(
        "Validation metric used for checkpoint selection. "
        "For the sentiment-regression experiments, use valid_mae to avoid "
        "selecting checkpoints on the test set."
    ),
)
parser.add_argument("--best_model_path", type=str, default="")
parser.add_argument("--modalities", choices=["t","a","v","ta","tv","av","tav"], default="tav")
args = parser.parse_args()

if args.use_correlation and not args.use_fusion_correlation:
    args.use_fusion_correlation = True

def apply_dataset_defaults(args):
    if args.dataset == "mosi":
        if args.learning_rate is None:
            args.learning_rate = 4e-5
        if args.d_l is None:
            args.d_l = 48
    elif args.dataset == "mosei":
        if args.learning_rate is None:
            args.learning_rate = 1e-5
        if args.d_l is None:
            args.d_l = 192
    else:
        if args.learning_rate is None:
            args.learning_rate = 4e-5
        if args.d_l is None:
            args.d_l = 96

    if args.cls_task == "binary":
        args.num_classes = 2
    else:
        args.num_classes = 7

apply_dataset_defaults(args)

ACOUSTIC_DIM = args.ACOUSTIC_DIM
VISUAL_DIM = args.VISUAL_DIM #47 FOR MOSI 35 FOR MOSEI
TEXT_DIM = args.TEXT_DIM

def str2bool(s):
    if isinstance(s, bool):
        return s
    if s.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif s.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError(
            "Boolean value expected. Recieved {0}".format(s)
        )


def seed(s):
    if isinstance(s, int):
        if 0 <= s <= 9999:
            return s
        else:
            raise argparse.ArgumentTypeError(
                "Seed must be between 0 and 2**32 - 1. Received {0}".format(s)
            )
    elif s == "random":
        return random.randint(0, 9999)
    else:
        raise argparse.ArgumentTypeError(
            "Integer value is expected. Recieved {0}".format(s)
        )


def return_unk():
    return 0


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, visual, acoustic, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.visual = visual
        self.acoustic = acoustic
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id



def convert_to_features(examples, max_seq_length, tokenizer):
    features = []

    for (ex_index, example) in enumerate(examples):

        (words, visual, acoustic), label_id, segment = example# 
        tokens, inversions = [], []
        for idx, word in enumerate(words):
            tokenized = tokenizer.tokenize(word)
            tokens.extend(tokenized)
            inversions.extend([idx] * len(tokenized))

        assert len(tokens) == len(inversions)

        aligned_visual = []
        aligned_audio = []

        for inv_idx in inversions:
            aligned_visual.append(visual[inv_idx, :])
            aligned_audio.append(acoustic[inv_idx, :])

        visual = np.array(aligned_visual)
        acoustic = np.array(aligned_audio)

        # Truncate input if necessary
        if len(tokens) > max_seq_length - 2:
            tokens = tokens[: max_seq_length - 2]
            acoustic = acoustic[: max_seq_length - 2]
            visual = visual[: max_seq_length - 2]

        if args.model == "bert-base-uncased":
            prepare_input = prepare_bert_input

        input_ids, visual, acoustic, input_mask, segment_ids = prepare_input(
            tokens, visual, acoustic, tokenizer
        )

        # Check input length
        assert len(input_ids) == args.max_seq_length
        assert len(input_mask) == args.max_seq_length
        assert len(segment_ids) == args.max_seq_length
        assert acoustic.shape[0] == args.max_seq_length
        assert visual.shape[0] == args.max_seq_length

        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                visual=visual,
                acoustic=acoustic,
                label_id=label_id,
            )
        )
    return features


def prepare_bert_input(tokens, visual, acoustic, tokenizer):# include the text or not 
    CLS = tokenizer.cls_token
    SEP = tokenizer.sep_token
    tokens = [CLS] + tokens + [SEP]

    # Pad zero vectors for acoustic / visual vectors to account for [CLS] / [SEP] tokens
    acoustic_zero = np.zeros((1, ACOUSTIC_DIM))
    acoustic = np.concatenate((acoustic_zero, acoustic, acoustic_zero))
    visual_zero = np.zeros((1, VISUAL_DIM))
    visual = np.concatenate((visual_zero, visual, visual_zero))

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    segment_ids = [0] * len(input_ids)
    input_mask = [1] * len(input_ids)

    pad_length = args.max_seq_length - len(input_ids)

    acoustic_padding = np.zeros((pad_length, ACOUSTIC_DIM))
    acoustic = np.concatenate((acoustic, acoustic_padding))

    visual_padding = np.zeros((pad_length, VISUAL_DIM))
    visual = np.concatenate((visual, visual_padding))

    padding = [0] * pad_length

    # Pad inputs
    input_ids += padding
    input_mask += padding
    segment_ids += padding

    return input_ids, visual, acoustic, input_mask, segment_ids




def get_tokenizer(model):
    if model == "bert-base-uncased":
        return BertTokenizer.from_pretrained('../BERT_EN/')
    
    else:
        raise ValueError(
            "Expected 'bert-base-uncased' or 'xlnet-base-cased, but received {}".format(
                model
            )
        )


def get_appropriate_dataset(data, corr_cache=None):

    tokenizer = get_tokenizer(args.model)

    features = convert_to_features(data, args.max_seq_length, tokenizer)
    all_input_ids = torch.tensor(
        [f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor(
        [f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor(
        [f.segment_ids for f in features], dtype=torch.long)
    all_visual = torch.tensor([f.visual for f in features], dtype=torch.float)
    all_acoustic = torch.tensor(
        [f.acoustic for f in features], dtype=torch.float)
    all_label_ids = torch.tensor(
        [f.label_id for f in features], dtype=torch.float)

    tensors = [
        all_input_ids,
        all_visual,
        all_acoustic,
        all_input_mask,
        all_segment_ids,
        all_label_ids]
    if corr_cache is not None:
        records = corr_cache["records"]
        if len(records) != len(features):
            raise ValueError("Correlation cache/sample count mismatch")
        expected_ids = [stringify_sample_id(sample[2]) for sample in data]
        actual_ids = [record["sample_id"] for record in records]
        if actual_ids != expected_ids or len(set(actual_ids)) != len(actual_ids):
            raise ValueError("Correlation cache sample IDs/order mismatch")
        max_words = max(record["length"] for record in records)
        def corr_tensor(name):
            dim = records[0][name].shape[-1]
            out = torch.zeros(len(records), max_words, dim, dtype=torch.float)
            for i, record in enumerate(records): out[i, :record["length"]] = record[name]
            return out
        corr_mask = torch.zeros(len(records), max_words, dtype=torch.bool)
        for i, record in enumerate(records): corr_mask[i, :record["length"]] = True
        tensors.extend([corr_tensor("text"), corr_tensor("audio"), corr_tensor("visual"), corr_mask])
    dataset = TensorDataset(*tensors)
    return dataset


def unpack_batch(batch):
    """Keep the historical six main tensors intact; optional four are correlation-only."""
    input_ids, visual, acoustic, input_mask, segment_ids, label_ids = batch[:6]
    corr = {}
    if len(batch) == 10:
        corr = {"corr_text": batch[6], "corr_audio": batch[7],
                "corr_visual": batch[8], "corr_mask": batch[9]}
    return input_ids, visual, acoustic, input_mask, segment_ids, label_ids, corr


def label_to_binary(value):
    return int(float(value) >= 0.0)


def label_to_seven(value):
    return int(np.clip(np.round(float(value)), -3, 3) + 3)


def prediction_to_binary(pred):
    return int(float(pred) >= 0.0)


def prediction_to_seven(pred):
    return int(np.clip(np.round(float(pred)), -3, 3) + 3)


def stringify_sample_id(meta):
    if isinstance(meta, str):
        return meta
    if isinstance(meta, np.ndarray):
        meta = meta.tolist()
    if isinstance(meta, (list, tuple)):
        return "_".join(str(x) for x in meta)
    return str(meta)


def extract_analysis_record(example):
    (words, _, _), label, meta = example
    label_value = float(np.array(label).reshape(-1)[0])
    text = " ".join(words)
    return {
        "sample_id": stringify_sample_id(meta),
        "text": text,
        "label_reg": label_value,
        "label_2": label_to_binary(label_value),
        "label_7": label_to_seven(label_value),
    }


def resolve_analysis_output_dir():
    return resolve_experiment_dir() / "analysis"


def resolve_experiment_tag():
    tag = args.experiment_tag.strip() or args.analysis_tag.strip()
    if tag:
        return tag
    tag_parts = [args.dataset]
    if args.test:
        tag_parts.append("test")
    elif args.use_fusion_correlation or args.use_moe_reliability:
        tag_parts.append("ours")
    else:
        tag_parts.append("baseline")
    tag_parts.append(f"seed{args.seed}")
    return "_".join(tag_parts)


def resolve_experiment_dir():
    base_dir = Path(__file__).resolve().parent
    return (base_dir / args.experiment_root / resolve_experiment_tag()).resolve()


def resolve_best_model_path():
    if args.best_model_path.strip():
        return Path(args.best_model_path)
    experiment_dir = resolve_experiment_dir()
    experiment_dir.mkdir(parents=True, exist_ok=True)
    return experiment_dir / "best_model.pt"


def resolve_metrics_path():
    experiment_dir = resolve_experiment_dir()
    experiment_dir.mkdir(parents=True, exist_ok=True)
    return experiment_dir / "metrics.json"


def resolve_command_path():
    experiment_dir = resolve_experiment_dir()
    experiment_dir.mkdir(parents=True, exist_ok=True)
    return experiment_dir / "command.sh"


def write_command_script():
    command_path = resolve_command_path()
    cmd = "python " + " ".join(shlex.quote(arg) for arg in sys.argv)
    with command_path.open("w", encoding="utf-8") as f:
        f.write("#!/usr/bin/env bash\n")
        f.write("set -euo pipefail\n")
        f.write(cmd + "\n")
    os.chmod(command_path, 0o755)
    return command_path


def write_metrics_summary(metrics: Dict[str, Any]):
    metrics_path = resolve_metrics_path()
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return metrics_path


def metric_direction(metric_name):
    return "min" if metric_name in {"mae", "valid_mse", "valid_mae"} else "max"


def metric_value(metric_name, metrics):
    mapping = {
        "acc2": metrics["test_acc2"],
        "acc2_non_zero": metrics["test_acc2_non_zero"],
        "f1": metrics["test_f_score"],
        "f1_non_zero": metrics["test_f_score_non_zero"],
        "corr": metrics["test_corr"],
        "mae": metrics["test_mae"],
        "valid_mse": metrics["valid_mse"],
        "valid_mae": metrics["valid_mae"],
    }
    return mapping[metric_name]


def is_better(metric_name, candidate, best_value):
    direction = metric_direction(metric_name)
    if best_value is None:
        return True
    if direction == "min":
        return candidate < best_value
    return candidate > best_value


def set_up_data_loader():
    with open(f"../datasets/{args.dataset}.pkl", "rb") as handle:# 
        data = pickle.load(handle)

    train_data = data["train"]
    dev_data = data["dev"]
    test_data = data["test"]

    if args.use_modality_text_aug:
        use_audio_desc = args.use_audio_desc or (not args.use_audio_desc and not args.use_visual_desc)
        use_visual_desc = args.use_visual_desc or (not args.use_audio_desc and not args.use_visual_desc)
        augmentor = ModalityTextAugmentor.from_training_examples(
            train_data,
            use_audio_desc=use_audio_desc,
            use_visual_desc=use_visual_desc,
            visual_desc_version=args.visual_desc_version,
        )
        train_data = augmentor.augment_split(train_data)
        dev_data = augmentor.augment_split(dev_data)
        test_data = augmentor.augment_split(test_data)
        print(
            "Using modality text augmentation: audio_desc={}, visual_desc={}, visual_desc_version={}".format(
                use_audio_desc, use_visual_desc, args.visual_desc_version
            )
        )

    # Quick label sanity check
    for name, split in (("train", train_data), ("dev", dev_data), ("test", test_data)):
        labels = np.array([x[1] for x in split], dtype=float)
        print(f"{name} labels: n={len(labels)} min={labels.min():.3f} max={labels.max():.3f} "
              f"mean={labels.mean():.3f} std={labels.std():.3f}")

    corr_caches = (None, None, None)
    if args.use_fusion_correlation or args.use_moe_reliability:
        corr_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Correlation_Pretraining", "modality_correlation"))
        if corr_dir not in sys.path: sys.path.append(corr_dir)
        build_or_load_cache = strict_cache
        cache_root = args.corr_cache_root or os.path.join("..", "repair_cache")
        pkl_path = os.path.abspath(os.path.join("..", "datasets", args.dataset + ".pkl"))
        corr_caches = tuple(build_or_load_cache(pkl_path, args.dataset, split, "../BERT_EN/", cache_root,
                                                 args.max_seq_length)[0]
                            for split in ("train", "dev", "test"))
        print("Using fixed shared correlation cache:", cache_root)
    train_dataset = get_appropriate_dataset(train_data, corr_caches[0])
    dev_dataset = get_appropriate_dataset(dev_data, corr_caches[1])
    test_dataset = get_appropriate_dataset(test_data, corr_caches[2])

    num_train_optimization_steps = (
        int(
            len(train_dataset) / args.train_batch_size /
            args.gradient_accumulation_step
        )
        * args.n_epochs
    )

    train_dataloader = seeded_loader(train_dataset, args.train_batch_size, True, args.seed + 10000)
    dev_dataloader = seeded_loader(dev_dataset, args.dev_batch_size, False, args.seed + 20000)
    test_dataloader = seeded_loader(test_dataset, args.test_batch_size, False, args.seed + 30000)

    return (
        train_dataloader,
        dev_dataloader,
        test_dataloader,
        test_data,
        num_train_optimization_steps,
    )


def set_random_seed(seed: int):
    """
    Helper function to seed experiment for reproducibility.
    If -1 is provided as seed, experiment uses random seed from 0~9999

    Args:
        seed (int): integer to be used as seed, use -1 to randomly seed experiment
    """
 
    print("Seed: {}".format(seed))

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prep_for_training(num_train_optimization_steps: int):

    if args.model == "bert-base-uncased":
        config = BertConfig.from_pretrained('../BERT_EN/')
        config.num_labels = 1
        model = GLoMo(config, args=args)
        # load only BERT weights into the text encoder
        state_dict = torch.load(os.path.join('../BERT_EN/', 'pytorch_model.bin'), map_location='cpu')
        remapped = {}
        for k, v in state_dict.items():
            if k.startswith("bert."):
                k = k[len("bert."):]
            k = k.replace("LayerNorm.gamma", "LayerNorm.weight")
            k = k.replace("LayerNorm.beta", "LayerNorm.bias")
            remapped[k] = v
        missing, unexpected = model.bert.load_state_dict(remapped, strict=False)
        print(f"BERT load: missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print("BERT missing keys (first 5):", missing[:5])
        if unexpected:
            print("BERT unexpected keys (first 5):", unexpected[:5])
        # reinitialize custom modules that are not in the BERT checkpoint
        if hasattr(model, "init_custom"):
            model.init_custom()
        # optional sanity print
        try:
            print("conv audio std", float(model.audio_network.projs.weight.std().cpu()))
            print("conv video std", float(model.video_network.projs.weight.std().cpu()))
        except Exception:
            pass

    total_para = 0
    for param in model.parameters():
        total_para += np.prod(param.size())
    print('total parameter for the model: ', total_para)
    
    model.to(DEVICE)
    # Delay the auxiliary branch until all inherited/custom initialization,
    # BERT loading, and device placement are complete.
    if args.use_fusion_correlation or args.use_moe_reliability:
        model.initialize_frozen_correlation_branch()
    if args.paired_init_load:
        model.load_state_dict(torch.load(args.paired_init_load, map_location=DEVICE), strict=True)
    if args.paired_init_save:
        Path(args.paired_init_save).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.paired_init_save)
    if args.load:
        model.load_state_dict(torch.load(args.model_path, map_location=DEVICE), strict=True)

    return model

@preserve_rng
def set_training_scale_means(model, train_dataloader):
    """Compute constant-control scales only from the complete training split."""
    if args.correlation_control != "constant":
        return None
    was_training = model.training
    # The constant values are defined as the mean of the *unmodified* dynamic
    # scales.  Temporarily bypass the constant-control override while they are
    # being accumulated; otherwise the branch would require its own result
    # before it can be computed.
    original_control = model.correlation_control
    model.correlation_control = "none"
    model.eval()
    total = torch.zeros(3, device=DEVICE)
    count = 0
    with torch.no_grad():
        for batch in DataLoader(train_dataloader.dataset, batch_size=args.dev_batch_size, shuffle=False):
            batch = tuple(t.to(DEVICE) for t in batch)
            _, _, _, _, _, labels, corr_inputs = unpack_batch(batch)
            signal = model._correlation_signals(corr_inputs["corr_text"], corr_inputs["corr_audio"], corr_inputs["corr_visual"], corr_inputs["corr_mask"])
            total += torch.stack([signal["s_t"].sum(), signal["s_a"].sum(), signal["s_v"].sum()])
            count += labels.numel()
    model.constant_scales.copy_((total / count).detach())
    model.correlation_control = original_control
    if was_training: model.train()
    print("Training-set constant scales:", model.constant_scales.tolist())
    return model.constant_scales.tolist()
    
def adjust_learning_rate(optimizer, epoch, args):# 
    """Decay the learning rate based on schedule"""
    lr = args.learning_rate
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.n_epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups: # 
        param_group['lr'] = lr

def build_optimizer(model):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)], 'weight_decay': 0.0} # 
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    return optimizer

def train_epoch(model: nn.Module, train_dataloader: DataLoader, optimizer, epoch=None):
    adjust_learning_rate(optimizer, epoch, args)  
 
    model.train()
    tr_loss = 0
    nb_tr_steps = 0


    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        batch = tuple(t.to(DEVICE) for t in batch)
        input_ids, visual, acoustic, input_mask, segment_ids, label_ids, corr_inputs = unpack_batch(batch)
        visual = torch.squeeze(visual, 1)
        acoustic = torch.squeeze(acoustic, 1)
        moe_losses, reg_logits, cls_logits, _ = model(
            input_ids,
            visual,
            acoustic,
            label_ids,
            token_type_ids=segment_ids,
            attention_mask=input_mask,
            labels=None,
            **corr_inputs,
        )

        loss_reg_fct = L1Loss()
        loss_cls_fct = CrossEntropyLoss()
        class_labels = build_class_labels(label_ids, args.cls_task)
        loss_reg = loss_reg_fct(reg_logits.view(-1), label_ids.view(-1))
        loss_cls = loss_cls_fct(cls_logits, class_labels)
        loss_all = args.reg_loss_weight * loss_reg + args.cls_loss_weight * loss_cls + moe_losses
        
        optimizer.zero_grad()
        loss_all.backward()
        optimizer.step()
        



        tr_loss += loss_all.item()
        nb_tr_steps += 1

    return tr_loss / nb_tr_steps


def eval_epoch(model: nn.Module, dev_dataloader: DataLoader):
    model.eval()
    dev_mse = 0
    dev_mae = 0
    total_examples = 0
    with torch.no_grad():
        for step, batch in enumerate(tqdm(dev_dataloader, desc="Iteration")):
            batch = tuple(t.to(DEVICE) for t in batch)

            input_ids, visual, acoustic, input_mask, segment_ids, label_ids, corr_inputs = unpack_batch(batch)
            visual = torch.squeeze(visual, 1)
            acoustic = torch.squeeze(acoustic, 1)
            reg_logits, _, _ = model.test(
                input_ids,
                 visual,
                 acoustic,
                token_type_ids=segment_ids,
                attention_mask=input_mask,
                **corr_inputs,
            )

            errors = reg_logits.view(-1) - label_ids.view(-1)
            dev_mse += torch.sum(errors.square()).item()
            dev_mae += torch.sum(errors.abs()).item()
            total_examples += errors.numel()
    return dev_mse / total_examples, dev_mae / total_examples


def test_epoch(model: nn.Module, test_dataloader: DataLoader):
    model.eval()
    preds = []
    labels = []
    corruption_generator = make_test_corruption_generator()
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            batch = tuple(t.to(DEVICE) for t in batch)

            input_ids, visual, acoustic, input_mask, segment_ids, label_ids, corr_inputs = unpack_batch(batch)
            visual = torch.squeeze(visual, 1)
            acoustic = torch.squeeze(acoustic, 1)
            visual, acoustic = apply_test_time_corruption(
                visual, acoustic, corruption_generator
            )
            reg_logits, _, _ = model.test(
                input_ids,
                 visual,
                 acoustic,
                token_type_ids=segment_ids,
                attention_mask=input_mask,
                labels=None,
                **corr_inputs,
            )

            logits = reg_logits.detach().cpu().numpy()
                 
            label_ids = label_ids.detach().cpu().numpy()
            logits = np.squeeze(logits).tolist()
            label_ids = np.squeeze(label_ids).tolist()
            preds.extend(logits)
            labels.extend(label_ids)

    preds = np.array(preds)
    labels = np.array(labels)

    return preds, labels


def make_test_corruption_generator():
    """Create a reproducible generator so metrics and reliability export use identical noise."""
    if args.test_audio_noise_std <= 0 and args.test_visual_mask_ratio <= 0:
        return None
    generator_device = DEVICE.type if DEVICE.type == "cuda" else "cpu"
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(args.test_corruption_seed)
    return generator


def apply_test_time_corruption(visual, acoustic, generator):
    """Apply isolated, deterministic feature-level corruptions during evaluation only."""
    if generator is None:
        return visual, acoustic

    if args.test_audio_noise_std > 0:
        # Scale noise per utterance, avoiding severity changes caused by raw feature magnitude.
        scale = acoustic.detach().std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        noise = torch.randn(
            acoustic.shape,
            device=acoustic.device,
            dtype=acoustic.dtype,
            generator=generator,
        )
        acoustic = acoustic + args.test_audio_noise_std * scale * noise

    if args.test_visual_mask_ratio > 0:
        valid_frames = visual.detach().abs().sum(dim=-1, keepdim=True) > 0
        mask = torch.rand(
            (visual.size(0), visual.size(1), 1),
            device=visual.device,
            generator=generator,
        ) < args.test_visual_mask_ratio
        visual = visual.masked_fill(mask & valid_frames, 0.0)

    return visual, acoustic


def collect_test_outputs(model: nn.Module, test_dataloader: DataLoader):
    model.eval()
    reg_preds = []
    cls_preds = []
    reprs = []
    signal_keys = ("raw_cosine_ta", "raw_cosine_tv", "raw_cosine_av", "c_ta", "c_tv", "c_av", "q_t", "q_a", "q_v", "s_t", "s_a", "s_v")
    signal_values = {key: [] for key in signal_keys}
    labels = []
    corruption_generator = make_test_corruption_generator()

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Test analysis"):
            batch = tuple(t.to(DEVICE) for t in batch)
            input_ids, visual, acoustic, input_mask, segment_ids, label_ids, corr_inputs = unpack_batch(batch)
            visual = torch.squeeze(visual, 1)
            acoustic = torch.squeeze(acoustic, 1)
            visual, acoustic = apply_test_time_corruption(
                visual, acoustic, corruption_generator
            )
            reg_logits, cls_logits, _, analysis = model.test(
                input_ids,
                visual,
                acoustic,
                token_type_ids=segment_ids,
                attention_mask=input_mask,
                labels=None,
                return_analysis=True,
                **corr_inputs,
            )

            reg_preds.append(reg_logits.detach().cpu().view(-1).numpy())
            cls_preds.append(torch.argmax(cls_logits, dim=1).detach().cpu().numpy())
            reprs.append(analysis["repr"].detach().cpu().numpy())
            labels.append(label_ids.detach().cpu().view(-1).numpy())

            batch_size = label_ids.size(0)
            for key in signal_keys:
                values = analysis.get(key)
                signal_values[key].append(np.full(batch_size, np.nan, dtype=np.float32) if values is None else values.detach().cpu().view(-1).numpy())

    return {
        "pred_reg": np.concatenate(reg_preds, axis=0),
        "pred_cls": np.concatenate(cls_preds, axis=0),
        "repr": np.concatenate(reprs, axis=0),
        "label_reg": np.concatenate(labels, axis=0),
        **{key: np.concatenate(values, axis=0) for key, values in signal_values.items()},
    }


def collect_reliability_summary(model: nn.Module, test_dataloader: DataLoader):
    """Summarize modality reliabilities without writing large representation files."""
    model.eval()
    values = {key: [] for key in ("raw_cosine_ta", "raw_cosine_tv", "raw_cosine_av", "c_ta", "c_tv", "c_av", "q_t", "q_a", "q_v", "s_t", "s_a", "s_v")}
    corruption_generator = make_test_corruption_generator()
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Reliability summary"):
            batch = tuple(t.to(DEVICE) for t in batch)
            input_ids, visual, acoustic, input_mask, segment_ids, _, corr_inputs = unpack_batch(batch)
            visual = torch.squeeze(visual, 1)
            acoustic = torch.squeeze(acoustic, 1)
            visual, acoustic = apply_test_time_corruption(
                visual, acoustic, corruption_generator
            )
            _, _, _, analysis = model.test(
                input_ids,
                visual,
                acoustic,
                token_type_ids=segment_ids,
                attention_mask=input_mask,
                return_analysis=True,
                **corr_inputs,
            )
            for key in values:
                score = analysis.get(key)
                if score is not None:
                    values[key].append(score.detach().cpu().view(-1).numpy())

    summary = {}
    for key, chunks in values.items():
        if not chunks:
            summary[key] = {"mean": None, "std": None, "count": 0}
            continue
        merged = np.concatenate(chunks)
        summary[key] = {
            "mean": float(np.mean(merged)),
            "std": float(np.std(merged)),
            "count": int(merged.size),
        }
    return summary


def export_analysis_files(
    model: nn.Module,
    test_dataloader: DataLoader,
    test_examples,
):
    collected = collect_test_outputs(model, test_dataloader)
    if len(test_examples) != len(collected["pred_reg"]):
        raise ValueError(
            f"Test example count mismatch: examples={len(test_examples)} preds={len(collected['pred_reg'])}"
        )

    output_dir = resolve_analysis_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    prediction_rows = []
    case_rows = []
    repr_sample_ids = []
    repr_texts = []
    repr_label_2 = []
    repr_label_7 = []
    repr_pred_2 = []
    repr_pred_7 = []

    for idx, example in enumerate(test_examples):
        base = extract_analysis_record(example)
        pred_reg = float(collected["pred_reg"][idx])
        row = {
            "sample_id": base["sample_id"],
            "text": base["text"],
            "label_reg": float(base["label_reg"]),
            "pred_reg": pred_reg,
            "label_2": int(base["label_2"]),
            "pred_2": prediction_to_binary(pred_reg),
            "label_7": int(base["label_7"]),
            "pred_7": prediction_to_seven(pred_reg),
        }
        prediction_rows.append(row)
        case_rows.append(
            {
                "sample_id": base["sample_id"],
                "text": base["text"],
                "label": float(base["label_reg"]),
                "pred": pred_reg,
                **{key: float(collected[key][idx]) for key in ("raw_cosine_ta", "raw_cosine_tv", "raw_cosine_av", "c_ta", "c_tv", "c_av", "q_t", "q_a", "q_v", "s_t", "s_a", "s_v")},
            }
        )
        repr_sample_ids.append(base["sample_id"])
        repr_texts.append(base["text"])
        repr_label_2.append(int(base["label_2"]))
        repr_label_7.append(int(base["label_7"]))
        repr_pred_2.append(prediction_to_binary(pred_reg))
        repr_pred_7.append(prediction_to_seven(pred_reg))

    prediction_path = output_dir / "predictions.csv"
    with prediction_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "text",
                "label_reg",
                "pred_reg",
                "label_2",
                "pred_2",
                "label_7",
                "pred_7",
            ],
        )
        writer.writeheader()
        writer.writerows(prediction_rows)

    case_path = output_dir / "cases.csv"
    with case_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sample_id", "text", "label", "pred", "raw_cosine_ta", "raw_cosine_tv", "raw_cosine_av", "c_ta", "c_tv", "c_av", "q_t", "q_a", "q_v", "s_t", "s_a", "s_v"],
        )
        writer.writeheader()
        writer.writerows(case_rows)

    repr_path = output_dir / "repr.npz"
    np.savez_compressed(
        repr_path,
        repr=collected["repr"].astype(np.float32),
        label_reg=collected["label_reg"].astype(np.float32),
        label_2=np.asarray(repr_label_2, dtype=np.int64),
        label_7=np.asarray(repr_label_7, dtype=np.int64),
        pred_reg=collected["pred_reg"].astype(np.float32),
        pred_2=np.asarray(repr_pred_2, dtype=np.int64),
        pred_7=np.asarray(repr_pred_7, dtype=np.int64),
        **{key: collected[key].astype(np.float32) for key in ("raw_cosine_ta", "raw_cosine_tv", "raw_cosine_av", "c_ta", "c_tv", "c_av", "q_t", "q_a", "q_v", "s_t", "s_a", "s_v")},
    )

    metadata_path = output_dir / "repr_meta.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": args.dataset,
                "sample_id": repr_sample_ids,
                "text": repr_texts,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Analysis files exported to: {output_dir}")
    print(f"  predictions: {prediction_path}")
    print(f"  repr: {repr_path}")
    print(f"  repr meta: {metadata_path}")
    print(f"  cases: {case_path}")

    return {
        "output_dir": str(output_dir),
        "prediction_path": str(prediction_path),
        "repr_path": str(repr_path),
        "repr_meta_path": str(metadata_path),
        "case_path": str(case_path),
    }


def multiclass_acc(preds, truths):
    """
    Compute the multiclass accuracy w.r.t. groundtruth

    :param preds: Float array representing the predictions, dimension (N,)
    :param truths: Float/int array representing the groundtruth classes, dimension (N,)
    :return: Classification accuracy
    """
    return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))

def build_class_labels(label_ids, cls_task):
    labels = label_ids.view(-1)
    if cls_task == "binary":
        return (labels >= 0).long()
    rounded = torch.round(labels).clamp(-3, 3).long()
    return rounded + 3

def test_score_model(model: nn.Module, test_dataloader: DataLoader, use_zero=False):

    test_preds, test_truth = test_epoch(model, test_dataloader)
    mae = np.mean(np.absolute(test_preds - test_truth))   # Average L1 distance between preds and truths
    corr = np.corrcoef(test_preds, test_truth)[0][1]
    
    non_zeros = np.array(
        [i for i, e in enumerate(test_truth) if e != 0 or use_zero])

    test_preds_a7 = np.clip(test_preds, a_min=-3., a_max=3.)
    test_truth_a7 = np.clip(test_truth, a_min=-3., a_max=3.)
    mult_a7 = multiclass_acc(test_preds_a7, test_truth_a7)
    
    test_preds_a5 = np.clip(test_preds, a_min=-2., a_max=2.)
    test_truth_a5 = np.clip(test_truth, a_min=-2., a_max=2.)
    mult_a5 = multiclass_acc(test_preds_a5, test_truth_a5)
    binary_truth_o = (test_truth[non_zeros] > 0) # 
    binary_preds_o = (test_preds[non_zeros] > 0) # 
    acc2_non_zero = accuracy_score(binary_truth_o, binary_preds_o)
    f_score_non_zero = f1_score(binary_truth_o, binary_preds_o,  average='weighted')
    

    binary_truth = (test_truth >= 0) # 
    binary_preds = (test_preds >= 0) # 
    acc2 = accuracy_score(binary_truth, binary_preds) # 
    f_score = f1_score(binary_truth, binary_preds, average='weighted')
    f_score_bias = f1_score((test_preds > 0), (test_truth >= 0), average='weighted')

    return mae, corr, mult_a7, mult_a5, acc2_non_zero, f_score_non_zero, acc2, f_score


def profile_inference(model: nn.Module, test_dataloader: DataLoader):
    """Profile a deterministic batch-1 forward pass, excluding data loading."""
    model.eval()
    batch = next(iter(test_dataloader))
    batch = tuple(t.to(DEVICE) for t in batch)
    input_ids, visual, acoustic, input_mask, segment_ids, _ = batch
    input_ids = input_ids[:1]
    visual = torch.squeeze(visual[:1], 1)
    acoustic = torch.squeeze(acoustic[:1], 1)
    input_mask = input_mask[:1]
    segment_ids = segment_ids[:1]

    def forward_once():
        model.test(
            input_ids,
            visual,
            acoustic,
            token_type_ids=segment_ids,
            attention_mask=input_mask,
        )

    with torch.no_grad():
        for _ in range(args.profile_warmup):
            forward_once()
        if DEVICE.type == "cuda":
            torch.cuda.synchronize(DEVICE)
            torch.cuda.reset_peak_memory_stats(DEVICE)
        start = time.perf_counter()
        for _ in range(args.profile_repeats):
            forward_once()
        if DEVICE.type == "cuda":
            torch.cuda.synchronize(DEVICE)
        elapsed_s = time.perf_counter() - start

    total_parameters = sum(p.numel() for p in model.parameters())
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    profile = {
        "protocol": "single-sample forward pass; excludes data loading and checkpoint loading",
        "device": str(DEVICE),
        "warmup_iterations": args.profile_warmup,
        "timed_iterations": args.profile_repeats,
        "total_parameters": int(total_parameters),
        "trainable_parameters": int(trainable_parameters),
        "latency_ms_per_sample": 1000.0 * elapsed_s / args.profile_repeats,
        "throughput_samples_per_second": args.profile_repeats / elapsed_s,
        "peak_gpu_memory_mib": (
            float(torch.cuda.max_memory_allocated(DEVICE)) / (1024.0 ** 2)
            if DEVICE.type == "cuda" else None
        ),
    }
    return profile


def train(model, train_dataloader, validation_dataloader, test_data_loader, test_examples=None):
    optimizer = build_optimizer(model)
    best_value, best_epoch, best_valid_mse = None, None, None
    best_model_path = resolve_best_model_path()
    history_path = resolve_experiment_dir() / "history.jsonl"
    for epoch_i in range(int(args.n_epochs)):
        train_loss = train_epoch(model, train_dataloader, optimizer, epoch_i)
        valid_mse, valid_mae = eval_epoch(model, validation_dataloader)
        if not np.isfinite([train_loss, valid_mse, valid_mae]).all():
            raise RuntimeError("Non-finite train/validation metric")
        improved = best_value is None or valid_mae < best_value
        if improved:
            best_value, best_epoch, best_valid_mse = valid_mae, epoch_i, valid_mse
            torch.save(model.state_dict(), best_model_path)
        record = {"epoch_zero_based": epoch_i, "train_loss": train_loss,
                  "valid_mae": valid_mae, "valid_mse": valid_mse,
                  "learning_rate": optimizer.param_groups[0]["lr"], "selected": improved}
        with history_path.open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
    if best_epoch is None:
        raise RuntimeError("No validation-selected checkpoint")
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE), strict=True)
    # Development pilot: evaluate only the selected checkpoint on dev.
    predictions, truth = test_epoch(model, validation_dataloader)
    np.savez_compressed(resolve_experiment_dir() / "dev_predictions.npz",
                        predictions=predictions, labels=truth)
    summary = {"protocol": VERSION, "dataset": args.dataset, "seed": args.seed,
        "evaluation_split": "dev", "test_evaluated": False,
        "best_epoch": best_epoch, "epoch_index": "zero_based",
        "valid_mae": best_value, "valid_mse": best_valid_mse,
        "reloaded_dev_mae": float(np.mean(np.abs(predictions - truth))),
        "correlation_control": args.correlation_control,
        "checkpoint_path": str(best_model_path),
        "checkpoint_sha256": file_sha256(best_model_path)}
    if not np.isclose(summary["reloaded_dev_mae"], best_value, atol=1e-6):
        raise RuntimeError("Reloaded dev MAE differs from selected checkpoint")
    write_metrics_summary(summary)
    print(json.dumps(summary), flush=True)
    return summary

def validate_formal_args():
    policy = json.loads((Path(__file__).resolve().parents[1] / "protocol/protocol.json").read_text())
    root = Path(__file__).resolve().parents[1]
    if Path(args.experiment_root).resolve() != root / "runs" or args.best_model_path or args.analysis_tag:
        raise ValueError("Formal run outputs must remain in their unique run directory")
    if args.analysis_output_dir != "analysis/outputs":
        raise ValueError("Custom analysis output paths are disabled")
    if os.environ.get("PYTHONHASHSEED") != str(args.seed):
        raise ValueError("PYTHONHASHSEED must be set before interpreter startup; use the launcher")
    if args.seed not in policy["seeds"]:
        raise ValueError("Seed is outside the frozen protocol")
    for key, expected in {**policy["shared"], **policy["datasets"][args.dataset]}.items():
        attr = "weight_decay" if key == "weight-decay" else key
        if getattr(args, attr) != expected:
            raise ValueError("Frozen protocol mismatch: " + key)
    if args.use_fusion_correlation or args.use_moe_reliability:
        if args.corr_model_path != policy["correlation_checkpoints"][args.dataset] or args.corr_cache_root != policy["cache_root"]:
            raise ValueError("Frozen correlation input paths mismatch")
    if args.save_best_by != "valid_mae" or args.gradient_accumulation_step != 1:
        raise ValueError("Formal protocol requires valid_mae and gradient_accumulation_step=1")
    if args.cos or args.test or args.load or args.paired_init_load or args.paired_init_save:
        raise ValueError("Use the formal launcher; legacy load/test/schedule paths are disabled")
    if args.test_audio_noise_std or args.test_visual_mask_ratio:
        raise ValueError("Corruption evaluation is pending synchronized-input validation")
    if args.use_modality_text_aug or args.drop_text or args.drop_audio or args.drop_visual:
        raise ValueError("Formal primary protocol uses all three original modalities")
    if args.model != "bert-base-uncased" or args.cls_task != "seven":
        raise ValueError("Formal protocol requires BERT and seven-class auxiliary loss")
    if args.correlation_control != "none":
        raise ValueError("Module ablations use dynamic guidance where enabled")
    if args.modalities not in policy['variants'] or [bool(args.use_fusion_correlation),bool(args.use_moe_reliability)] != policy['variants'][args.modalities]:
        raise ValueError('Guidance flags must match the frozen modality subset')
    expected_tag = '{}_{}_{}_seed{}'.format(policy['version'],args.dataset,args.modalities,args.seed)
    if args.experiment_tag != expected_tag:
        raise ValueError("Experiment tag must match the ablation switches")
    if not args.experiment_tag or Path(args.experiment_tag).name != args.experiment_tag:
        raise ValueError("A single-component, unique experiment tag is required")
    if args.n_epochs < 1:
        raise ValueError("n_epochs must be positive")


def main():

    validate_formal_args()
    root = Path(__file__).resolve().parents[1]
    source_identity = verify_source(root)
    runtime_identity = verify_environment(root)
    input_identity = verify_inputs(root, args.dataset)
    run_dir = Path(args.experiment_root).resolve() / args.experiment_tag
    run_dir.mkdir(parents=True, exist_ok=False)
    set_random_seed(args.seed)
    start_time = time.time()
    command_path = write_command_script()
    print(f"Command script saved to: {command_path}")

    (
        train_data_loader,
        dev_data_loader,
        test_data_loader,
        test_examples,
        num_train_optimization_steps,
    ) = set_up_data_loader()

    set_random_seed(args.seed)
    model = prep_for_training(num_train_optimization_steps)
    constants = set_training_scale_means(model, train_data_loader)
    config = {"protocol": VERSION, "arguments": vars(args), "source": source_identity,
        "input_sha256": input_identity, "backbone_initialization_sha256": backbone_digest(model),
        "routing_initialization_sha256": tensor_digest((k, v) for k, v in model.state_dict().items() if k.endswith(".w_rel")),
        "constant_scales": constants,
        "samples": {"train": len(train_data_loader.dataset), "dev": len(dev_data_loader.dataset), "test": len(test_data_loader.dataset)},
        "runtime_versions": runtime_identity,
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
            "transformers": __import__("transformers").__version__, "cuda": torch.version.cuda,
            "device": str(DEVICE), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}}
    (run_dir / "run_config.json").write_text(json.dumps(config, indent=2))
    set_random_seed(args.seed + 40000)

    if args.test:
        if not args.load:
            print("Warning: args.test=1 but args.load=0, evaluating current in-memory weights.")
        test_mae, test_corr, test_acc7, test_acc5, test_acc2_non_zero, test_f_score_non_zero, test_acc2, test_f_score = test_score_model(
            model, test_data_loader
        )
        print(
            "test mae:{:.4f}, corr:{:.4f}, acc7:{:.4f}, acc5:{:.4f}, acc2_non_zero:{:.4f}, f_score_non_zero:{:.4f}, acc2:{:.4f}, f_score:{:.4f}".format(
                test_mae, test_corr, test_acc7, test_acc5, test_acc2_non_zero, test_f_score_non_zero, test_acc2, test_f_score
            )
        )
        reliability_summary = collect_reliability_summary(model, test_data_loader)
        reliability_path = resolve_experiment_dir() / "reliability.json"
        with reliability_path.open("w", encoding="utf-8") as f:
            json.dump(reliability_summary, f, ensure_ascii=False, indent=2)
        print(f"Reliability summary saved to: {reliability_path}")

        efficiency_profile = None
        efficiency_path = None
        if args.profile_inference:
            efficiency_profile = profile_inference(model, test_data_loader)
            efficiency_path = resolve_experiment_dir() / "efficiency.json"
            with efficiency_path.open("w", encoding="utf-8") as f:
                json.dump(efficiency_profile, f, ensure_ascii=False, indent=2)
            print(f"Efficiency profile saved to: {efficiency_path}")

        analysis_info = None
        if args.export_analysis:
            analysis_info = export_analysis_files(model, test_data_loader, test_examples)
        summary = {
            "dataset": args.dataset,
            "seed": args.seed,
            "save_best_by": args.save_best_by,
            "use_modality_text_aug": args.use_modality_text_aug,
            "use_audio_desc": args.use_audio_desc,
            "use_visual_desc": args.use_visual_desc,
            "visual_desc_version": args.visual_desc_version,
            "best_epoch": None,
            "best_value": None,
            "mae": test_mae,
            "corr": test_corr,
            "acc2": test_acc2,
            "acc2_non_zero": test_acc2_non_zero,
            "f1": test_f_score,
            "f1_non_zero": test_f_score_non_zero,
            "acc7": test_acc7,
            "valid_mse": None,
            "valid_mae": None,
            "experiment_dir": str(resolve_experiment_dir()),
            "analysis_dir": None if analysis_info is None else analysis_info["output_dir"],
            "log_path": args.log_path,
            "checkpoint_path": args.model_path if args.load else "",
            "test_audio_noise_std": args.test_audio_noise_std,
            "test_visual_mask_ratio": args.test_visual_mask_ratio,
            "test_corruption_seed": args.test_corruption_seed,
            "correlation_control": args.correlation_control,
            "reliability_path": str(reliability_path),
            "efficiency_path": None if efficiency_path is None else str(efficiency_path),
            "efficiency_profile": efficiency_profile,
        }
        metrics_path = write_metrics_summary(summary)
        print(f"Metrics summary saved to: {metrics_path}")
    else:
        train(
            model,
            train_data_loader,
            dev_data_loader,
            test_data_loader,
            test_examples=test_examples,
        )
    end_time = time.time()
    print('Total runtime: %s ms' %((end_time - start_time) * 1000))


if __name__ == "__main__":
    main()


