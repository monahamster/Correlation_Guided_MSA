from __future__ import absolute_import, division, print_function

import argparse
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
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "7")

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


from GLoMo import GLoMo
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
parser.add_argument("--test", type=int, default=0)
parser.add_argument("--model_path", type=str, default='glomo.pth')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')
parser.add_argument("--cls_task", type=str, choices=["binary", "seven"], default="seven")
parser.add_argument("--reg_loss_weight", type=float, default=1.0)
parser.add_argument("--cls_loss_weight", type=float, default=1.0)
parser.add_argument("--use_correlation", action="store_true")
parser.add_argument("--use_fusion_correlation", action="store_true")
parser.add_argument("--corr_model_path", type=str, default=None)
parser.add_argument("--corr_alpha", type=float, default=1.0)
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
    default="acc2",
)
parser.add_argument("--best_model_path", type=str, default="")
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


def get_appropriate_dataset(data):

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

    dataset = TensorDataset(
        all_input_ids,
        all_visual,
        all_acoustic,
        all_input_mask,
        all_segment_ids,
        all_label_ids,
    )
    return dataset


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

    train_dataset = get_appropriate_dataset(train_data)
    dev_dataset = get_appropriate_dataset(dev_data)
    test_dataset = get_appropriate_dataset(test_data)

    num_train_optimization_steps = (
        int(
            len(train_dataset) / args.train_batch_size /
            args.gradient_accumulation_step
        )
        * args.n_epochs
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True
    )

    dev_dataloader = DataLoader(
        dev_dataset, batch_size=args.dev_batch_size, shuffle=True
    )

    test_dataloader = DataLoader(
        test_dataset, batch_size=args.test_batch_size, shuffle=False,
    )

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
    
    if args.load:
        model.load_state_dict(torch.load(args.model_path))

    model.to(DEVICE)

    return model
    
def adjust_learning_rate(optimizer, epoch, args):# 
    """Decay the learning rate based on schedule"""
    lr = args.learning_rate
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups: # 
        param_group['lr'] = lr

def train_epoch(model: nn.Module, train_dataloader: DataLoader, epoch=None):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0} # 
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    adjust_learning_rate(optimizer, epoch, args)  
 
    model.train()
    tr_loss = 0
    nb_tr_steps = 0


    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        batch = tuple(t.to(DEVICE) for t in batch)
        input_ids, visual, acoustic, input_mask, segment_ids, label_ids = batch
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
        


        if args.gradient_accumulation_step > 1:
            loss = loss / args.gradient_accumulation_step

        tr_loss += loss_all.item()
        nb_tr_steps += 1

    return tr_loss / nb_tr_steps


def eval_epoch(model: nn.Module, dev_dataloader: DataLoader):
    model.eval()
    dev_mse = 0
    dev_mae = 0
    nb_dev_steps = 0
    with torch.no_grad():
        for step, batch in enumerate(tqdm(dev_dataloader, desc="Iteration")):
            batch = tuple(t.to(DEVICE) for t in batch)

            input_ids, visual, acoustic, input_mask, segment_ids, label_ids = batch
            visual = torch.squeeze(visual, 1)
            acoustic = torch.squeeze(acoustic, 1)
            reg_logits, _, _ = model.test(
                input_ids,
                 visual,
                 acoustic,
                token_type_ids=segment_ids,
                attention_mask=input_mask,
            )

            mse_fct = MSELoss()
            mae_fct = L1Loss()
            mse_loss = mse_fct(reg_logits.view(-1), label_ids.view(-1))
            mae_loss = mae_fct(reg_logits.view(-1), label_ids.view(-1))

            if args.gradient_accumulation_step > 1:
                mse_loss = mse_loss / args.gradient_accumulation_step
                mae_loss = mae_loss / args.gradient_accumulation_step

            dev_mse += mse_loss.item()
            dev_mae += mae_loss.item()
            nb_dev_steps += 1
    return dev_mse / nb_dev_steps, dev_mae / nb_dev_steps


def test_epoch(model: nn.Module, test_dataloader: DataLoader):
    model.eval()
    preds = []
    labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            batch = tuple(t.to(DEVICE) for t in batch)

            input_ids, visual, acoustic, input_mask, segment_ids, label_ids = batch
            visual = torch.squeeze(visual, 1)
            acoustic = torch.squeeze(acoustic, 1)
            reg_logits, _, _ = model.test(
                input_ids,
                 visual,
                 acoustic,
                token_type_ids=segment_ids,
                attention_mask=input_mask,
                labels=None,
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


def collect_test_outputs(model: nn.Module, test_dataloader: DataLoader):
    model.eval()
    reg_preds = []
    cls_preds = []
    reprs = []
    reliabilities_t = []
    reliabilities_a = []
    reliabilities_v = []
    labels = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Test analysis"):
            batch = tuple(t.to(DEVICE) for t in batch)
            input_ids, visual, acoustic, input_mask, segment_ids, label_ids = batch
            visual = torch.squeeze(visual, 1)
            acoustic = torch.squeeze(acoustic, 1)
            reg_logits, cls_logits, _, analysis = model.test(
                input_ids,
                visual,
                acoustic,
                token_type_ids=segment_ids,
                attention_mask=input_mask,
                labels=None,
                return_analysis=True,
            )

            reg_preds.append(reg_logits.detach().cpu().view(-1).numpy())
            cls_preds.append(torch.argmax(cls_logits, dim=1).detach().cpu().numpy())
            reprs.append(analysis["repr"].detach().cpu().numpy())
            labels.append(label_ids.detach().cpu().view(-1).numpy())

            batch_size = label_ids.size(0)
            for key, bucket in (
                ("r_t", reliabilities_t),
                ("r_a", reliabilities_a),
                ("r_v", reliabilities_v),
            ):
                values = analysis.get(key)
                if values is None:
                    bucket.append(np.full(batch_size, np.nan, dtype=np.float32))
                else:
                    bucket.append(values.detach().cpu().view(-1).numpy())

    return {
        "pred_reg": np.concatenate(reg_preds, axis=0),
        "pred_cls": np.concatenate(cls_preds, axis=0),
        "repr": np.concatenate(reprs, axis=0),
        "label_reg": np.concatenate(labels, axis=0),
        "r_t": np.concatenate(reliabilities_t, axis=0),
        "r_a": np.concatenate(reliabilities_a, axis=0),
        "r_v": np.concatenate(reliabilities_v, axis=0),
    }


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
                "r_t": float(collected["r_t"][idx]),
                "r_a": float(collected["r_a"][idx]),
                "r_v": float(collected["r_v"][idx]),
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
            fieldnames=["sample_id", "text", "label", "pred", "r_t", "r_a", "r_v"],
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
        r_t=collected["r_t"].astype(np.float32),
        r_a=collected["r_a"].astype(np.float32),
        r_v=collected["r_v"].astype(np.float32),
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


def train(
    model,
    train_dataloader,
    validation_dataloader,
    test_data_loader,
    test_examples=None,
):
    valid_losses = []
    test_accuracies = []
    f1_scores = []
    best_value = None
    best_epoch = None
    best_metrics = None
    best_model_path = resolve_best_model_path()
    best_model_path.parent.mkdir(parents=True, exist_ok=True)
    for epoch_i in range(int(args.n_epochs)):
        train_loss = train_epoch(model, train_dataloader, epoch_i)
        valid_mse, valid_mae = eval_epoch(model, validation_dataloader)
        test_mae, test_corr, test_acc7, test_acc5, test_acc2_non_zero, test_f_score_non_zero, test_acc2, test_f_score= test_score_model(
            model, test_data_loader
        )
        current_metrics = {
            "valid_mse": valid_mse,
            "valid_mae": valid_mae,
            "test_mae": test_mae,
            "test_corr": test_corr,
            "test_acc7": test_acc7,
            "test_acc5": test_acc5,
            "test_acc2_non_zero": test_acc2_non_zero,
            "test_f_score_non_zero": test_f_score_non_zero,
            "test_acc2": test_acc2,
            "test_f_score": test_f_score,
        }

        print(
            "epoch:{}, train_loss:{:.4f}, valid_mse:{:.4f}, valid_mae:{:.4f}, test_acc2:{:.4f}".format(
                epoch_i, train_loss, valid_mse, valid_mae, test_acc2
            )
        )


        print(
            "current mae:{:.4f}, current corr:{:.4f}, acc7:{:.4f}, acc5:{:.4f},acc2_non_zero:{:.4f}, f_score_non_zero:{:.4f}, acc2:{:.4f}, f_score:{:.4f}".format(
                test_mae, test_corr, test_acc7, test_acc5, test_acc2_non_zero, test_f_score_non_zero, test_acc2, test_f_score
            )
        )


        valid_losses.append(valid_mse)
        test_accuracies.append(test_acc2)
        f1_scores.append(test_f_score_non_zero)

        tracked_value = metric_value(args.save_best_by, current_metrics)
        if is_better(args.save_best_by, tracked_value, best_value):
            best_value = tracked_value
            best_epoch = epoch_i
            best_metrics = dict(current_metrics)
            torch.save(model.state_dict(), best_model_path)
            print(
                "Saved best model at epoch {} by {}={:.4f} -> {}".format(
                    epoch_i,
                    args.save_best_by,
                    tracked_value,
                    best_model_path,
                )
            )

    analysis_info = None
    if test_examples is not None:
        if best_model_path.exists():
            print(
                "Loading best model from epoch {} by {}={:.4f} for analysis export.".format(
                    best_epoch,
                    args.save_best_by,
                    best_value,
                )
            )
            model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
        analysis_info = export_analysis_files(model, test_data_loader, test_examples)

    summary = {
        "dataset": args.dataset,
        "seed": args.seed,
        "save_best_by": args.save_best_by,
        "use_modality_text_aug": args.use_modality_text_aug,
        "use_audio_desc": args.use_audio_desc,
        "use_visual_desc": args.use_visual_desc,
        "visual_desc_version": args.visual_desc_version,
        "best_epoch": best_epoch,
        "best_value": best_value,
        "mae": None if best_metrics is None else best_metrics["test_mae"],
        "corr": None if best_metrics is None else best_metrics["test_corr"],
        "acc2": None if best_metrics is None else best_metrics["test_acc2"],
        "acc2_non_zero": None if best_metrics is None else best_metrics["test_acc2_non_zero"],
        "f1": None if best_metrics is None else best_metrics["test_f_score"],
        "f1_non_zero": None if best_metrics is None else best_metrics["test_f_score_non_zero"],
        "acc7": None if best_metrics is None else best_metrics["test_acc7"],
        "valid_mse": None if best_metrics is None else best_metrics["valid_mse"],
        "valid_mae": None if best_metrics is None else best_metrics["valid_mae"],
        "experiment_dir": str(resolve_experiment_dir()),
        "analysis_dir": None if analysis_info is None else analysis_info["output_dir"],
        "log_path": args.log_path,
        "checkpoint_path": str(best_model_path),
    }
    metrics_path = write_metrics_summary(summary)
    print(f"Metrics summary saved to: {metrics_path}")
    return summary


def main():

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

    model = prep_for_training(
        num_train_optimization_steps)#

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
            "analysis_dir": analysis_info["output_dir"],
            "log_path": args.log_path,
            "checkpoint_path": args.model_path if args.load else "",
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


