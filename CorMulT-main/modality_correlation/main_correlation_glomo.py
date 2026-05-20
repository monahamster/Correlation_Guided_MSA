import argparse
import os
import time

from correlation_train import pretrain_correlation_model
from glomo_dataset import GLoMoHumorPKLDataset, GLoMoPKLDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["mosi", "mosei", "mustard", "urfunny"], default="mosi")
    parser.add_argument("--pkl_path", type=str, default=None)
    parser.add_argument("--bert_path", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--build_cache", action="store_true")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--model_save_name", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--max_seq_length", type=int, default=None)
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    if args.pkl_path is None:
        default_pkl_name = {
            "mosi": "mosi.pkl",
            "mosei": "mosei.pkl",
            "mustard": "mustard.pkl",
            "urfunny": "ur_funny.pkl",
        }[args.dataset]
        args.pkl_path = os.path.join(repo_root, "datasets", default_pkl_name)
    if args.bert_path is None:
        args.bert_path = os.path.join(repo_root, "BERT_EN")
    if args.cache_dir is None:
        args.cache_dir = os.path.join("pre_trained_models", "glomo_cache")
    if args.model_save_name is None:
        args.model_save_name = f"correlation_glomo_{args.dataset}_{time.strftime('%m%d%H')}"
    if args.save_dir is None:
        args.save_dir = os.path.join(repo_root, "pretrained-model")
    if args.max_seq_length is None:
        args.max_seq_length = {
            "mosi": 60,
            "mosei": 80,
            "mustard": 70,
            "urfunny": 80,
        }[args.dataset]

    dataset_name_map = {
        "mosi": "glomo_mosi",
        "mosei": "glomo_mosei",
        "mustard": "glomo_mustard",
        "urfunny": "glomo_urfunny",
    }
    dataset_name = dataset_name_map[args.dataset]

    if args.dataset in {"mustard", "urfunny"}:
        train_data = GLoMoHumorPKLDataset(
            pkl_path=args.pkl_path,
            split="train",
            bert_path=args.bert_path,
            max_seq_length=args.max_seq_length,
            cache_dir=args.cache_dir,
            build_cache=args.build_cache,
            for_correlation=True,
        )
        valid_data = GLoMoHumorPKLDataset(
            pkl_path=args.pkl_path,
            split="dev",
            bert_path=args.bert_path,
            max_seq_length=args.max_seq_length,
            cache_dir=args.cache_dir,
            build_cache=args.build_cache,
            for_correlation=True,
        )
    else:
        train_data = GLoMoPKLDataset(
            pkl_path=args.pkl_path,
            split="train",
            bert_path=args.bert_path,
            cache_dir=args.cache_dir,
            build_cache=args.build_cache,
            for_correlation=True,
        )
        valid_data = GLoMoPKLDataset(
            pkl_path=args.pkl_path,
            split="dev",
            bert_path=args.bert_path,
            cache_dir=args.cache_dir,
            build_cache=args.build_cache,
            for_correlation=True,
        )

    from types import SimpleNamespace

    train_args = SimpleNamespace(
        dataset_name=dataset_name,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr=args.lr,
        model_save_name=args.model_save_name,
        data_path="",
        save_dir=args.save_dir,
    )

    pretrain_correlation_model(train_args, train_dataset=train_data, valid_dataset=valid_data)


if __name__ == "__main__":
    main()
