import os
import argparse
import random
import pickle
import json

import numpy as np
import torch
import torch.nn as nn

from tqdm import tqdm
from torch.utils.data import DataLoader, Subset

from load_data import T5Dataset, normal_collate_fn

from t5_utils import (
    initialize_model,
    initialize_optimizer_and_scheduler,
    save_model
)

from train_t5_ec import train_epoch
from utils import compute_metrics, save_queries_and_records

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

PAD_IDX = 0


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--experiment_name", required=True)
    parser.add_argument("--finetune", action="store_true")

    parser.add_argument("--optimizer_type", default="AdamW")
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    parser.add_argument("--scheduler_type", default="cosine")
    parser.add_argument("--num_warmup_epochs", type=int, default=1)
    parser.add_argument("--max_n_epochs", type=int, default=8)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--test_batch_size", type=int, default=8)

    parser.add_argument("--train_subset_size", type=int, default=1500)
    parser.add_argument("--dev_subset_size", type=int, default=150)

    parser.add_argument("--eval_every", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--use_wandb", action="store_true")

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_subset_data(args, output_dir):

    train_dataset = T5Dataset("data", "train")
    dev_dataset = T5Dataset("data", "dev")

    # Same subsets for all experiments
    rng = random.Random(args.seed)

    train_indices = sorted(
        rng.sample(
            range(len(train_dataset)),
            args.train_subset_size
        )
    )

    dev_indices = sorted(
        rng.sample(
            range(len(dev_dataset)),
            args.dev_subset_size
        )
    )

    train_subset = Subset(train_dataset, train_indices)
    dev_subset = Subset(dev_dataset, dev_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=normal_collate_fn,
        generator=torch.Generator().manual_seed(args.seed)
    )

    dev_loader = DataLoader(
        dev_subset,
        batch_size=args.test_batch_size,
        shuffle=False,
        collate_fn=normal_collate_fn
    )

    # Save sampled indices for reproducibility
    with open(os.path.join(output_dir, "subset_indices.json"), "w") as f:
        json.dump({
            "train_indices": train_indices,
            "dev_indices": dev_indices
        }, f)

    # Save matching dev gold SQL
    gt_sql_path = os.path.join(output_dir, "ground_truth_dev.sql")

    with open(gt_sql_path, "w") as f:
        for i in dev_indices:
            f.write(dev_dataset.raw_sql[i] + "\n")

    # Save matching dev gold records
    with open("records/ground_truth_dev.pkl", "rb") as f:
        all_records, all_errors = pickle.load(f)

    assert len(all_records) == len(dev_dataset)
    assert len(all_errors) == len(dev_dataset)

    subset_records = [all_records[i] for i in dev_indices]
    subset_errors = [all_errors[i] for i in dev_indices]

    gt_record_path = os.path.join(
        output_dir, "ground_truth_dev.pkl"
    )

    with open(gt_record_path, "wb") as f:
        pickle.dump((subset_records, subset_errors), f)

    tokenizer = train_dataset.tokenizer

    assert tokenizer.get_vocab() == dev_dataset.tokenizer.get_vocab()

    print("Train subset:", len(train_subset))
    print("Dev subset:", len(dev_subset))
    print("Train batches:", len(train_loader))
    print("Vocabulary size:", len(tokenizer))

    return (
        train_loader,
        dev_loader,
        tokenizer,
        gt_sql_path,
        gt_record_path
    )


def eval_epoch(
    args,
    model,
    dev_loader,
    tokenizer,
    full_eval,
    gt_sql_path,
    gt_record_path,
    model_sql_path,
    model_record_path
):

    model.eval()

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_tokens = 0
    generated_queries = []

    with torch.no_grad():

        for (
            encoder_input,
            encoder_mask,
            decoder_input,
            decoder_targets,
            initial_decoder_inputs
        ) in tqdm(dev_loader):

            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)
            decoder_input = decoder_input.to(DEVICE)
            decoder_targets = decoder_targets.to(DEVICE)
            initial_decoder_inputs = initial_decoder_inputs.to(DEVICE)

            # Teacher-forced dev loss
            logits = model(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                decoder_input_ids=decoder_input
            )["logits"]

            non_pad = decoder_targets != PAD_IDX

            loss = criterion(
                logits[non_pad],
                decoder_targets[non_pad]
            )

            num_tokens = non_pad.sum().item()

            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

            # Only generate SQL at selected epochs
            if full_eval:

                generated_ids = model.generate(
                    input_ids=encoder_input,
                    attention_mask=encoder_mask,
                    decoder_input_ids=initial_decoder_inputs,
                    max_new_tokens=512,
                    num_beams=2
                )

                queries = tokenizer.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )

                generated_queries.extend(queries)

    dev_loss = total_loss / total_tokens

    # No generation or database execution this epoch
    if not full_eval:
        return dev_loss, None, None, None, None, None

    assert len(generated_queries) == len(dev_loader.dataset)

    save_queries_and_records(
        generated_queries,
        model_sql_path,
        model_record_path
    )

    sql_em, record_em, record_f1, error_msgs = compute_metrics(
        gt_sql_path,
        model_sql_path,
        gt_record_path,
        model_record_path
    )

    error_rate = sum(
        bool(msg) for msg in error_msgs
    ) / len(error_msgs)

    timeout_count = sum(
        msg == "Query timed out" for msg in error_msgs
    )

    return (
        dev_loss,
        record_f1,
        record_em,
        sql_em,
        error_rate,
        timeout_count
    )


def main():

    args = get_args()

    assert not args.finetune, "EC must train from scratch"
    assert args.eval_every > 0

    set_seed(args.seed)

    checkpoint_dir = os.path.join(
        "checkpoints",
        "scr_experiments",
        args.experiment_name
    )

    output_dir = os.path.join(
        checkpoint_dir,
        "dev_outputs"
    )

    os.makedirs(output_dir, exist_ok=True)

    print("\nExperiment:", args.experiment_name)
    print("T5 from scratch")
    print(vars(args))

    with open(os.path.join(checkpoint_dir, "config_args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    (
        train_loader,
        dev_loader,
        tokenizer,
        gt_sql_path,
        gt_record_path
    ) = load_subset_data(args, output_dir)

    # Independent random initialization
    set_seed(args.seed)

    model = initialize_model(args)

    model.resize_token_embeddings(len(tokenizer))

    optimizer, scheduler = initialize_optimizer_and_scheduler(
        args,
        model,
        len(train_loader)
    )

    best_f1 = -1
    best_epoch = None
    best_metrics = None
    history = []

    for epoch in range(1, args.max_n_epochs + 1):

        train_loss = train_epoch(
            args,
            model,
            train_loader,
            optimizer,
            scheduler
        )

        print(f"\nEpoch {epoch}: Train loss = {train_loss:.6f}")

        # Save before expensive evaluation
        save_model(checkpoint_dir, model, best=False)

        full_eval = (
            epoch % args.eval_every == 0
            or epoch == args.max_n_epochs
        )

        model_sql_path = os.path.join(
            output_dir,
            f"epoch_{epoch}.sql"
        )

        model_record_path = os.path.join(
            output_dir,
            f"epoch_{epoch}.pkl"
        )

        (
            dev_loss,
            record_f1,
            record_em,
            sql_em,
            error_rate,
            timeout_count
        ) = eval_epoch(
            args,
            model,
            dev_loader,
            tokenizer,
            full_eval,
            gt_sql_path,
            gt_record_path,
            model_sql_path,
            model_record_path
        )

        print(f"Epoch {epoch}: Dev loss = {dev_loss:.6f}")

        metrics = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "dev_loss": float(dev_loss),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "record_f1": None,
            "record_em": None,
            "sql_em": None,
            "error_rate": None,
            "timeout_count": None
        }

        if full_eval:

            print(f"Record F1: {record_f1:.4%}")
            print(f"Record EM: {record_em:.4%}")
            print(f"Query EM: {sql_em:.4%}")
            print(f"SQL error rate: {error_rate:.4%}")
            print(f"Query timeouts: {timeout_count}")

            metrics.update({
                "record_f1": float(record_f1),
                "record_em": float(record_em),
                "sql_em": float(sql_em),
                "error_rate": float(error_rate),
                "timeout_count": int(timeout_count)
            })

            if record_f1 > best_f1:

                best_f1 = record_f1
                best_epoch = epoch
                best_metrics = metrics.copy()

                save_model(
                    checkpoint_dir,
                    model,
                    best=True
                )

                with open(
                    os.path.join(checkpoint_dir, "best_metrics.json"),
                    "w"
                ) as f:
                    json.dump(best_metrics, f, indent=4)

                print("New best checkpoint saved.")

        else:
            print("Full SQL evaluation skipped.")

        # Save history after every completed evaluation
        history.append(metrics)

        with open(
            os.path.join(checkpoint_dir, "history.json"),
            "w"
        ) as f:
            json.dump(history, f, indent=4)

    summary = {
        "experiment_name": args.experiment_name,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "best_epoch": best_epoch,
        "best_metrics": best_metrics
    }

    with open(
        os.path.join(checkpoint_dir, "summary.json"),
        "w"
    ) as f:
        json.dump(summary, f, indent=4)

    print("\nGrid search experiment completed.")
    print(json.dumps(summary, indent=4))


if __name__ == "__main__":
    main()