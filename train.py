import csv
import datetime
import json
import os
import random
import time
from argparse import ArgumentParser

import datasets
import numpy as np
import pandas as pd
import torch
from eval_metrics import (compute_classification_scores,
                          compute_generation_scores)
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from toxcl import ToXCL, ToxclDataset
from tqdm import tqdm

from transformers import (AdamW, AutoModelForSeq2SeqLM,
                          AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)


def initialize_seed(seed_val=42):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)


def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))


def load_data(file_path, text_column_num):
    data = []
    with open(file_path) as file:
        csvreader = csv.reader(file)
        _ = next(csvreader)
        for row in csvreader:
            data.append({
                "document": row[text_column_num].strip(),
                "label": row[2].strip(),
                "summary": row[3].strip(),
            })
    return data


def main(args):
    # Initialization
    initialize_seed()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Backbone model and tokenizer initialization
    decoder_tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    decoder_model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path).to(device)
    if args.teacher_name_or_path:
        teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher_name_or_path)
        teacher_model = AutoModelForSequenceClassification.from_pretrained(args.teacher_name_or_path)
        for param in teacher_model.parameters():
            param.requires_grad = False

    # Data loading
    train_data = load_data(args.train_file, args.text_column_num)
    valid_data = load_data(args.valid_file, args.text_column_num)

    train_data = datasets.Dataset.from_pandas(pd.DataFrame(data=train_data))
    valid_data = datasets.Dataset.from_pandas(pd.DataFrame(data=valid_data))

    # Initialize datasets
    def collate_fn(batch):
        input_texts = [item["document"] for item in batch]
        label_texts = [item["label"] for item in batch]

        new_batch = decoder_tokenizer(input_texts, max_length=args.max_length, padding="max_length", return_tensors="pt", truncation=True)
        labels = decoder_tokenizer(label_texts, max_length=args.max_length, padding="max_length", return_tensors="pt", truncation=True).input_ids
        labels[labels == decoder_tokenizer.pad_token_id] = -100
        new_batch["labels"] = labels
        new_batch["student_cls_labels"] = torch.as_tensor([item["student_cls_labels"] for item in batch])
        new_batch["teacher_cls_labels"] = torch.as_tensor([item["teacher_cls_labels"] for item in batch])

        if args.teacher_name_or_path:
            teacher_inputs = teacher_tokenizer(input_texts, max_length=args.max_length, padding="max_length", return_tensors="pt", truncation=True)
            new_batch["teacher_input_ids"] = teacher_inputs["input_ids"]
            new_batch["teacher_attention_mask"] = teacher_inputs["attention_mask"]

        return new_batch

    train_dataset = ToxclDataset(train_data)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=args.train_batch_size,
        collate_fn=collate_fn,
        num_workers=8
    )

    valid_dataset = ToxclDataset(valid_data)
    validation_dataloader = DataLoader(
        valid_dataset,
        sampler=SequentialSampler(valid_dataset),
        batch_size=args.valid_batch_size,
        collate_fn=collate_fn,
        num_workers=8
    )

    print(f'{len(train_data):,} training samples')
    print(f'{len(valid_data):,} validation samples')

    # Initialize model, optimizer, and scheduler
    model = ToXCL(decoder_model).to(device)
    teacher_model = teacher_model.to(device) if args.teacher_name_or_path else None
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)
    total_steps = (len(train_dataloader) * args.num_epochs) // args.accumulation_steps
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)

    # Training loop
    total_t0 = time.time()
    total_train_lm_loss = 0
    total_train_cls_loss = 0
    total_train_kl_loss = 0
    training_stats = []
    best_result = float('inf')  # lm_loss by default
    num_step = 0

    if args.resume_training:
        model.load_checkpoint(args.output_dir, optimizer, scheduler)
        with open(os.path.join(args.output_dir, "training_model_stats.json"), "r") as file:
            training_stats = json.load(file)
        best_result = training_stats[-1]['Best result']
        num_step = training_stats[-1]['Step']

    for epoch_i in range(0, args.num_epochs):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, args.num_epochs))

        model.train()

        train_loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader), leave=False)
        for train_step, batch in train_loop:
            num_step += 1
            train_loop.update()

            batch = {k: v.to(device) for k, v in batch.items()}

            b_input_ids = batch.get("input_ids")
            b_lm_labels = batch.get("labels")
            b_attention_mask = batch.get("attention_mask")
            b_cls_labels = batch.get("student_cls_labels")
            
            b_teacher_input_ids = batch.get("teacher_input_ids")
            b_teacher_masks = batch.get("teacher_attention_mask")
            b_teacher_cls = batch.get("teacher_cls_labels")

            teacher_logits = None
            if args.teacher_name_or_path:
                with torch.no_grad():
                    teacher_outputs = teacher_model(
                        input_ids=b_teacher_input_ids,
                        attention_mask=b_teacher_masks,
                        labels=b_teacher_cls
                    )
                    teacher_logits = teacher_outputs.logits

            model.zero_grad()
            cls_outputs, lm_loss, cls_loss, kl_loss = model(
                input_ids=b_input_ids,
                attention_mask=b_attention_mask,
                lm_labels=b_lm_labels,
                cls_labels=b_cls_labels,
                teacher_logits=teacher_logits
            )

            total_loss = (lm_loss + cls_loss + kl_loss) / args.accumulation_steps
            total_loss.backward()

            total_train_lm_loss += lm_loss.item()
            total_train_cls_loss += cls_loss.item()
            total_train_kl_loss += kl_loss.item()

            if (num_step % args.accumulation_steps == 0) or (num_step == total_steps):
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

                if num_step >= args.eval_delay and ((num_step % args.logging_steps == 0) or (num_step == total_steps)):
                    print(f"Running Validation at step {num_step}...")

                    model.eval()

                    total_eval_lm_loss = 0
                    total_eval_cls_loss = 0
                    total_eval_kl_loss = 0

                    eval_cls_preds = []
                    eval_cls_labels = []
                    eval_gen_preds = []
                    eval_gen_labels = []

                    for eval_step, batch in tqdm(enumerate(validation_dataloader), total=len(validation_dataloader), leave=False):
                        batch = {k: v for k, v in batch.items()}

                        b_input_ids = batch.get("input_ids")
                        b_lm_labels = batch.get("labels")
                        b_attention_mask = batch.get("attention_mask")
                        b_cls_labels = batch.get("student_cls_labels")
                        
                        b_teacher_input_ids = batch.get("teacher_input_ids")
                        b_teacher_masks = batch.get("teacher_attention_mask")
                        b_teacher_cls = batch.get("teacher_cls_labels")

                        with torch.no_grad():
                            teacher_logits = None
                            if args.teacher_name_or_path:
                                teacher_outputs = teacher_model(
                                    input_ids=b_teacher_input_ids,
                                    attention_mask=b_teacher_masks,
                                    labels=b_teacher_cls
                                )
                                teacher_logits = teacher_outputs.logits

                            cls_outputs, lm_loss, cls_loss, kl_loss = model(
                                input_ids=b_input_ids,
                                attention_mask=b_attention_mask,
                                lm_labels=b_lm_labels,
                                cls_labels=b_cls_labels,
                                teacher_logits=teacher_logits
                            )

                            gen_preds = model.generate_expl(
                                input_ids=b_input_ids,
                                attention_mask=b_attention_mask,
                                num_beams=4,
                                do_sample=True,
                                top_p=0.92,
                                max_new_tokens=50
                            )

                        cls_preds = cls_outputs.argmax(dim=-1).cpu().numpy()
                        cls_labels = b_teacher_cls.cpu().numpy()

                        gen_preds = decoder_tokenizer.batch_decode(gen_preds, skip_special_tokens=True)
                        gen_labels = b_lm_labels.detach().clone()
                        gen_labels[gen_labels == -100] = decoder_tokenizer.pad_token_id
                        gen_labels = decoder_tokenizer.batch_decode(gen_labels, skip_special_tokens=True)

                        # Conditional Decoding Constraint
                        gen_preds = ["none" if cls_pred == 0 else expl for cls_pred, expl in zip(cls_preds, gen_preds)]

                        eval_cls_preds.extend(cls_preds)
                        eval_cls_labels.extend(cls_labels)
                        eval_gen_preds.extend([text.split('SEP>')[-1].strip() for text in gen_preds])
                        eval_gen_labels.extend([text.split('SEP>')[-1].strip() for text in gen_labels])

                        total_eval_lm_loss += lm_loss.item()
                        total_eval_cls_loss += cls_loss.item()   
                        total_eval_kl_loss += kl_loss.item()

                    print("gen_preds: ", gen_preds)
                    print("gen_labels: ", gen_labels)
                    acc, f1 = compute_classification_scores(eval_cls_labels, eval_cls_preds)
                    bleu, rouge, meteor, bertscore = compute_generation_scores(eval_gen_labels, eval_gen_preds)

                    print()
                    print("  Average valid LM: {0:.4f}".format(total_eval_lm_loss/len(validation_dataloader)))
                    print("  Average valid CLS: {0:.4f}".format(total_eval_cls_loss/len(validation_dataloader)))
                    print("  Average valid KL: {0:.4f}".format(total_eval_kl_loss/len(validation_dataloader)))
                    
                    print(f"Epoch {epoch_i + 1} step {num_step} classification evaluations: Acc: {acc}, F1: {f1}")
                    print(f"Epoch {epoch_i + 1} step {num_step} generation evaluations: BLEU-4: {bleu}, ROUGE-L: {rouge}, METEOR: {meteor}, BERTSCORE: {bertscore}")

                    # Record all statistics at this epoch
                    training_stats.append({
                        'Step': num_step,
                        'Best result': best_result,
                        'Avg train LM loss': total_train_lm_loss / (num_step+1),
                        'Avg train CLS loss': total_train_cls_loss / (num_step+1),
                        'Avg train KL loss': total_train_kl_loss / (num_step+1),
                        'Avg valid LM loss': total_eval_lm_loss/len(validation_dataloader),
                        'Avg valid CLS loss': total_eval_cls_loss/len(validation_dataloader),
                        'Avg valid KL loss': total_eval_kl_loss/len(validation_dataloader),
                        'CLS evaluation': f"Acc: {acc}, F1: {f1}",
                        'Generation evaluation': f"BLEU-4: {bleu}, ROUGE-L: {rouge}, METEOR: {meteor}, BERTSCORE: {bertscore}",
                    })

                    if total_eval_lm_loss/len(validation_dataloader) < best_result:
                        print(f"New best checkpoint at Epoch {epoch_i + 1}, Train_step {num_step}")
                        best_result = total_eval_lm_loss/len(validation_dataloader)
                        training_stats[-1]["Best result"] = best_result

                        model.save_checkpoint(os.path.join(args.output_dir, "best_ckpt"), is_best=True)
                    model.save_checkpoint(args.output_dir, is_best=False,
                                          optimizer=optimizer, scheduler=scheduler, training_stats=training_stats)
                    print("Successfully saved checkpoint.")

            train_loop.set_postfix(loss_lm=round(total_train_lm_loss / (num_step+1), 4),
                                   loss_cls=round(total_train_cls_loss / (num_step+1), 4),
                                   loss_kl=round(total_train_kl_loss / (num_step+1), 4))

    print("")
    print("Training complete!")
    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

    with open(os.path.join(args.output_dir, "training_model_stats.json"), "w") as file:
        json.dump(training_stats, file)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to the pre-trained model")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the model checkpoints")
    parser.add_argument("--teacher_name_or_path", type=str, default=None, help="Path to the teacher model")
    parser.add_argument("--text_column_num", type=int, default=1, help="Column number for text input")
    parser.add_argument('--num_epochs', type=int, default=10, help="Number of epochs")
    parser.add_argument("--train_batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--valid_batch_size", type=int, default=32, help="Validation batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument('--warmup_steps', type=int, default=100)
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument("--logging_steps", type=int, default=500, help="Number of steps between logging")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--resume_training", action="store_true", help="Resume training from checkpoint")
    parser.add_argument('--eval_delay', type=int, default=0)
    args = parser.parse_args()

    args.train_file = f"data/{args.dataset_name}_train.csv"
    args.valid_file = f"data/{args.dataset_name}_valid.csv"
    main(args)
