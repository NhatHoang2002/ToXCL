import csv
import datetime
import json
import os
import random
import time
from argparse import ArgumentParser
from ast import literal_eval

import datasets
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from eval_metrics import (compute_classification_scores,
                          compute_generation_scores)
from torch.utils.data import (DataLoader, Dataset, RandomSampler,
                              SequentialSampler)
from tqdm import tqdm
from transformers import (AdamW, AutoModelForSeq2SeqLM,
                          AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

id2label = {0: "normal", 1: "hate"}
label2id = {"normal": 0, "hate": 1}

def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))

class ToXCL(nn.Module):
    def __init__(self, decoder_model, decoder_tokenizer, teacher_model, hidden_size=768, num_labels=2):
        super(ToXCL, self).__init__()
        self.device = decoder_model.device
        self.decoder_model = decoder_model
        self.teacher_model = teacher_model
        for param in self.teacher_model.parameters():
            param.requires_grad = False

        self.decoder_tokenizer = decoder_tokenizer
        self.num_labels = num_labels
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.loss_fct = nn.BCELoss()
        self.kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
        self.activation = nn.Softmax(dim=-1)

    def get_decoder_model(self):
        return self.decoder_model

    def get_decoder_tokenizer(self):
        return self.decoder_tokenizer
    
    def classify(self, input_ids, attention_mask):
        outputs = self.decoder_model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        last_hidden_state = outputs.encoder_last_hidden_state

        cls_token_emb = torch.mean(last_hidden_state, dim=1).squeeze()
        logits = self.classifier(cls_token_emb).squeeze().to(self.device)
        logits = logits.view(-1, self.num_labels)
        return self.activation(logits)

    def generate(self, **kwargs):
        return self.decoder_model.generate(**kwargs)

    def forward(self, input_ids=None, lm_labels=None, cls_labels=None, attention_mask=None, teacher_input_ids=None, teacher_labels=None, teacher_attention_mask=None, alpha=0.2):
        lm_outputs = self.decoder_model(input_ids, attention_mask=attention_mask, labels=lm_labels)
        
        last_hidden_state = lm_outputs.encoder_last_hidden_state
        cls_token_emb = torch.mean(last_hidden_state, dim=1).squeeze()

        student_logits = self.classifier(cls_token_emb).squeeze().to(self.device)
        student_logits = student_logits.view(-1, self.num_labels)

        with torch.no_grad():
            teacher_outputs = self.teacher_model(teacher_input_ids, labels=teacher_labels, attention_mask=teacher_attention_mask)
            teacher_logits = teacher_outputs.logits
        student_output = self.activation(student_logits)
        teacher_output = self.activation(teacher_logits)

        assert torch.all(torch.isclose(torch.sum(student_output, dim=1), torch.tensor(1.0), rtol=1e-5)) and torch.all(torch.isclose(torch.sum(teacher_output, dim=1), torch.tensor(1.0), rtol=1e-5))
        assert torch.all(student_output >= 0) and torch.all(teacher_output >= 0)

        student_output = self.activation(student_logits)
        cls_loss = self.loss_fct(student_output, alpha*teacher_output + (1-alpha)*cls_labels.float())
        kl_loss = self.kl_loss(student_output, teacher_output)

        return lm_outputs, student_output, cls_loss, kl_loss

    def save_checkpoint(self, output_dir, is_best=False, optimizer=None, scheduler=None, training_stats=None):
        self.decoder_model.save_pretrained(output_dir)
        self.decoder_tokenizer.save_pretrained(output_dir)
        torch.save(self.classifier.state_dict(), os.path.join(output_dir, "classifier.pt"))
        if not is_best:
            torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
            torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
            with open(os.path.join(output_dir, "training_model_stats.json"), "w") as file:
                json.dump(training_stats, file)
        print("Successfully saved checkpoint.")

    def load_checkpoint(self, output_dir, optimizer=None, scheduler=None):
        self.decoder_model = AutoModelForSeq2SeqLM.from_pretrained(output_dir).to(self.device)
        self.classifier.load_state_dict(torch.load(os.path.join(output_dir, "classifier.pt")))
        if optimizer is not None:
            optimizer.load_state_dict(torch.load(os.path.join(output_dir, "optimizer.pt")))
            scheduler.load_state_dict(torch.load(os.path.join(output_dir, "scheduler.pt")))
        print("Successfully loaded checkpoint.")


class ToxclDataset(Dataset):
    def __init__(self, data):
        self.inputs = ["summarize: " + doc for doc in data["document"]]
        self.outputs = data["summary"]

        encoded_labels = [label2id[i] for i in data["label"]]
        self.student_cls_labels = [[1,0] if int(i)==0 else [0,1] for i in encoded_labels]
        self.teacher_cls_labels = [int(i) for i in encoded_labels]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        output = self.outputs[idx]
        try:    # SBIC dataset
            output = literal_eval(output)
        except: # IHC datset
            output = [output]
        label = np.random.choice(output)
        return dict(
            document=self.inputs[idx],
            label=label,
            student_cls_labels=self.student_cls_labels[idx],
            teacher_cls_labels=self.teacher_cls_labels[idx]
        )


def main(args):

    # (0) Initialization
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    epochs = 2
    max_length = 256
    warmup_steps = 100

    decoder_tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    decoder_model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path).to(device)

    teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher_name_or_path)
    teacher_model = AutoModelForSequenceClassification.from_pretrained(args.teacher_name_or_path)

    # (1.1) Read data
    train_data = []
    valid_data = []

    with open(args.train_file) as file:
        csvreader = csv.reader(file)
        _ = next(csvreader)
        for row in csvreader:
            train_data.append({
                "document": row[args.text_column_num].strip(),
                "label": row[2].strip(),
                "summary": row[3].strip(),
            })
    with open(args.valid_file) as file:
        csvreader = csv.reader(file)
        _ = next(csvreader)
        for row in csvreader:
            valid_data.append({
                "document": row[args.text_column_num].strip(),
                "label": row[2].strip(),
                "summary": row[3].strip(),
            })

    train_data = datasets.Dataset.from_pandas(pd.DataFrame(data=train_data))
    valid_data = datasets.Dataset.from_pandas(pd.DataFrame(data=valid_data))

    # (1.2) Tokenize data
    def collate_fn(batch):
        input_texts = [item["document"] for item in batch]
        label_texts = [item["label"] for item in batch]

        new_batch = decoder_tokenizer(input_texts, max_length=max_length, padding="max_length", return_tensors="pt", truncation=True)
        labels = decoder_tokenizer(label_texts, max_length=max_length, padding="max_length", return_tensors="pt", truncation=True).input_ids
        labels[labels == decoder_tokenizer.pad_token_id] = -100
        new_batch["labels"] = labels

        teacher_inputs = teacher_tokenizer(input_texts, max_length=max_length, padding="max_length", return_tensors="pt", truncation=True)
        new_batch["teacher_input_ids"] = teacher_inputs["input_ids"]
        new_batch["teacher_attention_mask"] = teacher_inputs["attention_mask"]

        new_batch["student_cls_labels"] = torch.as_tensor([item["student_cls_labels"] for item in batch])
        new_batch["teacher_cls_labels"] = torch.as_tensor([item["teacher_cls_labels"] for item in batch])

        return new_batch

    # (1.3) Initialize datasets
    train_dataset = ToxclDataset(train_data)
    train_dataloader = DataLoader(
                train_dataset,                          # The training samples.
                sampler=RandomSampler(train_dataset),   # Select batches randomly
                batch_size=args.train_batch_size,       # Trains with this batch size.
                collate_fn=collate_fn,
                num_workers=8
            )

    valid_dataset = ToxclDataset(valid_data)
    validation_dataloader = DataLoader(
                valid_dataset,                              # The validation samples.
                sampler=SequentialSampler(valid_dataset),   # Pull out batches sequentially.
                batch_size=args.valid_batch_size,           # Evaluate with this batch size.
                collate_fn=collate_fn,
                num_workers=8
            )

    print('{:>5,} training samples'.format(len(train_data)))
    print('{:>5,} validation samples'.format(len(valid_data)))

    # (2) Initialize model, optimizer, scheduler
    model = ToXCL(decoder_model, decoder_tokenizer, teacher_model).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)

    total_steps = (len(train_dataloader) * epochs) // args.accumulation_steps
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    
    # (3) Train ToXCL
    total_t0 = time.time()

    training_stats = []
    best_result = 100   # lm_loss by default
    num_step = 0
    skipped_steps = 0

    if args.resume_training:
        model.load_checkpoint(args.output_dir, optimizer, scheduler)
        with open(os.path.join(args.output_dir, "training_model_stats.json"), "r") as file:
            training_stats = json.load(file)
        best_result = training_stats[-1]['Best result']
        skipped_steps = training_stats[-1]['Step']
        num_step = skipped_steps

    best_checkpoint = model
    for epoch_i in range(0, epochs):

        # ========================================
        #               Training
        # ========================================

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        t0 = time.time()

        total_train_lm_loss = 0
        total_train_cls_loss = 0
        total_train_kl_loss = 0

        model.train()

        train_loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader), leave=True)
        for train_step, batch in train_loop:
            train_loop.update()
            num_step += 1
            # if num_step <= skipped_steps: continue

            b_input_ids = batch.get("input_ids").to(device)
            b_lm_labels = batch.get("labels").to(device)
            b_attention_mask = batch.get("attention_mask").to(device)
            b_cls_labels = batch.get("student_cls_labels").to(device)
            
            b_teacher_input_ids = batch.get("teacher_input_ids").to(device)
            b_teacher_masks = batch.get("teacher_attention_mask").to(device)
            b_teacher_cls = batch.get("teacher_cls_labels").to(device)

            model.zero_grad()
            lm_outputs, cls_outputs, cls_loss, kl_loss = model(b_input_ids, 
                            lm_labels=b_lm_labels, cls_labels = b_cls_labels, 
                            attention_mask = b_attention_mask, 
                            teacher_input_ids=b_teacher_input_ids, 
                            teacher_labels=b_teacher_cls, 
                            teacher_attention_mask=b_teacher_masks)
            
            if args.no_teacher:
                kl_loss = torch.zeros_like(kl_loss)

            lm_loss = lm_outputs[0]

            batch_loss_lm = lm_loss.item()
            batch_loss_cls = cls_loss.item()
            batch_loss_kl = kl_loss.item()

            total_train_lm_loss += batch_loss_lm
            total_train_cls_loss += batch_loss_cls
            total_train_kl_loss += batch_loss_kl

            overall_loss = (lm_loss + 10 * cls_loss + 10 * kl_loss) / args.accumulation_steps
            overall_loss.backward()

            if (num_step % args.accumulation_steps == 0) or (num_step == total_steps):
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

                if num_step >= args.eval_delay and ((num_step % args.sample_every == 0) or (num_step == total_steps)):

                    # Calculate the average loss over all of the batches.
                    avg_train_loss_lm = total_train_lm_loss / (train_step + 1)   
                    avg_train_loss_cls = total_train_cls_loss / (train_step + 1) 
                    avg_train_loss_kl = total_train_kl_loss / (train_step + 1)
                    
                    # Measure how long this epoch took.
                    training_time = format_time(time.time() - t0)

                    print("")
                    print("  Average total_train_cls_loss: {0:.7f}".format(avg_train_loss_cls))
                    print("  Average total_train_lm_loss: {0:.7f}".format(avg_train_loss_lm))
                    print("  Average total_train_kl_loss: {0:.7f}".format(avg_train_loss_kl))
                    print("  Training epoch took: {:}".format(training_time))
                    
                    # ========================================
                    #               Validation
                    # ========================================
                    
                    print(f"Running Validation at step {num_step}...")

                    eval_t0 = time.time()

                    model.eval()

                    total_eval_lm_loss = 0
                    total_eval_cls_loss = 0
                    total_eval_kl_loss = 0

                    epoch_cls_ground_truth = []
                    epoch_generation_ground_truth = []
                    epoch_cls_generated = []
                    epoch_generation_generated = []

                    eval_loop = tqdm(enumerate(validation_dataloader), total=len(validation_dataloader), leave=True)
                    for eval_step, batch in eval_loop:
                        eval_loop.update()
                    
                        b_input_ids = batch.get("input_ids").to(device)
                        b_lm_labels = batch.get("labels").to(device)
                        b_attention_mask = batch.get("attention_mask").to(device)
                        b_cls_labels = batch.get("student_cls_labels").to(device)
                        
                        b_teacher_input_ids = batch.get("teacher_input_ids").to(device)
                        b_teacher_masks = batch.get("teacher_attention_mask").to(device)
                        b_teacher_cls = batch.get("teacher_cls_labels").to(device)

                        with torch.no_grad():
                            lm_outputs, cls_outputs, cls_loss, kl_loss = model(input_ids=b_input_ids, 
                                                                            lm_labels=b_lm_labels,
                                                                            cls_labels=b_cls_labels, 
                                                                            attention_mask=b_attention_mask, 
                                                                            teacher_input_ids=b_teacher_input_ids, 
                                                                            teacher_labels=b_teacher_cls, 
                                                                            teacher_attention_mask=b_teacher_masks)
                            cls_outputs = model.classify(input_ids=b_input_ids, attention_mask=b_attention_mask)

                            generated_cls = cls_outputs.tolist()
                            assert len(generated_cls[0]) == 2

                            generated_cls = [np.argmax(ele) for ele in generated_cls]
                            epoch_cls_generated.extend(generated_cls)
                            assert len(b_input_ids) == len(generated_cls)

                            generated_lm_texts = model.generate(
                                input_ids=b_input_ids, 
                                attention_mask=b_attention_mask, 
                                num_beams=4,
                                do_sample=True,
                                top_p=0.92,
                                top_k=0,
                                max_new_tokens=50
                            )
                            generated_lm_texts = decoder_tokenizer.batch_decode(generated_lm_texts, skip_special_tokens=True)

                            # Conditional Decoding Constraint
                            for idx in range(len(b_input_ids)):
                                if generated_cls[idx] == 0:
                                    generated_lm_texts[idx] = "none"

                            epoch_generation_generated.extend(generated_lm_texts)

                        epoch_cls_ground_truth.extend(b_teacher_cls.tolist())
                        batch_lm_labels = b_lm_labels.detach().clone()
                        batch_lm_labels[batch_lm_labels == -100] = decoder_tokenizer.pad_token_id
                        batch_lm_labels = decoder_tokenizer.batch_decode(batch_lm_labels, skip_special_tokens=True)

                        epoch_generation_ground_truth.extend(batch_lm_labels)

                        batch_loss_lm = lm_outputs.loss.item()
                        batch_loss_cls = cls_loss.item()
                        batch_loss_kl = kl_loss.item()

                        total_eval_lm_loss += batch_loss_lm
                        total_eval_cls_loss += batch_loss_cls   
                        total_eval_kl_loss += batch_loss_kl

                        eval_loop.set_postfix(loss_lm=round(total_eval_lm_loss / (eval_step+1), 5),
                                            loss_cls=round(total_eval_cls_loss / (eval_step+1), 5),
                                            loss_kl=round(total_eval_kl_loss / (eval_step+1), 5))

                    print("generated_lm_texts: ", generated_lm_texts)
                    print("groundtruth_lm_texts: ", batch_lm_labels)
                    generation_scores = compute_generation_scores(epoch_generation_ground_truth, epoch_generation_generated)
                    bleu = generation_scores[0]
                    rouge = generation_scores[1] 
                    meteor = generation_scores[2]
                    bertscore = generation_scores[3]
                    acc, f1 = compute_classification_scores(epoch_cls_ground_truth, epoch_cls_generated)

                    print()
                    print("  Average valid LM: {0:.7f}".format(total_eval_lm_loss/len(validation_dataloader)))
                    print("  Average valid CLS: {0:.7f}".format(total_eval_cls_loss/len(validation_dataloader)))
                    print("  Average valid KL: {0:.7f}".format(total_eval_kl_loss/len(validation_dataloader)))
                    
                    print(f"Epoch {epoch_i + 1} step {num_step} generation evaluations: BLEU-4: {bleu}, ROUGE-L: {rouge}, METEOR: {meteor}, BERTSCORE: {bertscore}")
                    print(f"Epoch {epoch_i + 1} step {num_step} classification evaluations: Acc: {acc}, F1: {f1}")

                    validation_time = format_time(time.time() - eval_t0)   
                    print(f"Evaluation time: {validation_time}") 

                    # Record all statistics at this epoch
                    training_stats.append({
                        'Step': num_step,
                        'Best result': best_result,
                        'Avg train LM loss': avg_train_loss_lm,
                        'Avg train CLS loss': avg_train_loss_cls,
                        'Avg valid KL loss': avg_train_loss_kl,
                        'Avg valid LM loss': total_eval_lm_loss/len(validation_dataloader),
                        'Avg valid CLS loss': total_eval_cls_loss/len(validation_dataloader),
                        'Avg valid KL loss': total_eval_kl_loss/len(validation_dataloader),
                        'Generation evaluation': f"BLEU-4: {bleu}, ROUGE-L: {rouge}, METEOR: {meteor}, BERTSCORE: {bertscore}",
                        'CLS evaluation': f"Acc: {acc}, F1: {f1}",
                        'Validation Time': validation_time
                    })

                    if total_eval_lm_loss/len(validation_dataloader) < best_result:
                        print(f"New best checkpoint at Epoch {epoch_i + 1}, Train_step {num_step}")
                        best_result = total_eval_lm_loss/len(validation_dataloader)
                        training_stats[-1]["Best result"] = best_result

                        best_checkpoint = model
                        best_checkpoint.save_checkpoint(os.path.join(args.output_dir, "best_ckpt"), is_best=True)

                    model.save_checkpoint(args.output_dir, is_best=False,
                                          optimizer=optimizer, scheduler=scheduler, training_stats=training_stats)

            train_loop.set_postfix(loss_lm=round(total_train_lm_loss / (train_step+1), 5),
                                   loss_cls=round(total_train_cls_loss / (train_step+1), 5),
                                   loss_kl=round(total_train_kl_loss / (train_step+1), 5))

    print("")
    print("Training complete!")
    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

    with open(os.path.join(args.output_dir, "training_model_stats.json"), "w") as file:
        json.dump(training_stats, file)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--model_name_or_path', default='google/flan-t5-base')
    parser.add_argument('--teacher_name_or_path')
    parser.add_argument('--output_dir')
    parser.add_argument('--dataset_name')
    parser.add_argument('--text_column_num', type=int, default=1)
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--resume_training', action='store_true')
    parser.add_argument('--no_teacher', action='store_true',
                        help="Used to perform the ablation study, still need to pass the `teacher_name_or_path` argument")
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--valid_batch_size', type=int, default=32)
    parser.add_argument('--accumulation_steps', type=int, default=1)
    parser.add_argument('--sample_every', type=int, default=500)
    parser.add_argument('--eval_delay', type=int, default=0)
    args = parser.parse_args()

    args.train_file = f"data/{args.dataset_name}_train.csv"
    args.valid_file = f"data/{args.dataset_name}_valid.csv"
    main(args)
