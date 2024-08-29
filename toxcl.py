import json
import os
from ast import literal_eval

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoModelForSeq2SeqLM


id2label = {0: "normal", 1: "hate"}
label2id = {"normal": 0, "hate": 1}

class ToXCL(nn.Module):
    def __init__(self, decoder_model, decoder_tokenizer=None, tg_model=None, tg_tokenizer=None, hidden_size=768, num_labels=2):
        super(ToXCL, self).__init__()
        self.device = decoder_model.device
        self.decoder_model = decoder_model
        self.decoder_tokenizer = decoder_tokenizer
        self.tg_model = tg_model
        self.tg_tokenizer = tg_tokenizer

        self.num_labels = num_labels
        self.classifier = nn.Linear(hidden_size, num_labels).to(self.device)
        self.loss_fct = nn.BCELoss()
        self.kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
        self.activation = nn.Softmax(dim=-1)

    def classify(self, input_ids, attention_mask=None):
        outputs = self.decoder_model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        last_hidden_state = outputs.encoder_last_hidden_state

        cls_token_emb = torch.mean(last_hidden_state, dim=1).squeeze()
        logits = self.classifier(cls_token_emb).squeeze()
        logits = logits.view(-1, self.num_labels)
        return self.activation(logits)

    def generate_tg(self, **kwargs):
        return self.tg_model.generate(**kwargs)

    def generate_expl(self, **kwargs):
        return self.decoder_model.generate(**kwargs)

    def generate_e2e(self, prompts, apply_constraints=True, tg_generation_params=None, explanation_params=None, **kwargs):
        # (1) generate the Target Groups
        tg_prompts = ["summarize: " + p for p in prompts]
        tg_inputs = self.tg_tokenizer(tg_prompts, padding="max_length", truncation=True, max_length=256, return_tensors="pt").to(self.device)
        generated_tg_outputs = self.generate_tg(
            input_ids=tg_inputs["input_ids"],
            attention_mask=tg_inputs["attention_mask"],
            repetition_penalty=1,
            no_repeat_ngram_size=1,
            **tg_generation_params
        )
        decoded_tg_outputs = self.tg_tokenizer.batch_decode(generated_tg_outputs, skip_special_tokens=True)

        # (2) classify + explain
        new_prompts = [f"Target: {tg} Post: {ip}" for tg, ip in zip(decoded_tg_outputs, prompts)]
        input_ids = self.decoder_tokenizer(new_prompts, padding="max_length", truncation=True, max_length=256, return_tensors="pt").to(self.device)

        predictions = self.classify(
            input_ids=input_ids["input_ids"],
            attention_mask=input_ids["attention_mask"]
        ).argmax(dim=-1).cpu().numpy()
        prediction_labels = [id2label[pred] for pred in predictions]

        explainations = self.generate_expl(
            input_ids=input_ids["input_ids"],
            attention_mask=input_ids["attention_mask"],
            **explanation_params
        )
        decoded_explanations = self.decoder_tokenizer.batch_decode(explainations, skip_special_tokens=True)

        # (3) Conditional Decoding Constraint
        if apply_constraints:
            decoded_explanations = ["none" if cls_pred == "normal" else expl for cls_pred, expl in zip(prediction_labels, decoded_explanations)]

        return dict(target_groups=decoded_tg_outputs, detections=prediction_labels, explanations=decoded_explanations)

    def forward(self, input_ids=None, attention_mask=None, lm_labels=None, cls_labels=None, teacher_logits=None, alpha=0.2):
        lm_outputs = self.decoder_model(input_ids, attention_mask=attention_mask, labels=lm_labels)
        lm_loss = lm_outputs.loss
        
        last_hidden_state = lm_outputs.encoder_last_hidden_state
        cls_token_emb = torch.mean(last_hidden_state, dim=1).squeeze()

        cls_logits = self.classifier(cls_token_emb).squeeze().to(self.device)
        cls_logits = cls_logits.view(-1, self.num_labels)
        cls_outputs = self.activation(cls_logits)

        if teacher_logits is not None:
            teacher_output = self.activation(teacher_logits)
            cls_loss = self.loss_fct(cls_outputs, alpha*teacher_output + (1-alpha)*cls_labels.float())
            kl_loss = self.kl_loss(cls_outputs, teacher_output)
        else:
            cls_loss = self.loss_fct(cls_outputs, cls_labels.float())
            kl_loss = torch.tensor(0.0, device=self.device)

        return cls_outputs, lm_loss, cls_loss, kl_loss

    def save_checkpoint(self, output_dir, is_best=False, optimizer=None, scheduler=None, training_stats=None):
        self.decoder_model.save_pretrained(output_dir)
        torch.save(self.classifier.state_dict(), os.path.join(output_dir, "classifier.pt"))
        if not is_best:
            torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
            torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
            with open(os.path.join(output_dir, "training_model_stats.json"), "w") as file:
                json.dump(training_stats, file)

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
