import pandas as pd
from argparse import ArgumentParser
import datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import csv


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--model_name')
    parser.add_argument('--output_dir')
    parser.add_argument('--dataset_name')
    parser.add_argument('--train_file')
    parser.add_argument('--valid_file')
    parser.add_argument('--test_file')
    parser.add_argument('--text_column_num', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--every_step', type=int, default=500)
    parser.add_argument('--eval_delay', type=int, default=0)
    args = parser.parse_args()

    if args.dataset_name is not None:
        args.train_file = f"data/{args.dataset_name}_train.csv"
        args.valid_file = f"data/{args.dataset_name}_valid.csv"
        args.test_file = args.valid_file if args.dataset_name in ["IHC"] else f"data/{args.dataset_name}_test.csv"

    train_data = []
    valid_data = []
    test_data = []

    with open(args.train_file) as file:
        csvreader = csv.reader(file)
        _ = next(csvreader)
        for row in csvreader:
            train_data.append({
                "text": row[args.text_column_num],
                "label": row[2]
            })

    with open(args.valid_file) as file:
        csvreader = csv.reader(file)
        _ = next(csvreader)
        for row in csvreader:
            valid_data.append({
                "text": row[args.text_column_num],
                "label": row[2]
            })

    with open(args.test_file) as file:
        csvreader = csv.reader(file)
        _ = next(csvreader)
        for row in csvreader:
            test_data.append({
                "text": row[args.text_column_num],
                "label": row[2]
            })

    train_data = datasets.Dataset.from_pandas(pd.DataFrame(data=train_data))
    valid_data = datasets.Dataset.from_pandas(pd.DataFrame(data=valid_data))
    test_data = datasets.Dataset.from_pandas(pd.DataFrame(data=test_data))

    id2label = {0: "normal", 1: "hate"}
    label2id = {"normal": 0, "hate": 1}
    print(label2id)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, max_length=256)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name)

    def preprocess_data(batch):
        encoding = tokenizer(batch["text"], padding="max_length", truncation=True, max_length=256)
        encoding["label"] = [label2id[item] for item in batch["label"]]
        return encoding

    train_data = train_data.map(preprocess_data, batched=True, batch_size=len(train_data))
    valid_data = valid_data.map(preprocess_data, batched=True, batch_size=len(valid_data))
    test_data = test_data.map(preprocess_data, batched=True, batch_size=len(test_data))

    train_data.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    valid_data.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    test_data.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        f1 = f1_score(labels, preds, average='macro')
        precision = precision_score(labels, preds, average='macro')
        recall = recall_score(labels, preds, average='macro')
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=10,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=1,    
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        disable_tqdm=False,
        save_total_limit=2,
        load_best_model_at_end=True,
        warmup_steps=100,
        save_strategy="steps",
        save_steps=args.every_step,
        evaluation_strategy="steps",
        eval_steps=args.every_step,
        eval_delay=args.eval_delay,
        weight_decay=0.01,
        metric_for_best_model="f1",
        logging_steps=args.every_step,
        fp16=False,
        logging_dir=args.output_dir,
        dataloader_num_workers=2,
        report_to="none",
        run_name='bert-classification'
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_data,
        eval_dataset=valid_data,
    )
    trainer.train()
    print(trainer.evaluate(test_data))

"""
python -m train_encoder_arch \
    --model_name bert-base-uncased \
    --output_dir saved/BERT-B_IHC \
    --dataset_name IHC
"""