from kfp.dsl import component, Input, Output, Dataset, Model, Metrics

# ✅ Training Component (Runs on CPU)
@component(
    packages_to_install=[
        "accelerate==0.34.2",
        "fastparquet==2024.5.0",
        "pandas==2.2.2",
        "pyarrow==17.0.0",
        "scikit-learn==1.5.2",
        "torch==2.4.1",
        "transformers==4.44.2",
        "tensorboard==2.15.0"
    ],
    base_image="python:3.12",
)
def train_model(
    input_dataset: Input[Dataset],
    output_model: Output[Model],
    output_metrics: Output[Metrics],
):
    import joblib
    import numpy as np
    import pandas as pd
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    from sklearn.preprocessing import LabelEncoder
    from torch.utils.tensorboard import SummaryWriter
    from pathlib import Path

    device = torch.device("cpu")  # ✅ Force CPU
    print(f"✅ Using device: {device}")

    df = pd.read_parquet(input_dataset.path)
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(df["label"])
    num_labels = len(np.unique(labels))

    texts = df["text"].tolist()
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, stratify=labels
    )

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=num_labels)
    model.to(device)

    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

    class TextDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels
        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
            return item
        def __len__(self):
            return len(self.labels)

    train_dataset = TextDataset(train_encodings, train_labels)
    val_dataset = TextDataset(val_encodings, val_labels)

    log_dir = str(Path(output_model.path) / "logs")
    writer = SummaryWriter(log_dir)

    training_args = TrainingArguments(
        output_dir=str(Path(output_model.path) / "results"),
        num_train_epochs=5,
        per_device_train_batch_size=32,
        evaluation_strategy="epoch",
        logging_dir=log_dir,
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    trainer.train()

    predictions = trainer.predict(val_dataset)
    preds = predictions.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(val_labels, preds, average="macro")
    accuracy = accuracy_score(val_labels, preds)

    output_metrics.log_metric("accuracy", accuracy)
    writer.add_scalar("Accuracy", accuracy)
    writer.add_scalar("Precision", precision)
    writer.add_scalar("Recall", recall)
    writer.add_scalar("F1", f1)

    model.save_pretrained(output_model.path)
    tokenizer.save_pretrained(output_model.path)
    writer.close()
    joblib.dump(label_encoder, f"{output_model.path}/label_encoder.joblib")

    print(f"✅ Training complete. Model saved to {output_model.path}")
