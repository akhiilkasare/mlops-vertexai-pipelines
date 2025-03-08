import datetime
from typing import NamedTuple
from collections import namedtuple
from google.cloud.aiplatform import pipeline_jobs
from google_cloud_pipeline_components.v1.custom_job import create_custom_training_job_from_component
from kfp import compiler, dsl
from kfp.dsl import component, Dataset, Input, Metrics, Model, Output
from google_cloud_pipeline_components.v1 import custom_job

# ✅ Configuration Componentr
@component
def config_component() -> NamedTuple(
    "ConfigOutputs",
    [
        ("project_id", str),
        ("region", str),
        ("bucket_name", str),
        ("dataset_id", str),
        ("table_id", str),
        ("service_account", str),
        ("repository_name", str),
        ("image_name", str),
        ("model_tag", str),
        ("model_display_name", str),
        ("endpoint_display_name", str),
        ("tensorboard", str),  # ✅ Added TensorBoard Path
    ],
):
    from collections import namedtuple
    outputs = namedtuple(
        "ConfigOutputs",
        [
            "project_id",
            "region",
            "bucket_name",
            "dataset_id",
            "table_id",
            "service_account",
            "repository_name",
            "image_name",
            "model_tag",
            "model_display_name",
            "endpoint_display_name",
            "tensorboard",
        ],
    )

    return outputs(
        "rock-flag-452514-v8",
        "us-central1",
        "dataset-bike-share",
        "vertexai_test_dataset",
        "text_classification_data",
        "1014348685594-compute@developer.gserviceaccount.com",
        "repo-vertexai",
        "text-classification-model",
        "v1",
        "text-classification-model",
        "text-classification-endpoint",
        "projects/1014348685594/locations/us-central1/tensorboards/5353069116850700288"
    )

# ✅ Fetch Data from BigQuery Component
@component(
    packages_to_install=[
        "db-dtypes==1.3.0",
        "google-cloud-bigquery==3.25.0",
        "pandas==2.2.2",
        "pyarrow==17.0.0",
    ],
    base_image="python:3.12",
)
def fetch_data_from_bigquery(
    project_id: str,
    dataset_id: str,
    table_id: str,
    output_dataset: Output[Dataset],
):
    from google.cloud import bigquery
    import datetime

    start_timestamp = (datetime.datetime.now() - datetime.timedelta(days=2)).strftime("%Y-%m-%d %H:%M:%S")
    query = f"""
        SELECT text, label
        FROM `{project_id}.{dataset_id}.{table_id}`
        WHERE timestamp >= '{start_timestamp}'
    """

    client = bigquery.Client(project=project_id)
    df = client.query(query).to_dataframe()
    print(f"Number of rows fetched: {len(df)}")
    df.to_parquet(output_dataset.path)

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
    from pathlib import Path
    import joblib
    import numpy as np
    import pandas as pd
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    from sklearn.preprocessing import LabelEncoder
    from torch.utils.tensorboard import SummaryWriter  # ✅ Import TensorBoard

    device = torch.device("cpu")  # ✅ Force CPU
    print(f"Using device: {device}")

    df = pd.read_parquet(input_dataset.path)
    print("Class Distribution in the Dataset:")
    print(df["label"].value_counts())

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

    # ✅ Define TensorBoard Writer
    log_dir = str(Path(output_model.path) / "logs")
    writer = SummaryWriter(log_dir)  # ✅ TensorBoard logging path

    training_args = TrainingArguments(
        output_dir=str(Path(output_model.path) / "results"),
        num_train_epochs=5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        evaluation_strategy="epoch",
        logging_dir=log_dir,  # ✅ TensorBoard Logs Path
        logging_steps=10,  # ✅ Log every 10 steps
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
    output_metrics.log_metric("precision", precision)
    output_metrics.log_metric("recall", recall)
    output_metrics.log_metric("f1", f1)

    writer.add_scalar("Accuracy", accuracy)  # ✅ Log Accuracy in TensorBoardð
    writer.add_scalar("Precision", precision)  # ✅ Log Precision in TensorBoardð
    writer.add_scalar("recall", recall)  # ✅ Log Recall in TensorBoardð
    writer.add_scalar("f1", f1)  # ✅ Log F1 in TensorBoardð

    model.save_pretrained(output_model.path)
    tokenizer.save_pretrained(output_model.path)
    writer.close()  # ✅ Close TensorBoard Writer
    joblib.dump(label_encoder, f"{output_model.path}/label_encoder.joblib")
    print(f"✅ Training complete. Model saved to {output_model.path}")

# ✅ Create Custom Training Job Using `v1.custom_job.create_custom_training_job_from_component`
CustomTrainingJobOp = custom_job.create_custom_training_job_from_component(
    component_spec=train_model,
    display_name="vertex-custom-training-job-cpu",
    machine_type="n1-standard-4",  # ✅ Runs on CPU
    accelerator_type="",  # ✅ No GPU
    accelerator_count=0,
    timeout="3600s",
    boot_disk_type="pd-ssd",
    boot_disk_size_gb=100,
    restart_job_on_worker_restart=False,
    tensorboard="projects/1014348685594/locations/us-central1/tensorboards/5353069116850700288",
    service_account="1014348685594-compute@developer.gserviceaccount.com",
    base_output_directory="gs://dataset-bike-share/training_outputs/"
)

# ✅ Define the Pipeline
@dsl.pipeline(name="config-fetch-data-and-custom-train-cpu-pipeline-tensorboard")
def fetch_data_pipeline():
    config = config_component()

    fetch_data_task = fetch_data_from_bigquery(
        project_id=config.outputs["project_id"],
        dataset_id=config.outputs["dataset_id"],
        table_id=config.outputs["table_id"],
    )

    # ✅ Run training as a Vertex AI Custom Job (CPU only)
    custom_train_job = CustomTrainingJobOp(
        input_dataset=fetch_data_task.outputs["output_dataset"]
    )

if __name__ == "__main__":
    pipeline_json = "custom_training_cpu_pipeline_tensorboard.json"

    compiler.Compiler().compile(
        pipeline_func=fetch_data_pipeline,
        package_path=pipeline_json,
    )

    job = pipeline_jobs.PipelineJob(
        display_name="fetch-data-and-custom-train-cpu-pipeline-tensorboard",
        template_path=pipeline_json,
        project="rock-flag-452514-v8",
        location="us-central1",
    )

    job.run()
