import datetime
from google.cloud.aiplatform import pipeline_jobs
from google_cloud_pipeline_components.v1 import custom_job
from kfp import compiler, dsl
from kfp.dsl import Dataset

# ✅ Import components
from config import config_component
from fetch_data import fetch_data_from_bigquery
from train import train_model

print("✅ Imported all components successfully.")

# ✅ Create Custom Training Job
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
    tensorboard="",
    service_account="",
    base_output_directory=""
)
print("✅ Created CustomTrainingJobOp.")

@dsl.pipeline(name="fetch-data-and-custom-train-pipeline")
def fetch_data_pipeline():
    print("✅ Fetching config...")
    config = config_component()

    print("✅ Running fetch_data_from_bigquery...")
    fetch_data_task = fetch_data_from_bigquery(
        project_id=config.outputs["project_id"],
        dataset_id=config.outputs["dataset_id"],
        table_id=config.outputs["table_id"],
    )

    print("✅ Running CustomTrainingJobOp...")
    CustomTrainingJobOp(input_dataset=fetch_data_task.outputs["output_dataset"])

if __name__ == "__main__":
    print("✅ Compiling pipeline...")
    compiler.Compiler().compile(fetch_data_pipeline, "custom_training_cpu_pipeline.json")

    print("✅ Running pipeline...")
    job = pipeline_jobs.PipelineJob(
        display_name="fetch-data-and-custom-train-pipeline",
        template_path="custom_training_cpu_pipeline.json",
        project="",
        location="",
    )

    print("✅ Submitting job to Vertex AI...")
    job.run()
