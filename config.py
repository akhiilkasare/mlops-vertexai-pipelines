from typing import NamedTuple
from kfp.dsl import component
from collections import namedtuple

# ✅ Configuration Component
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
        ("display_name", str),
        ("machine_type", str),



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
            "display_name",
            "machine_type",  # ✅ Added Display Name and Machine Type
        ],
    )

    return outputs(
        "",
        "us-central1",
        "dataset-bike-share",
        "vertexai_test_dataset",
        "text_classification_data",
        "",
        "repo-vertexai",
        "text-classification-model",
        "v1",
        "text-classification-model",
        "text-classification-endpoint",
        "",
        "vertex-custom-training-job-cpu-tensorboard",
        "n1-standard-4",  # ✅ Added Machine Type
    )
