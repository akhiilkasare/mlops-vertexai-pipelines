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
        "", #project_id, 
        "us-central1", #region
        "dataset-bike-share", #bucket_name
        "vertexai_test_dataset", #dataset_id
        "text_classification_data", #table_id
        "", #service_account
        "repo-vertexai", #repository_name
        "text-classification-model", #image_name
        "v1", #model_tag
        "text-classification-model", #model_display_name
        "text-classification-endpoint", #endpoint_display_name
        "", #display_name
        "vertex-custom-training-job-cpu-tensorboard", #display_nmae
        "n1-standard-4",  # ✅ Machine Type
    )
