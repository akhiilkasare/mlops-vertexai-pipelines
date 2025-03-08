from kfp.dsl import component, Output, Dataset

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
    print(f"✅ Number of rows fetched: {len(df)}")
    df.to_parquet(output_dataset.path)
