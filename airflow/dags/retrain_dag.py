"""
Airflow DAG: retrain the news classifier every week and deploy it ONLY if it is better.

    load_data  ->  train  ->  evaluate  ->  deploy_if_better

A DAG (Directed Acyclic Graph) is a list of tasks plus the arrows between them.
Airflow runs each task in order, on a schedule, retries failures and shows
everything in a web UI. This file is NOT executed in this project (Airflow is
not installed). It shows how the pipeline would be automated in production.

Data is passed between tasks via XCom (Airflow's small key-value store):
the @task functions simply return values, Airflow stores and forwards them.
"""
from datetime import datetime, timedelta

from airflow.decorators import dag, task

PROJECT_DIR = "/opt/airflow/news-classifier"   # where the repo is checked out on the Airflow worker
S3_BUCKET = "news-classifier-<yourname>-2026"
S3_PREFIX = "news-classifier/model"

default_args = {
    "owner": "ml-team",
    "retries": 1,                          # retry a failed task once ...
    "retry_delay": timedelta(minutes=10),  # ... after 10 minutes
}


@dag(
    dag_id="news_classifier_weekly_retrain",
    schedule="@weekly",                    # every Sunday at midnight
    start_date=datetime(2026, 1, 1),
    catchup=False,                         # don't run all missed weeks since start_date
    default_args=default_args,
    tags=["ml", "nlp", "retraining"],
)
def retrain_pipeline():

    @task
    def load_data() -> int:
        """Download (or refresh) the dataset and check that it looks sane."""
        import sys
        sys.path.insert(0, PROJECT_DIR)
        from datasets import load_dataset
        from src.config import DATASET_ID

        ds = load_dataset(DATASET_ID)
        n = len(ds["train"])
        if n < 1000:                       # simple data-quality gate
            raise ValueError(f"Only {n} training rows - something is wrong with the data")
        return n

    @task
    def train(n_rows: int) -> str:
        """Fine-tune a new model into a NEW folder (the live model is not touched)."""
        import subprocess
        out_dir = f"{PROJECT_DIR}/models/candidate"
        subprocess.run(
            ["python", "-m", "src.train", "--preset", "full",
             "--run-name", "weekly-retrain", "--output-dir", out_dir],
            cwd=PROJECT_DIR, check=True,
        )
        return out_dir

    @task
    def evaluate(candidate_dir: str) -> dict:
        """Read the test metrics that train.py wrote to reports/metrics.json."""
        import json
        with open(f"{PROJECT_DIR}/reports/metrics.json") as f:
            metrics = json.load(f)
        return {"candidate_dir": candidate_dir, "macro_f1": metrics["macro_f1"]}

    @task
    def deploy_if_better(result: dict) -> str:
        """Champion vs. challenger: only replace the live model if the new one is better."""
        import json
        import boto3

        s3 = boto3.client("s3")
        try:
            obj = s3.get_object(Bucket=S3_BUCKET, Key=f"{S3_PREFIX}/metrics.json")
            current_f1 = json.loads(obj["Body"].read())["macro_f1"]
        except s3.exceptions.NoSuchKey:
            current_f1 = 0.0               # nothing deployed yet

        if result["macro_f1"] <= current_f1:
            return f"Kept current model (new F1 {result['macro_f1']:.4f} <= {current_f1:.4f})"

        import sys
        sys.path.insert(0, PROJECT_DIR)
        from pathlib import Path
        from deploy.upload_model_to_s3 import upload_folder

        upload_folder(Path(result["candidate_dir"]), S3_BUCKET, S3_PREFIX)
        s3.put_object(Bucket=S3_BUCKET, Key=f"{S3_PREFIX}/metrics.json",
                      Body=json.dumps({"macro_f1": result["macro_f1"]}))
        # Next step: restart the API containers so they pull the new model from S3.
        return f"Deployed new model (F1 {result['macro_f1']:.4f} > {current_f1:.4f})"

    # The arrows of the DAG: Airflow infers them from how the outputs are passed on.
    deploy_if_better(evaluate(train(load_data())))


retrain_pipeline()
