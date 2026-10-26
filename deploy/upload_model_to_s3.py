"""
Upload the trained model folder (models/model) to an S3 bucket.

    python deploy/upload_model_to_s3.py --bucket my-news-classifier-bucket

Afterwards the API can load it with:
    MODEL_S3_URI=s3://my-news-classifier-bucket/news-classifier/model

Credentials: boto3 reads them automatically from `aws configure`
(~/.aws/credentials) or from an IAM role on EC2. Never hard-code keys!
"""
import argparse
from pathlib import Path

import boto3

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "model"


def upload_folder(model_dir: Path, bucket: str, prefix: str) -> str:
    if not (model_dir / "config.json").exists():
        raise FileNotFoundError(f"No trained model in {model_dir}. Run: python -m src.train")

    s3 = boto3.client("s3")
    for path in sorted(model_dir.rglob("*")):
        if path.is_file():
            key = f"{prefix}/{path.relative_to(model_dir).as_posix()}"
            size_mb = path.stat().st_size / 1e6
            print(f"uploading {path.name:<28} {size_mb:7.1f} MB -> s3://{bucket}/{key}")
            s3.upload_file(str(path), bucket, key)  # handles big files (multipart) automatically

    uri = f"s3://{bucket}/{prefix}"
    print(f"\nDone. Start the API with:  MODEL_S3_URI={uri}")
    return uri


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Upload models/model to S3")
    p.add_argument("--bucket", required=True, help="S3 bucket name (must already exist)")
    p.add_argument("--prefix", default="news-classifier/model", help="folder inside the bucket")
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    a = p.parse_args()
    upload_folder(a.model_dir, a.bucket, a.prefix.strip("/"))
