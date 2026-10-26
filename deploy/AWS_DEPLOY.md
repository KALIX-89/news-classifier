# Deploying the API on AWS (S3 + EC2)

**Goal:** the model lives in **S3** (cloud storage). An **EC2** virtual machine runs our
Docker container, downloads the model from S3 at startup and serves `/predict` on port 8000.

```
laptop ──upload_model_to_s3.py──▶ S3 bucket ◀──download at startup── EC2 (Docker: API :8000) ◀── curl
```

> **Costs:** a free-tier eligible instance (`t3.micro` / `t2.micro`) plus a few hundred MB
> in S3 is free or costs a few cents. **Always stop the instance when you're done (step 7).**
> Region used here: `eu-central-1` (Frankfurt).

---

## 0. One-time setup on your laptop

1. Create an AWS account and an **IAM user** for yourself (don't use the root account day to day).
2. Install the AWS CLI: <https://aws.amazon.com/cli/>
3. Create an access key for your IAM user (IAM → Users → Security credentials), then run:
   ```bash
   aws configure          # enter key, secret, region eu-central-1, output json
   aws sts get-caller-identity   # check: prints your account id
   ```

## 1. Create the S3 bucket and upload the model

Bucket names are **globally unique**, so add something personal:

```bash
aws s3 mb s3://news-classifier-<yourname>-2026 --region eu-central-1
python deploy/upload_model_to_s3.py --bucket news-classifier-<yourname>-2026
aws s3 ls s3://news-classifier-<yourname>-2026/news-classifier/model/
```
The bucket is **private** by default. Keep it that way.

## 2. Create an IAM role so EC2 can read S3 (no keys on the server!)

IAM → Roles → **Create role** → trusted entity *AWS service* → *EC2* →
permission **AmazonS3ReadOnlyAccess** → name it `news-classifier-ec2-role`.

> Why? The EC2 machine receives temporary credentials automatically. Copying access keys
> onto a server is a classic security mistake.

## 3. Launch the EC2 instance

EC2 → **Launch instance**:

| Setting | Value |
|---|---|
| Name | `news-classifier` |
| AMI | **Amazon Linux 2023** |
| Instance type | `t3.micro` (or whatever says *Free tier eligible*) |
| Key pair | create new → download the `.pem` file |
| Network → Security group | create new, see below |
| Storage | **20 GiB** gp3 (the default 8 GiB is too small for Docker + PyTorch) |
| Advanced → IAM instance profile | `news-classifier-ec2-role` |

**Security group** (= the instance's firewall) inbound rules:

| Type | Port | Source | Why |
|---|---|---|---|
| SSH | 22 | **My IP** | so only you can log in |
| Custom TCP | 8000 | My IP (or 0.0.0.0/0 for a public demo) | the API |

## 4. Connect and install Docker

```bash
# Windows PowerShell / macOS / Linux
ssh -i news-classifier.pem ec2-user@<PUBLIC-IP>
```
On the instance:
```bash
sudo dnf update -y
sudo dnf install -y docker git
sudo systemctl enable --now docker
sudo usermod -aG docker ec2-user
exit            # log out and back in so the group change applies
```

`t3.micro` has only 1 GB RAM. Add 2 GB swap so PyTorch doesn't run out of memory:
```bash
sudo dd if=/dev/zero of=/swapfile bs=1M count=2048 && sudo chmod 600 /swapfile
sudo mkswap /swapfile && sudo swapon /swapfile
```

## 5. Build and run the container

```bash
git clone https://github.com/<your-user>/news-classifier.git
cd news-classifier
docker build -t news-classifier .          # takes a few minutes on a micro instance

docker run -d --name api -p 8000:8000 --restart unless-stopped \
  -e MODEL_S3_URI=s3://news-classifier-<yourname>-2026/news-classifier/model \
  -e AWS_DEFAULT_REGION=eu-central-1 \
  news-classifier

docker logs -f api        # wait for "Application startup complete", then Ctrl+C
```

## 6. Test it

From your laptop:
```bash
curl http://<PUBLIC-IP>:8000/health
curl -X POST http://<PUBLIC-IP>:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "Bayern Munich wins the Champions League final"}'
```
(PowerShell: use `curl.exe` and escape the inner quotes: `-d '{\"text\": \"...\"}'`.)
Or open `http://<PUBLIC-IP>:8000/docs` in the browser.

## 7. ⚠️ STOP the instance (avoid costs)

EC2 → Instances → select `news-classifier` → **Instance state → Stop instance**
(or `aws ec2 stop-instances --instance-ids <id>`).

- **Stop** = machine off, disk kept (pennies per month for the 20 GB disk). Note: the public IP changes on restart.
- **Terminate** = everything deleted. Do this when you no longer need it.
- Delete the S3 bucket when finished: `aws s3 rb s3://news-classifier-<yourname>-2026 --force`
- Tip: set a **billing alarm / AWS Budget** (e.g. $5) in the Billing console.

---

### Production next steps (good to mention in interviews)
- Push the image to **ECR** (Elastic Container Registry) in CI, instead of building on the server.
- Run it on **ECS/Fargate** or **SageMaker endpoints** instead of a hand-managed EC2 machine.
- Put an **Application Load Balancer + HTTPS** in front, and add autoscaling.
- Send logs and metrics to **CloudWatch** for monitoring and alerting.
