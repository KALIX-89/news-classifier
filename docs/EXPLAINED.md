# news-classifier : Everything Explained

This document explains **every file and every concept** in the project in plain language,
with everyday analogies. At the end there is an **Interview** section in German.

**The real numbers from this project** (use these, don't round them up):

| | Value |
|---|---|
| Final model | DistilBERT, 20,000 training headlines, 2 epochs, max_len 128, LR 5e-5 |
| Test accuracy | **0.9284** (92.8 % of 7,600 unseen articles correct) |
| Test macro-F1 | **0.9284** |
| Best / worst class | Sports F1 0.9775 / Business F1 0.8949 |
| Biggest confusion | Business → Sci/Tech (156), Sci/Tech → Business (115) |
| FAST experiments (2k samples, 1 epoch) | LR 5e-5: val 0.9040, test 0.8840 · LR 2e-5: val 0.8820, test 0.8840 |
| Training time | ~5 min (FULL) / ~1 min (FAST) on an RTX 4060 laptop GPU |
| Tests | 13 pytest tests, ~5 s |
| Docker image | 1.78 GB (CPU-only PyTorch), ready in ~5 s, ~25 ms per prediction, ~300 MiB RAM, status `healthy` |

---

## Part 1 — The files, one by one

### `src/config.py` — the control panel
All settings live here: dataset name, model name, learning rate, batch size, the label names,
and two **presets** (`fast` for a CPU, `full` for a GPU).
*Analogy:* the settings menu of a game. You change difficulty in one place instead of editing the game code.
*Why it matters:* reproducibility. Every run's settings are also logged to MLflow, so you always know which settings produced which result.

### `src/data.py` — getting the data ready
1. **Download** AG News from Hugging Face (120,000 train + 7,600 test news articles, 4 balanced classes).
2. **Clean** (`clean_text`): fix HTML leftovers like `&amp;` → `&` and `#39;` → `'`. We do *not* do heavy
   cleaning (no stop-word removal, no stemming) because Transformers were pre-trained on normal text and use every word.
3. **Split** (`make_splits`): train and validation come from the original train split (shuffled, no overlap),
   test comes from the original test split.
4. **Tokenize**: turn text into numbers (see *Tokenization* below).
5. **DataLoaders**: serve the data in batches of 16, shuffled for training.

### `src/train.py` — teaching the model
Loads pre-trained DistilBERT, adds a new 4-class "head" and runs the **training loop** by hand
(forward pass → loss → backpropagation → optimizer step → scheduler step). After every epoch it checks
the validation set. At the end it evaluates on the test set, saves the model with `save_pretrained`
and logs everything to MLflow. It detects the hardware automatically: NVIDIA GPU (`cuda`) → Apple GPU (`mps`) → `cpu`.
`--skip-if-exists` means: if a model is already saved, don't train again, just evaluate.

*Note:* the log says `classifier.weight MISSING`. That is **expected**: it's the new 4-class head,
which doesn't exist in the pre-trained model and gets trained by us.

### `src/evaluate.py` — the exam
Runs the model on data it has never seen and computes accuracy, macro-F1, a per-class report and
the confusion matrix (PNG). Everything is saved to `reports/` and logged to MLflow.

### `src/predict.py` — using the model
`NewsClassifier` loads model + tokenizer **once**. `predict(text)` returns
`{"label": "Sports", "confidence": 0.99}`. Confidence = the softmax probability of the chosen class.

### `api/main.py` — the waiter
A FastAPI web service. `GET /health` says "I'm alive". `POST /predict` takes `{"text": "..."}` and
returns label + confidence. The model is loaded **once at startup** (not on every request, which would be
very slow), either from `./models/model` or, if `MODEL_S3_URI` is set, downloaded from S3 first.
If the model can't be loaded, the server still starts, `/health` reports `model_loaded: false`
and `/predict` returns **503 Service Unavailable**. Bad input (empty text, missing field) gets **422**,
checked automatically by pydantic.

### `tests/test_api.py` and `tests/test_data.py` — the quality checks
- API tests use a **fake model** (always answers "Sports" if the text contains "goal"). This tests *our* code
  (routing, validation, status codes) in seconds, without the 250 MB model, which isn't in git and so isn't in CI.
- Data tests use a tiny in-memory dataset. They check cleaning, split sizes, **no overlap between train and validation**
  (no data leakage), reproducibility (same seed → same split) and error handling.

### `Dockerfile`, `.dockerignore`, `docker-compose.yml` — the shipping box
The Dockerfile describes how to build the image: slim Python 3.11, **CPU-only PyTorch** (much smaller than
the GPU version, and inference for one headline is fast on a CPU), the API dependencies, then our code.
The model is **not baked in**: it's mounted as a folder or pulled from S3, so the same image works for every model version.
Runs as a non-root user and has a HEALTHCHECK. `docker-compose.yml` starts the API plus the MLflow UI together.

### `requirements.txt` vs `requirements-api.txt`
The full list (training, MLflow, tests) is for development. The small list is for the Docker image: only what serving needs → smaller, faster, safer image.

### `.github/workflows/ci.yml` — the robot reviewer
On every push to GitHub: install everything, run pytest, then build the Docker image and check that `/health` answers.

### `deploy/upload_model_to_s3.py` and `deploy/AWS_DEPLOY.md`
Upload the model folder to S3 with boto3. The guide walks through: S3 bucket → IAM role → EC2 instance → security group → Docker → `docker run` → curl → **stop the instance**.

### `airflow/dags/retrain_dag.py` — the weekly schedule
`load_data → train → evaluate → deploy_if_better`, every week. The new model is only uploaded if its F1
is **better than the current one** (champion/challenger). The file is a design and is not executed here.

### `mlflow.db` and `mlruns/` (not in git)
MLflow's storage: `mlflow.db` (SQLite) holds the run table (params, metrics, registry), `mlruns/` holds the files (models, PNGs).

---

## Part 2 — The concepts

### NLP & Transformers

**Tokenization**: computers only understand numbers. The tokenizer cuts text into pieces ("tokens") and
gives each one an ID from a fixed vocabulary of ~30,000 tokens. Rare words get split into known pieces:
`"unbelievable" → "un", "##believ", "##able"`. Special tokens `[CLS]` (start) and `[SEP]` (end) are added.
*Analogy:* LEGO. Any word can be built from a fixed set of bricks.
`max_len` = maximum number of tokens (longer texts get cut, shorter ones get **padding**).
The **attention mask** tells the model which tokens are real (1) and which are padding (0).
Our texts average ~38 words, so `max_len=128` covers almost everything, while 64 cuts some texts off. That's one reason FULL is better.

**Embeddings**: each token ID is turned into a list of 768 numbers (a vector) that represents its meaning.
Words with similar meanings get similar vectors. *Analogy:* GPS coordinates for words, where "football" lives near "soccer" and far from "inflation".

**Attention**: each word looks at all the other words in the sentence and decides which ones matter for its
meaning. In "Apple shares fell", attention lets "Apple" look at "shares" and understand it's the company, not the fruit.
*Analogy:* in a meeting, everyone listens most to the people who are relevant to them. A **Transformer** is a stack of
such attention layers (DistilBERT: 6 layers, 66 M parameters, a smaller and faster "distilled" version of BERT that keeps ~97 % of BERT's quality).

**Fine-tuning vs. training from scratch**: DistilBERT was already *pre-trained* on huge amounts of English text
(Wikipedia + books), so it already "knows" English. We only **fine-tune**: add a small classification head and train
everything a little on our 20,000 headlines. *Analogy:* hiring an experienced journalist and teaching them your
newsroom's 4 categories in a day, versus teaching a baby to read. From scratch you'd need millions of examples and
days of GPU time. We needed 5 minutes.

### The training loop

- **Batch**: a small group of examples processed together (we use 16). It's faster than one at a time, and more stable.
- **Forward pass**: push a batch through the model → it outputs 4 scores (**logits**) per headline.
- **Loss** (cross-entropy): one number that says how wrong the predictions are. Confident and right → small; confident and wrong → big.
- **Backpropagation**: calculates for **every one of the 66 M weights** in which direction to nudge it to reduce the loss (the gradient).
  *Analogy:* after a wrong answer, figuring out which part of your reasoning caused the mistake.
- **Optimizer** (AdamW): actually nudges the weights. "W" = weight decay, a small penalty against huge weights that helps against overfitting.
- **Learning rate (LR)**: the step size of each nudge. Too big → you overshoot and training becomes unstable; too small → it learns very slowly.
  For fine-tuning, typical values are 2e-5 to 5e-5. **Warmup + linear decay**: start with tiny steps (the new head is random at first), rise to the full LR during the first 10 % of steps, then shrink linearly to 0.
  *Our result:* with only 125 steps, 5e-5 reached val acc 0.904 vs 0.882 for 2e-5, because the smaller steps hadn't gotten as far yet.
- **Epoch**: one full pass over all training data. FULL = 2 epochs = 2 × 1,250 batches.
- **Gradient clipping**: cap the size of updates so one weird batch can't wreck the model.
- `model.train()` / `model.eval()`: switch **dropout** on/off (dropout randomly ignores some neurons during training → robustness).
  `torch.no_grad()` during evaluation: no gradients needed → faster, less memory.

### Data splits, overfitting, metrics

**Train / validation / test**: *Analogy:* school.
- **Train** = the textbook exercises you learn from.
- **Validation** = practice tests: used during development to compare settings (e.g. which LR) and to see whether you're still improving.
- **Test** = the final exam, used **once** at the very end. If you used it to choose settings, you'd be "cheating" and the score would be too optimistic.
We picked LR 5e-5 because of **validation** accuracy, not test accuracy. Train and validation never overlap (a unit test checks this).

**Overfitting**: the model memorises the training data instead of learning general rules, like a student who memorises answers but fails new questions.
Sign: training loss keeps falling while validation loss rises. *Our run:* train loss 0.37 → 0.17, and val loss also fell 0.277 → 0.241, so **no overfitting** yet after 2 epochs.
Remedies: more data, fewer epochs (early stopping), weight decay, dropout.

**Accuracy**: correct ÷ total. Fine here because the classes are balanced (1,900 test articles each). With imbalanced data
(e.g. 99 % "not fraud") accuracy is misleading: a model that always says "not fraud" gets 99 %.

**Precision** (for class X): of everything the model *called* X, how much really was X? *"When the alarm rings, is there really a fire?"*
**Recall** (for class X): of everything that *really* was X, how much did the model find? *"Of all real fires, how many triggered the alarm?"*
**F1**: harmonic mean of precision and recall, high only if *both* are high.
**Macro-F1**: compute F1 per class, then take the simple average, so every class counts equally, including small ones.
*Our Business class:* precision 0.8999, recall 0.8900. It's the weakest class.

**Confusion matrix**: a table with rows = true class and columns = predicted class. The diagonal = correct.
Off-diagonal cells show *which* mistakes happen. Ours: 156 Business articles predicted as Sci/Tech, 115 the other way round.
That's understandable, because "Google earnings beat expectations" is both. Sports is almost never confused.

### MLOps

**MLOps lifecycle**: DevOps for machine learning. The model is only ~10 % of the work; the rest is getting it reliably into production and keeping it good:
`data → train → evaluate → track/register → package (Docker) → test (CI) → deploy (AWS) → monitor → retrain (Airflow)` → and around again.
This project has one piece for every step.

**MLflow tracking**: a lab notebook that fills itself. Every run stores its **params** (LR, epochs…), **metrics** (val accuracy per epoch, test F1),
and **artifacts** (confusion matrix, model files). In the UI you can **compare runs** side by side. Without it you get
"which settings gave 92 % again?" and Excel chaos.
**MLflow Model Registry**: a version-controlled shelf for models: "news-classifier **version 1**", with stages/aliases like *staging* and *production*,
so everyone knows which model is live and you can roll back.

**REST API**: a standard way for programs to talk over HTTP. The client sends a request (`POST /predict` with JSON) and gets a response (JSON + status code: 200 OK, 422 bad input, 503 not ready).
*Analogy:* a restaurant. The menu is the API docs, you (the client) order, the waiter (API) brings it from the kitchen (model). You never enter the kitchen.
**FastAPI**: a Python web framework. It's fast, validates input automatically with type hints (pydantic), and generates interactive docs at `/docs` (Swagger UI) for free.
**uvicorn**: the server program that actually runs the FastAPI app and listens on port 8000.

**Docker**: solves "it works on my machine".
- **Dockerfile** = the *recipe* (start from Python 3.11, install these packages, copy this code, run this command).
- **Image** = the *frozen meal* cooked from the recipe: a read-only package with OS libraries, Python, packages and code.
- **Container** = the meal *being eaten*: a running instance of an image. You can run many containers from one image.
Unlike a virtual machine, containers share the host's OS kernel, so they're lightweight and start in seconds.
**Volume** (`-v ./models:/app/models`) = a folder from the host shared into the container.
**docker compose**: one YAML file that starts several containers together (here: API + MLflow UI) with one command, `docker compose up`.

**A real problem I hit (good interview story):** in the compose MLflow UI, metrics showed up but the plots and model files gave 404.
The reason: MLflow had stored the artifact location as an absolute **Windows** path (`file:///C:/Users/...`), which doesn't exist inside a **Linux** container.
The fix: mount `./mlruns` normally and, at container start, create a symlink at that Windows-style path pointing to the mounted folder (path set via a gitignored `.env` file).
The production fix would be an MLflow **tracking server with an artifact store in S3**, so artifact paths don't depend on any one machine.

**Docker commands used:**
`docker build -t news-classifier .` (build the image from the Dockerfile) · `docker images` (list images and sizes) ·
`docker run -d --name news-api -p 8000:8000 -v ./models:/app/models:ro news-classifier` (start a container in the background, map the port, mount the model read-only) ·
`docker ps` (running containers + health) · `docker logs news-api` (container output) · `docker stats` (CPU/RAM) · `docker rm -f news-api` (stop + delete the container, the image stays) ·
`docker compose up -d --build` / `docker compose ps` / `docker compose logs mlflow` / `docker compose down`.

**AWS EC2** (Elastic Compute Cloud): rent a virtual computer in Amazon's data centre by the hour. We use a small free-tier `t3.micro`.
**S3** (Simple Storage Service): unlimited cloud storage for files ("objects") in "buckets". It's the standard place for models and datasets: cheap and durable.
**Security group**: the firewall of the EC2 instance. Only explicitly allowed ports are open: 22 (SSH) only from my IP, 8000 for the API.
**IAM role**: lets the EC2 machine read S3 **without storing passwords/keys on it** (it gets temporary credentials automatically).
**Stop vs. terminate**: stop = switch off (keep the disk), terminate = delete. Always stop after a demo → no surprise bill.

**CI/CD**: **Continuous Integration** = every code change is automatically tested and built, so bugs are caught immediately, not in production.
**Continuous Delivery/Deployment** = after passing, it's automatically released (e.g. push image to a registry and deploy).
**GitHub Actions**: GitHub's built-in CI service. A YAML file in `.github/workflows/` says "on push: start a fresh Ubuntu machine, install, run pytest, build Docker".
A green tick means everything passed. *Analogy:* a robot that re-checks your homework every time you change it.

**Airflow & DAGs**: Airflow is a scheduler/orchestrator for data pipelines. A **DAG** (Directed Acyclic Graph) is a set of tasks with arrows showing
the order (*directed*), and no loops (*acyclic*). Airflow runs it on a schedule (`@weekly`), retries failed tasks, and shows logs and history in a UI.
*Analogy:* a recipe with steps where you can't bake before mixing, plus a kitchen timer that starts cooking every Sunday.
Our DAG: `load_data → train → evaluate → deploy_if_better`. **Champion/challenger**: the new model only replaces the live one if it's better.

**Monitoring & data drift**: after deployment, the world changes but the model doesn't. AG News is from ~2004–2005; today's headlines are about
AI, crypto and new politicians the model has never seen. **Data drift** = the input data changes (new words, new topics, longer texts).
**Concept drift** = the meaning changes (e.g. "Twitter" moves from Sci/Tech to Business/World news).
What you'd monitor: latency, error rate, requests per second (ops metrics); **the distribution of predicted labels and of confidence**
(if average confidence drops or suddenly 60 % is "World", something changed); and, with some human-labelled samples, real accuracy.
Tools: Prometheus/Grafana, CloudWatch, Evidently AI. Drift detected → trigger the Airflow retraining.

---

## Part 3 — Interview (Deutsch)

### 1-Minuten-Pitch

> „Ich habe ein End-to-End-MLOps-Projekt gebaut: einen News-Klassifikator, der Schlagzeilen in vier Kategorien einordnet –
> World, Sports, Business und Sci/Tech. Dafür habe ich DistilBERT, ein vortrainiertes Transformer-Modell, mit einer
> selbst geschriebenen PyTorch-Trainingsschleife auf dem AG-News-Datensatz feinjustiert. Auf 7.600 ungesehenen Test-Artikeln
> erreicht das Modell **92,8 % Accuracy und einen Macro-F1 von 0,928** – bei nur fünf Minuten Training auf meiner Laptop-GPU.
> Mir ging es aber vor allem um den MLOps-Teil: Alle Experimente werden mit **MLflow** getrackt und das beste Modell in der
> Model Registry versioniert. Das Modell wird über eine **FastAPI**-REST-API ausgeliefert, in **Docker** verpackt, mit **pytest**
> getestet, und **GitHub Actions** führt bei jedem Push automatisch Tests und Docker-Build aus. Für das Deployment gibt es eine
> Anleitung für **AWS** mit S3 und EC2, und ein **Airflow-DAG** beschreibt, wie das Modell wöchentlich neu trainiert und nur dann
> ausgerollt wird, wenn es besser ist. Damit habe ich den kompletten Lebenszyklus eines ML-Modells einmal selbst gebaut."

### 2-Minuten-Live-Demo (Reihenfolge)

Vorher: `.venv\Scripts\activate`, MLflow UI und API schon gestartet (bzw. `docker compose up`), Tabs offen.

1. **GitHub-Repo / README (15 s)** – *„Hier ist das Repo. Das Architekturdiagramm zeigt den Weg von den Daten über Training und MLflow bis zur API in Docker und AWS."* Kurz auf den grünen CI-Haken zeigen.
2. **Code `src/train.py` (25 s)** – Funktion `train_one_epoch` zeigen: *„Ich habe bewusst keinen HF-Trainer benutzt, sondern die Schleife selbst geschrieben: Forward Pass, Loss, Backpropagation, Optimizer-Step, Scheduler mit Warmup."*
3. **MLflow UI, http://localhost:5000 (30 s)** – Die zwei FAST-Runs anhaken → *Compare*: *„Nur die Learning Rate unterscheidet sich; 5e-5 hat eine höhere Validation Accuracy, 0,904 gegenüber 0,882."* Dann den FULL-Run öffnen → Confusion Matrix im Artifacts-Tab: *„Die meisten Fehler sind Business gegen Sci/Tech – logisch, z. B. bei Nachrichten über Tech-Firmen."* Kurz den *Models*-Tab: Version 1 in der Registry.
4. **API, http://localhost:8000/docs (30 s)** – `POST /predict` → *Try it out* → englische Schlagzeile (das Modell ist auf Englisch trainiert): `"Bayern Munich beats Real Madrid in the Champions League"` → Sports (0.99). Dann eine knifflige: `"Microsoft shares jump after strong cloud earnings"` → Sci/Tech mit nur **0.57** Confidence: *„Eigentlich eher Business – genau die Grenze Business/Sci-Tech aus der Confusion Matrix. Die niedrige Confidence zeigt, dass das Modell unsicher ist; in Produktion könnte man solche Fälle an einen Menschen weiterleiten."* Leeren Text senden → 422: *„Die Eingabe wird automatisch validiert."*
5. **Terminal (20 s)** – `pytest -v` → 13 grüne Tests: *„Die API-Tests laufen mit einem Fake-Modell, damit sie in der CI ohne das 250-MB-Modell funktionieren."* Optional `docker ps` zeigen.
6. **Abschluss (10 s)** – *„Nächste Schritte wären Monitoring auf Data Drift und Deployment über ECR und ECS statt manuell auf EC2."*

### Die 15 wahrscheinlichsten Fragen

1. **Warum DistilBERT und nicht BERT oder ein LLM?**
   DistilBERT ist ca. 40 % kleiner und 60 % schneller als BERT bei ~97 % der Qualität. Für 4 Klassen reicht das völlig; es läuft auf CPU im Docker-Container. Ein LLM wäre für diese Aufgabe teuer, langsam und überdimensioniert.

2. **Was ist der Unterschied zwischen Fine-Tuning und Training von Grund auf?**
   Das Modell kennt Englisch schon aus dem Pre-Training. Ich habe nur einen Klassifikationskopf ergänzt und alles kurz auf meinen Daten nachtrainiert. Dafür reichen 20.000 Beispiele und 5 Minuten, statt Millionen Texten und Tagen auf vielen GPUs.

3. **Warum eine eigene Trainingsschleife statt des HF Trainers?**
   Um jeden Schritt zu verstehen und zu kontrollieren: Forward Pass, Loss, `backward()`, Optimizer, Scheduler, Gradient Clipping. In Produktion würde ich je nach Team auch den Trainer oder Lightning nutzen.

4. **Wie hast du die Daten aufgeteilt und warum?**
   Train und Validation aus dem originalen Train-Split, ohne Überschneidung (per Test geprüft), Test aus dem originalen Test-Split. Validation zum Vergleichen von Einstellungen, Test nur einmal am Ende, sonst wäre das Ergebnis zu optimistisch.

5. **Warum Macro-F1 und nicht nur Accuracy?**
   F1 berücksichtigt Precision und Recall; Macro gewichtet jede Klasse gleich. Hier sind die Klassen balanciert, daher sind beide Werte fast gleich (0,9284). Bei unbalancierten Daten wäre Accuracy irreführend.

6. **Wo macht das Modell die meisten Fehler?**
   Business ↔ Sci/Tech: 156 Business-Artikel als Sci/Tech, 115 umgekehrt. Nachrichten über Tech-Unternehmen gehören oft zu beidem. Sports ist am einfachsten (F1 0,978).

7. **Gibt es Overfitting?**
   Nein, nach 2 Epochen sinkt nicht nur der Train-Loss (0,37 → 0,17), sondern auch der Validation-Loss (0,277 → 0,241). Dagegen helfen außerdem Weight Decay, Dropout und Early Stopping.

8. **Wie hast du die Learning Rate gewählt?**
   Zwei Experimente mit 5e-5 und 2e-5, in MLflow verglichen. 5e-5 hatte die bessere Validation Accuracy (0,904 vs. 0,882), weil bei nur 125 Schritten die größeren Schritte schneller lernen. Mit Warmup, damit der zufällig initialisierte Kopf am Anfang nichts kaputt macht.

9. **Wofür MLflow?**
   Jeder Run speichert automatisch Parameter, Metriken und Artefakte, sodass Experimente reproduzierbar und vergleichbar sind. In der Model Registry wird das Modell versioniert, damit klar ist, welches Modell live ist und man zurückrollen kann.

10. **Was ist der Unterschied zwischen Image und Container?**
    Das Dockerfile ist das Rezept, das Image das daraus gebaute, unveränderliche Paket, und der Container eine laufende Instanz davon. Aus einem Image kann man viele Container starten.

11. **Warum ist das Modell nicht im Docker-Image?**
    So ist das Image unabhängig von der Modellversion: Das Modell kommt per Volume oder beim Start aus S3 (`MODEL_S3_URI`). Neues Modell heißt dann: kein neues Image, nur Neustart. Außerdem nutze ich CPU-PyTorch, damit das Image klein bleibt.

12. **Wie testest du die API, ohne das echte Modell?**
    Mit einem Fake-Modell, das dieselbe `predict()`-Schnittstelle hat. So teste ich meinen Code (Routing, Validierung, Statuscodes 200/422/503) in Sekunden, auch in der CI, wo das Modell nicht liegt. Die Modellqualität prüft separat `evaluate.py`.

13. **Was macht deine CI-Pipeline?**
    GitHub Actions startet bei jedem Push eine frische Ubuntu-Maschine, installiert die Abhängigkeiten, führt pytest aus und baut danach das Docker-Image samt Health-Check. CD wäre der nächste Schritt: Image in ECR pushen und automatisch deployen.

14. **Wie würdest du das Modell in Produktion überwachen?**
    Technisch: Latenz, Fehlerrate, Requests. Fachlich: die Verteilung der vorhergesagten Klassen und der Confidence. Wenn die sich verschiebt, deutet das auf **Data Drift** hin, z. B. neue Themen wie KI, die es in den Daten von 2004 nicht gab. Dazu stichprobenartig gelabelte Daten für echte Accuracy. Bei Drift wird über Airflow neu trainiert.

15. **Was würdest du als Nächstes verbessern?**
    Mehr Daten bzw. das ganze Dataset, Hyperparameter-Suche (z. B. Optuna), Early Stopping; Deployment über ECR + ECS/Fargate mit Load Balancer und HTTPS; Monitoring mit Prometheus/Grafana oder Evidently; und den Airflow-DAG tatsächlich betreiben.

**Ehrlichkeitstipp:** Wenn du etwas nicht ausprobiert hast (z. B. Airflow wurde nur entworfen, nicht ausgeführt), sag das offen: *„Den DAG habe ich entworfen, aber nicht betrieben – im Projekt lag der Fokus auf Training, Tracking und Serving."* Das kommt besser an als zu übertreiben.
