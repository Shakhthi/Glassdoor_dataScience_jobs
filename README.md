DS_Jobs
=======

Overview
--------
DS_Jobs is an end-to-end data pipeline and ML project that ingests job-posting data, validates and transforms it, trains models, and serves predictions via a FastAPI application. The project includes containerization, CI/CD to push images to AWS ECR, and simple deployment automation for a self-hosted runner or EC2 instance.

Repository layout
-----------------
- `app.py` — FastAPI app exposing `/train` and `/predict` endpoints.
- `Dockerfile` — builds a container image for the app.
- `ec2_bootstrap.sh` — helper to provision Docker on EC2 (used for self-hosted deployment).
- `Artifacts/` — pipeline output snapshots grouped by run timestamps.
- `data/` — raw and cleaned CSVs (`ds_jobs.csv`, `cleaned_ds_jobs.csv`) and DVC pointers.
- `data_schema/` — `schema.yaml` describing expected data schema.
- `final_model/` — saved model and preprocessor artifacts (`model.pkl`, `preprocessor.pkl`).
- `mlruns/` — MLflow run artifacts (if used in training pipeline).
- `prediction_output/` — CSV output of predictions (`output.csv`).
- `src/` — source code (pipeline, constants, utils, logging, exception handling, etc.).
- `.github/workflows/main.yml` — GitHub Actions CI/CD workflow to build/push image to AWS ECR and deploy on a self-hosted runner.

Important components & behavior
-------------------------------
- FastAPI server: `app.py` runs the server (default run in file uses `uvicorn` to bind `0.0.0.0` and port `8000`).
- MongoDB integration: connection URI read from `MONGODB_URI` environment variable; database/collection names defined in `src/constant/training_pipeline`.
- Model serving: prediction flow loads `final_model/preprocessor.pkl` and `final_model/model.pkl`, wraps them in `DsEstimator`, and returns predictions appended to uploaded CSV as HTML table.
- Training: hitting `/train` triggers the `TrainingPipeline` inside `src/pipeline` which writes artifacts to `Artifacts/` and `final_model/`.
- CI/CD: GitHub Actions builds and pushes Docker image to AWS ECR, then a self-hosted job pulls and runs the container. The workflow uses `aws-actions/amazon-ecr-login@v2` output `steps.login-ecr.outputs.registry` for registry URI.

Environment / Secrets
---------------------
Set the following environment variables (locally or in CI/CD secrets):
- `MONGODB_URI` — MongoDB connection URI (use TLS CA config if needed).
- `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION` — AWS credentials for pushing to ECR and deployment.
- `ECR_REPOSITORY_NAME` — name of the ECR repo.
- `AWS_ECR_LOGIN_URI` — (optional) ECR login URI; workflow uses the `login-ecr` output for correctness.

Local development
-----------------
1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate    # or .venv\\Scripts\\activate on Windows
pip install -r requirements.txt
```

2. Configure `.env` with `MONGODB_URI` and other vars.
3. Run the app:

```bash
python app.py
# or with uvicorn directly if preferred:
uvicorn app:app --host 0.0.0.0 --port 8000
```

4. Open `http://localhost:8000/docs` to explore APIs.

Docker (build & run)
--------------------
- Build image:

```bash
docker build -t ds_jobs .
```

- Run container mapping host port to container:

```bash
# if app runs on 8000 inside the container
docker run -p 8080:8000 ds_jobs
# then access http://localhost:8080
```

Note: Ensure the container `EXPOSE` in `Dockerfile` matches the port the app binds to (either change `EXPOSE` to `8000` or run with `-p host:8000`).

CI/CD & Deployment
------------------
- GitHub Actions (`.github/workflows/main.yml`) builds and pushes the Docker image to AWS ECR.
- Secrets required in GitHub repo: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, `ECR_REPOSITORY_NAME`, optionally `AWS_ECR_LOGIN_URI` for older configs.
- The self-hosted deployment job pulls the image and runs `docker run`. The workflow was fixed to use `steps.login-ecr.outputs.registry` to avoid malformed URIs (fixes `docker: invalid proto:` errors).

Files of interest
-----------------
- `templates/table.html` — template used for rendering prediction results.
- `src/utils` — helper utilities including `load_object` and model wrapper `DsEstimator`.
- `ec2_bootstrap.sh` — commands to install Docker on EC2 instances used as self-hosted runners.

Troubleshooting
---------------
- docker invalid proto: ensure image URI uses a valid registry (workflow now uses `steps.login-ecr.outputs.registry`).
- Port mismatch: check `app.py` port and `Dockerfile` `EXPOSE` and adjust mapping when running container.
- MongoDB connection errors: confirm `MONGODB_URI` and TLS CA (the project uses `certifi` for CA file).

Contact
-------
For questions, collaboration, or support, please contact the project maintainer:

- **Name:** Mathanbabu Kaliappan
- **Email:** sakthikaliappan7797@gmail.com
- **LinkedIn:** https://www.linkedin.com/in/mathanbabu-kaliappan-58b7171a3/


