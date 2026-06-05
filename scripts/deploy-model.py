"""
scripts/deploy-model.py

Upload model.tar.gz to Azure Blob Storage and register + deploy it to
the Azure ML managed online endpoint provisioned by Terraform.

Prerequisites:
  az login (or service principal env vars set)
  python scripts/package-model.py  (creates model.tar.gz)
  terraform apply                  (creates AML workspace + endpoint)

Usage (values from terraform output):
  python scripts/deploy-model.py \
    --resource-group rrs-rg \
    --workspace     rrs-aml-demo \
    --endpoint      rrs-endpoint-demo \
    --storage-account rrsmodelsdemo \
    --container     model-artifacts

The script is idempotent: re-running updates the deployment to a new
model version with a rolling blue/green swap.
"""

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent


def parse_args():
    p = argparse.ArgumentParser(description="Deploy readmission risk scorer to Azure ML")
    p.add_argument("--resource-group", required=True)
    p.add_argument("--workspace", required=True)
    p.add_argument("--endpoint", required=True)
    p.add_argument("--storage-account", required=True)
    p.add_argument("--container", required=True, default="model-artifacts")
    p.add_argument("--model-version", default=None,
                   help="Version string; defaults to timestamp")
    p.add_argument("--traffic-percent", type=int, default=100,
                   help="Traffic to route to new deployment (0-100)")
    return p.parse_args()


def upload_artifact(storage_account: str, container: str, tar_path: Path) -> str:
    """Upload model.tar.gz to blob storage, return blob URL."""
    from azure.storage.blob import BlobServiceClient
    from azure.identity import DefaultAzureCredential

    print(f"Uploading {tar_path.name} to {storage_account}/{container}...")
    credential = DefaultAzureCredential()
    url = f"https://{storage_account}.blob.core.windows.net"
    client = BlobServiceClient(account_url=url, credential=credential)
    container_client = client.get_container_client(container)
    blob_name = tar_path.name

    with open(tar_path, "rb") as f:
        container_client.upload_blob(blob_name, f, overwrite=True)

    blob_url = f"{url}/{container}/{blob_name}"
    print(f"  -> {blob_url}")
    return blob_url


def deploy(args, blob_url: str, version: str):
    """Register model and create/update deployment on the AML endpoint."""
    from azure.ai.ml import MLClient
    from azure.ai.ml.entities import (
        Model,
        ManagedOnlineDeployment,
        Environment,
        CodeConfiguration,
    )
    from azure.identity import DefaultAzureCredential

    subscription_id = _get_subscription_id()
    ml_client = MLClient(
        DefaultAzureCredential(),
        subscription_id,
        args.resource_group,
        args.workspace,
    )

    # Register model
    print(f"Registering model version {version}...")
    model = ml_client.models.create_or_update(
        Model(
            name="readmission-risk-scorer",
            version=version,
            path=blob_url,
            type="custom_model",
            description="XGBoost 30-day readmission risk scorer",
        )
    )
    print(f"  Model registered: {model.name}:{model.version}")

    # Create environment from conda.yml
    conda_file = REPO_ROOT / "infra" / "azure-ml" / "conda.yml"
    env = ml_client.environments.create_or_update(
        Environment(
            name="readmission-risk-scorer-env",
            conda_file=str(conda_file),
            image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
        )
    )

    # Create deployment
    deployment_name = f"v{version.replace('.', '-').replace(':', '-')}"[:32]
    print(f"Creating deployment '{deployment_name}' on endpoint '{args.endpoint}'...")
    deployment = ml_client.online_deployments.begin_create_or_update(
        ManagedOnlineDeployment(
            name=deployment_name,
            endpoint_name=args.endpoint,
            model=model,
            environment=env,
            code_configuration=CodeConfiguration(
                code=str(REPO_ROOT / "infra" / "azure-ml"),
                scoring_script="score.py",
            ),
            instance_type="Standard_DS2_v2",
            instance_count=1,
        )
    ).result()

    print(f"  Deployment ready: {deployment.name}")

    # Route traffic
    if args.traffic_percent > 0:
        print(f"Setting traffic: {deployment_name} -> {args.traffic_percent}%")
        endpoint = ml_client.online_endpoints.get(args.endpoint)
        endpoint.traffic = {deployment_name: args.traffic_percent}
        ml_client.online_endpoints.begin_create_or_update(endpoint).result()
        print("  Traffic updated.")


def _get_subscription_id() -> str:
    """Read subscription ID from az CLI context."""
    import subprocess, json
    result = subprocess.run(
        ["az", "account", "show", "--query", "id", "-o", "tsv"],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def main():
    args = parse_args()

    tar_path = REPO_ROOT / "model.tar.gz"
    if not tar_path.exists():
        print("ERROR: model.tar.gz not found. Run: python scripts/package-model.py")
        sys.exit(1)

    version = args.model_version or time.strftime("%Y%m%d%H%M%S")

    try:
        blob_url = upload_artifact(args.storage_account, args.container, tar_path)
        deploy(args, blob_url, version)
    except ImportError as exc:
        print(f"ERROR: missing dependency — {exc}")
        print("Install: pip install azure-ai-ml azure-storage-blob azure-identity")
        sys.exit(1)

    print("\nDeployment complete.")
    print(f"Test with: python scripts/test-aml-endpoint.py --endpoint-name {args.endpoint}")


if __name__ == "__main__":
    main()
