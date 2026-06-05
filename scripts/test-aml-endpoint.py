"""
scripts/test-aml-endpoint.py

Validate the Azure ML managed online endpoint with the two standard
demo scenarios (high-risk and low-risk) and print the results.

Usage:
  python scripts/test-aml-endpoint.py \
    --endpoint-name rrs-endpoint-demo \
    [--resource-group rrs-rg] \
    [--workspace rrs-aml-demo]

If --resource-group / --workspace are omitted the script reads them
from the terraform output in the current working directory:
  terraform -chdir=infra/terraform output -json
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

SCENARIO_1 = {
    "primary_dx_group": "HF",
    "comorbidity_index": 5,
    "length_of_stay_days": 7,
    "age_group": "75-84",
    "prior_admissions_12m": 3,
    "procedure_count": 2,
    "discharge_disposition": "SNF",
    "icu_flag": True,
    "emergency_admit_flag": True,
    "insurance_type": "MEDICARE",
    "specialist_consult_ct": 1,
    "incomplete_dc_flag": True,
    "weekend_discharge": False,
}

SCENARIO_2 = {
    "primary_dx_group": "KNEE_HIP",
    "comorbidity_index": 1,
    "length_of_stay_days": 2,
    "age_group": "45-54",
    "prior_admissions_12m": 0,
    "procedure_count": 1,
    "discharge_disposition": "HOME",
    "icu_flag": False,
    "emergency_admit_flag": False,
    "insurance_type": "COMMERCIAL",
    "specialist_consult_ct": 0,
    "incomplete_dc_flag": False,
    "weekend_discharge": False,
}


def tf_output(key: str) -> str | None:
    """Read a value from terraform output JSON, if available."""
    tf_dir = Path(__file__).parent.parent / "infra" / "terraform"
    try:
        result = subprocess.run(
            ["terraform", f"-chdir={tf_dir}", "output", "-json"],
            capture_output=True, text=True, check=True, timeout=15,
        )
        outputs = json.loads(result.stdout)
        return outputs.get(key, {}).get("value")
    except Exception:
        return None


def parse_args():
    p = argparse.ArgumentParser(description="Validate Azure ML endpoint")
    p.add_argument("--endpoint-name", required=True)
    p.add_argument("--resource-group", default=None)
    p.add_argument("--workspace", default=None)
    return p.parse_args()


def score(ml_client, endpoint_name: str, payload: dict) -> dict:
    import tempfile, os
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(payload, f)
        tmp = f.name
    try:
        response = ml_client.online_endpoints.invoke(
            endpoint_name=endpoint_name,
            request_file=tmp,
        )
        return json.loads(response)
    finally:
        os.unlink(tmp)


def main():
    args = parse_args()

    resource_group = args.resource_group or tf_output("resource_group_name")
    workspace = args.workspace or tf_output("aml_workspace_name")

    if not resource_group or not workspace:
        print("ERROR: --resource-group and --workspace are required (or run terraform apply first)")
        sys.exit(1)

    try:
        from azure.ai.ml import MLClient
        from azure.identity import DefaultAzureCredential
    except ImportError:
        print("ERROR: pip install azure-ai-ml azure-identity")
        sys.exit(1)

    subscription_id = subprocess.run(
        ["az", "account", "show", "--query", "id", "-o", "tsv"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()

    ml_client = MLClient(
        DefaultAzureCredential(), subscription_id, resource_group, workspace
    )

    print(f"Endpoint: {args.endpoint_name}")
    print(f"Workspace: {workspace} ({resource_group})\n")

    for label, payload in [("Scenario 1 (HIGH-risk HF/SNF)", SCENARIO_1),
                            ("Scenario 2 (LOW-risk KNEE_HIP/HOME)", SCENARIO_2)]:
        print(f"--- {label} ---")
        try:
            result = score(ml_client, args.endpoint_name, payload)
            print(f"  risk_score : {result.get('risk_score')}")
            print(f"  risk_tier  : {result.get('risk_tier')}")
            print(f"  shap_factors:")
            for f in result.get("shap_factors", []):
                print(f"    {f['feature']}: {f['shap_value']:+.4f} ({f['direction']})")
            print(f"  inference_ms: {result.get('inference_ms')}")
        except Exception as exc:
            print(f"  ERROR: {exc}")
        print()

    print("Validation complete.")


if __name__ == "__main__":
    main()
