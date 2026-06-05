"""
Azure Function: POST /api/predict

Thin proxy in front of the Azure ML managed online endpoint.
Handles auth (AML key via managed identity), request forwarding,
and response pass-through. Same Pydantic contract as the local API.
"""

import json
import logging
import os

import azure.functions as func
from azure.identity import ManagedIdentityCredential
from azure.ai.ml import MLClient
import urllib.request
import urllib.error

logger = logging.getLogger(__name__)

AML_ENDPOINT_URL = os.environ["AML_ENDPOINT_URL"]
AML_ENDPOINT_NAME = os.environ["AML_ENDPOINT_NAME"]


def _get_aml_token() -> str:
    """Obtain a bearer token for the AML endpoint via managed identity."""
    credential = ManagedIdentityCredential()
    token = credential.get_token("https://ml.azure.com/.default")
    return token.token


def main(req: func.HttpRequest) -> func.HttpResponse:
    logger.info("POST /api/predict received")

    try:
        body = req.get_body()
    except Exception as exc:
        logger.error("Failed to read request body: %s", exc)
        return func.HttpResponse(
            json.dumps({"error": "invalid request body"}),
            status_code=400,
            mimetype="application/json",
        )

    try:
        token = _get_aml_token()
    except Exception as exc:
        logger.error("Failed to acquire AML token: %s", exc)
        return func.HttpResponse(
            json.dumps({"error": "authentication failure"}),
            status_code=502,
            mimetype="application/json",
        )

    try:
        aml_req = urllib.request.Request(
            AML_ENDPOINT_URL,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}",
            },
            method="POST",
        )
        with urllib.request.urlopen(aml_req, timeout=30) as resp:
            result = resp.read()
            status = resp.status
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        logger.error("AML endpoint returned %d: %s", exc.code, error_body)
        return func.HttpResponse(
            json.dumps({"error": "upstream AML error", "detail": error_body}),
            status_code=502,
            mimetype="application/json",
        )
    except Exception as exc:
        logger.error("AML request failed: %s", exc)
        return func.HttpResponse(
            json.dumps({"error": "upstream AML unreachable"}),
            status_code=503,
            mimetype="application/json",
        )

    return func.HttpResponse(result, status_code=status, mimetype="application/json")
