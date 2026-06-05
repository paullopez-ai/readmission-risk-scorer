resource "azurerm_service_plan" "functions" {
  name                = "${var.prefix}-plan-${var.environment}"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  os_type             = "Linux"
  sku_name            = "Y1" # Consumption plan — pay-per-call, no idle cost

  tags = local.tags
}

resource "azurerm_linux_function_app" "predict" {
  name                = "${var.prefix}-func-${var.environment}"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location

  storage_account_name       = azurerm_storage_account.functions.name
  storage_account_access_key = azurerm_storage_account.functions.primary_access_key
  service_plan_id            = azurerm_service_plan.functions.id

  identity {
    type = "SystemAssigned"
  }

  site_config {
    application_stack {
      python_version = "3.11"
    }

    application_insights_connection_string = azurerm_application_insights.main.connection_string
    application_insights_key               = azurerm_application_insights.main.instrumentation_key
  }

  app_settings = {
    # AML endpoint forwarding
    AML_ENDPOINT_URL  = azurerm_machine_learning_online_endpoint.predict.scoring_uri
    AML_ENDPOINT_NAME = azurerm_machine_learning_online_endpoint.predict.name

    # Key Vault reference for Anthropic key (populated after kv secret is set)
    ANTHROPIC_API_KEY = "@Microsoft.KeyVault(VaultName=${azurerm_key_vault.main.name};SecretName=anthropic-api-key)"

    # Risk tier thresholds (matches local defaults)
    HIGH_THRESHOLD     = "0.65"
    MODERATE_THRESHOLD = "0.35"
    INFERENCE_MODE     = "azure_ml"

    # Required Function App settings
    FUNCTIONS_WORKER_RUNTIME        = "python"
    SCM_DO_BUILD_DURING_DEPLOYMENT  = "true"
    ENABLE_ORYX_BUILD               = "true"
  }

  tags = local.tags
}

# Optional: API Management in front of the Function App
resource "azurerm_api_management" "main" {
  count               = var.enable_apim ? 1 : 0
  name                = "${var.prefix}-apim-${var.environment}"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  publisher_name      = "readmission-risk-scorer"
  publisher_email     = "noreply@example.com"
  sku_name            = "Developer_1" # ~$0.07/hr; use Consumption for pay-per-call

  tags = local.tags
}

resource "azurerm_api_management_api" "predict" {
  count               = var.enable_apim ? 1 : 0
  name                = "predict"
  resource_group_name = azurerm_resource_group.main.name
  api_management_name = azurerm_api_management.main[0].name
  revision            = "1"
  display_name        = "Readmission Risk Scorer"
  path                = "rrs"
  protocols           = ["https"]

  import {
    content_format = "openapi"
    content_value  = <<-OPENAPI
      openapi: "3.0.0"
      info:
        title: Readmission Risk Scorer
        version: "1"
      paths:
        /predict:
          post:
            operationId: predict
            summary: Score a discharge record for 30-day readmission risk
            requestBody:
              required: true
              content:
                application/json:
                  schema:
                    type: object
            responses:
              "200":
                description: RiskAssessment
    OPENAPI
  }
}

resource "azurerm_api_management_backend" "function" {
  count               = var.enable_apim ? 1 : 0
  name                = "function-app"
  resource_group_name = azurerm_resource_group.main.name
  api_management_name = azurerm_api_management.main[0].name
  protocol            = "http"
  url                 = "https://${azurerm_linux_function_app.predict.default_hostname}/api"
}
