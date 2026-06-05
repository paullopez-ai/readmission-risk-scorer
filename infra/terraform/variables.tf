variable "prefix" {
  description = "Short resource name prefix. Keep to 3-5 chars (used in storage account names which have strict length limits)."
  type        = string
  default     = "rrs"
}

variable "location" {
  description = "Azure region for all resources."
  type        = string
  default     = "eastus"
}

variable "environment" {
  description = "Deployment environment label (demo, staging, prod)."
  type        = string
  default     = "demo"
}

variable "aml_compute_sku" {
  description = "VM SKU for the Azure ML managed online endpoint. Standard_DS2_v2 is the cheapest tier that supports sklearn/XGBoost inference. ~$0.096/hr."
  type        = string
  default     = "Standard_DS2_v2"
}

variable "aml_endpoint_instance_count" {
  description = "Number of instances behind the AML online endpoint. 1 for demo; scale up for production."
  type        = number
  default     = 1
}

variable "enable_apim" {
  description = "Set true to provision Azure API Management in front of the Function App. Developer tier adds ~$0.07/hr. Set false to expose the Function URL directly."
  type        = bool
  default     = false
}

variable "anthropic_api_key" {
  description = "Anthropic API key stored in Key Vault for the optional /predict/explain LLM narrative. Set via TF_VAR_anthropic_api_key env var; do not commit."
  type        = string
  sensitive   = true
  default     = ""
}
