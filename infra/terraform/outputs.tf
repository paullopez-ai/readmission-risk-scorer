output "resource_group_name" {
  description = "Name of the provisioned resource group."
  value       = azurerm_resource_group.main.name
}

output "storage_account_name" {
  description = "Storage account name (used to upload model artifacts)."
  value       = azurerm_storage_account.models.name
}

output "model_container_name" {
  description = "Blob container name for model artifacts."
  value       = azurerm_storage_container.models.name
}

output "aml_workspace_name" {
  description = "Azure ML workspace name (needed for deploy-model.py)."
  value       = azurerm_machine_learning_workspace.main.name
}

output "aml_endpoint_name" {
  description = "Azure ML managed online endpoint name."
  value       = azurerm_machine_learning_online_endpoint.predict.name
}

output "aml_endpoint_scoring_uri" {
  description = "Azure ML managed online endpoint scoring URI."
  value       = azurerm_machine_learning_online_endpoint.predict.scoring_uri
  sensitive   = false
}

output "function_app_name" {
  description = "Azure Function App name."
  value       = azurerm_linux_function_app.predict.name
}

output "function_app_url" {
  description = "Azure Function App default hostname (POST /api/predict)."
  value       = "https://${azurerm_linux_function_app.predict.default_hostname}/api/predict"
}

output "key_vault_name" {
  description = "Azure Key Vault name (stores ANTHROPIC_API_KEY)."
  value       = azurerm_key_vault.main.name
}

output "log_analytics_workspace_id" {
  description = "Log Analytics workspace ID for Azure Monitor dashboards."
  value       = azurerm_log_analytics_workspace.main.workspace_id
}

output "deploy_command" {
  description = "Command to run after terraform apply to deploy the model to the AML endpoint."
  value = join(" ", [
    "python scripts/deploy-model.py",
    "--resource-group ${azurerm_resource_group.main.name}",
    "--workspace ${azurerm_machine_learning_workspace.main.name}",
    "--endpoint ${azurerm_machine_learning_online_endpoint.predict.name}",
    "--storage-account ${azurerm_storage_account.models.name}",
    "--container ${azurerm_storage_container.models.name}",
  ])
}
