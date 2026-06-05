resource "azurerm_key_vault" "main" {
  name                = "${var.prefix}-kv-${var.environment}"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  sku_name            = "standard"
  tenant_id           = data.azurerm_client_config.current.tenant_id

  # Allow the deploying principal to manage secrets
  access_policy {
    tenant_id = data.azurerm_client_config.current.tenant_id
    object_id = data.azurerm_client_config.current.object_id

    secret_permissions = [
      "Get",
      "List",
      "Set",
      "Delete",
      "Purge",
      "Recover",
    ]
  }

  # Allow the Function App managed identity to read secrets
  access_policy {
    tenant_id = data.azurerm_client_config.current.tenant_id
    object_id = azurerm_linux_function_app.predict.identity[0].principal_id

    secret_permissions = ["Get", "List"]
  }

  tags = local.tags
}

resource "azurerm_key_vault_secret" "anthropic_api_key" {
  name         = "anthropic-api-key"
  value        = var.anthropic_api_key != "" ? var.anthropic_api_key : "placeholder-not-set"
  key_vault_id = azurerm_key_vault.main.id

  lifecycle {
    ignore_changes = [value]
  }
}
