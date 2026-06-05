resource "azurerm_machine_learning_workspace" "main" {
  name                    = "${var.prefix}-aml-${var.environment}"
  resource_group_name     = azurerm_resource_group.main.name
  location                = azurerm_resource_group.main.location
  application_insights_id = azurerm_application_insights.main.id
  key_vault_id            = azurerm_key_vault.main.id
  storage_account_id      = azurerm_storage_account.models.id

  identity {
    type = "SystemAssigned"
  }

  tags = local.tags
}

resource "azurerm_machine_learning_online_endpoint" "predict" {
  name                = "${var.prefix}-endpoint-${var.environment}"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  workspace_id        = azurerm_machine_learning_workspace.main.id

  identity {
    type = "SystemAssigned"
  }

  tags = local.tags
}
