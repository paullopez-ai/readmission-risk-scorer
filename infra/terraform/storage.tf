resource "azurerm_storage_account" "models" {
  name                     = "${var.prefix}models${var.environment}"
  resource_group_name      = azurerm_resource_group.main.name
  location                 = azurerm_resource_group.main.location
  account_tier             = "Standard"
  account_replication_type = "LRS"
  min_tls_version          = "TLS1_2"

  blob_properties {
    delete_retention_policy {
      days = 7
    }
  }

  tags = local.tags
}

resource "azurerm_storage_container" "models" {
  name                  = "model-artifacts"
  storage_account_id    = azurerm_storage_account.models.id
  container_access_type = "private"
}

# Second storage account for Azure Function App (required by Azure Functions runtime)
resource "azurerm_storage_account" "functions" {
  name                     = "${var.prefix}func${var.environment}"
  resource_group_name      = azurerm_resource_group.main.name
  location                 = azurerm_resource_group.main.location
  account_tier             = "Standard"
  account_replication_type = "LRS"
  min_tls_version          = "TLS1_2"

  tags = local.tags
}
