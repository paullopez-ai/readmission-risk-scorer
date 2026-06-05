resource "azurerm_log_analytics_workspace" "main" {
  name                = "${var.prefix}-logs-${var.environment}"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  sku                 = "PerGB2018"
  retention_in_days   = 30

  tags = local.tags
}

resource "azurerm_application_insights" "main" {
  name                = "${var.prefix}-ai-${var.environment}"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  workspace_id        = azurerm_log_analytics_workspace.main.id
  application_type    = "web"

  tags = local.tags
}

# Cost alert: warn at $20/month to catch runaway endpoint costs
resource "azurerm_consumption_budget_resource_group" "main" {
  name              = "${var.prefix}-budget-${var.environment}"
  resource_group_id = azurerm_resource_group.main.id

  amount     = 20
  time_grain = "Monthly"

  time_period {
    start_date = formatdate("YYYY-MM-01'T'00:00:00Z", timestamp())
  }

  notification {
    enabled   = true
    threshold = 80
    operator  = "GreaterThan"

    contact_emails = []
  }

  notification {
    enabled   = true
    threshold = 100
    operator  = "GreaterThan"

    contact_emails = []
  }

  lifecycle {
    ignore_changes = [time_period]
  }
}
