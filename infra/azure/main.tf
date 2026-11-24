# Infraestructura de Azure como código (Terraform). Este fichero describe lo que debe existir en la
# suscripción "Azure for Students"; Terraform compara con lo que hay y crea, cambia o borra lo justo.
#
#   terraform init     # descarga el "plugin" de Azure (una vez)
#   terraform plan     # enseña qué haría, sin tocar nada
#   terraform apply    # lo hace (pide confirmación)
#   terraform destroy  # lo borra todo
#
# Autenticación: usa la sesión de `az login`. No hay claves en estos ficheros.

terraform {
  required_version = ">= 1.9"
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 5.13"
    }
  }
}

provider "azurerm" {
  features {}
  subscription_id = var.subscription_id
  # Por defecto el plugin activa decenas de servicios de Azure en la suscripción al arrancar (tarda
  # minutos). Mejor activar solo los que usemos, cuando los usemos.
  resource_provider_registrations = "none"
}

# Alerta de presupuesto: Azure manda un correo cuando el gasto del mes llega al 50 % y al 100 % de
# `monthly_budget_usd`, y otro si la previsión del mes pasa del 100 %. No para nada por sí sola: es un
# aviso. (Sin tarjeta, al acabarse los 100 $ del año Azure para los servicios en vez de cobrar.)
resource "azurerm_consumption_budget_subscription" "monthly" {
  name            = "amr-research-agent-monthly"
  subscription_id = "/subscriptions/${var.subscription_id}"
  amount          = var.monthly_budget_usd
  time_grain      = "Monthly"

  time_period {
    start_date = "2026-11-01T00:00:00Z" # tiene que ser el día 1 de un mes
  }

  notification {
    threshold      = 50
    operator       = "GreaterThan"
    threshold_type = "Actual"
    contact_emails = [var.alert_email]
  }

  notification {
    threshold      = 100
    operator       = "GreaterThan"
    threshold_type = "Actual"
    contact_emails = [var.alert_email]
  }

  notification {
    threshold      = 100
    operator       = "GreaterThan"
    threshold_type = "Forecasted"
    contact_emails = [var.alert_email]
  }
}
