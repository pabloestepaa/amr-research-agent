# Los "huecos" de main.tf (como values.yaml en Helm). Los valores van en terraform.tfvars, que no se
# sube a git (lleva el correo y el ID de la suscripción). Plantilla: terraform.tfvars.example.

variable "subscription_id" {
  description = "ID de la suscripción Azure for Students (az account show --query id -o tsv)"
  type        = string
}

variable "alert_email" {
  description = "Correo que recibe las alertas de presupuesto"
  type        = string
}

variable "monthly_budget_usd" {
  description = "Presupuesto mensual en dólares. 100 $/año de crédito: ~8 $/mes para que dure el año"
  type        = number
  default     = 8
}
