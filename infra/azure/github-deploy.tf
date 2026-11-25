# Despliegue desde GitHub Actions sin contraseñas (OIDC).
#
# En cada ejecución, GitHub firma un "pase" que dice de qué repo, rama y evento viene. Azure lo acepta
# solo si coincide con la credencial federada de abajo y, a cambio, da un acceso que caduca en
# minutos. No hay ninguna clave guardada en GitHub que se pueda filtrar.

resource "azurerm_resource_provider_registration" "identity" {
  name = "Microsoft.ManagedIdentity"
}

# La "cuenta" de GitHub en Azure: una identidad sin contraseña.
resource "azurerm_user_assigned_identity" "github" {
  name                = "github-deploy"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  depends_on          = [azurerm_resource_provider_registration.identity]
}

# De quién se fía: solo de los pases de este repo en la rama main (un fork o una PR no sirven).
resource "azurerm_federated_identity_credential" "github_main" {
  name                      = "github-main"
  user_assigned_identity_id = azurerm_user_assigned_identity.github.id
  issuer                    = "https://token.actions.githubusercontent.com"
  audience                  = ["api://AzureADTokenExchange"]
  subject                   = "repo:${var.github_repository}:ref:refs/heads/main"
}

# Qué puede hacer: actualizar esta app y nada más (ni la BD, ni el presupuesto, ni otros recursos).
resource "azurerm_role_assignment" "github_deploy" {
  scope                = azurerm_container_app.api.id
  role_definition_name = "Container Apps Contributor"
  principal_id         = azurerm_user_assigned_identity.github.principal_id
}

# Lo que hay que poner en GitHub (Settings > Secrets and variables > Actions > Variables).
# No son secretos: sin un pase firmado por GitHub para este repo no sirven de nada.
output "github_variables" {
  description = "Variables del repositorio para el job deploy-azure"
  value = {
    AZURE_CLIENT_ID       = azurerm_user_assigned_identity.github.client_id
    AZURE_TENANT_ID       = azurerm_user_assigned_identity.github.tenant_id
    AZURE_SUBSCRIPTION_ID = var.subscription_id
  }
}
