# La API en Azure Container Apps: como Render (le das una imagen y la publica en una URL https), pero
# escala a cero cuando nadie la usa y cobra por segundo de uso, con una capa gratuita mensual de
# 180.000 vCPU·s (unas 200 h a 0,25 vCPU). La base de datos es la misma Neon que usa Render.

# Activar el servicio de Container Apps en la suscripción (una vez; tarda un par de minutos).
resource "azurerm_resource_provider_registration" "app" {
  name = "Microsoft.App"
}

# Grupo de recursos: la "carpeta" de Azure donde vive todo lo de este proyecto. Borrarla lo borra todo.
resource "azurerm_resource_group" "main" {
  name = "amr-research-agent"
  # Azure for Students solo deja 5 regiones (norwayeast, spaincentral, italynorth, switzerlandnorth,
  # polandcentral): Madrid es la más cercana a nosotros y a Neon (Fráncfort).
  location = "spaincentral"
}

# Entorno: la red y el "clúster" gestionado donde corren las apps (Azure lo lleva por dentro con
# Kubernetes). Sin Log Analytics, que se paga por GB: los logs se ven en directo con
# `az containerapp logs show -n amr-api -g amr-research-agent --follow`.
resource "azurerm_container_app_environment" "main" {
  name                = "amr-env"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  depends_on          = [azurerm_resource_provider_registration.app]
}

resource "azurerm_container_app" "api" {
  name                         = "amr-api"
  container_app_environment_id = azurerm_container_app_environment.main.id
  resource_group_name          = azurerm_resource_group.main.name
  revision_mode                = "Single" # cada despliegue sustituye al anterior

  # Secretos de la app: se guardan cifrados en Azure y llegan al contenedor como variables.
  # (También quedan en terraform.tfstate, que no se sube a git.)
  secret {
    name  = "database-url"
    value = var.database_url
  }
  secret {
    name  = "llm-api-key"
    value = var.llm_api_key
  }

  # Una URL pública https (*.azurecontainerapps.io) que lleva al puerto 8000 del contenedor.
  ingress {
    external_enabled = true
    target_port      = 8000
    traffic_weight {
      latest_revision = true
      percentage      = 100
    }
  }

  template {
    # 0 = se apaga sin visitas (gratis; el primer acceso tarda en arrancar, como Render).
    # 1 como máximo: los límites de /ask y /agent viven en la memoria del proceso.
    min_replicas = 0
    max_replicas = 1

    container {
      name  = "api"
      image = "ghcr.io/pabloestepaa/amr-research-agent:${var.image_tag}" # pública: sin credenciales
      # 0,25 vCPU y 0,5 Gi es la combinación más pequeña: 512 MB, como en Render Free.
      cpu    = 0.25
      memory = "0.5Gi"

      env {
        name  = "PORT"
        value = "8000"
      }
      env {
        name        = "DATABASE_URL"
        secret_name = "database-url"
      }
      env {
        name        = "LLM_API_KEY"
        secret_name = "llm-api-key"
      }
      env {
        name  = "LLM_BASE_URL"
        value = "https://api.groq.com/openai/v1"
      }
      env {
        name  = "LLM_MODEL"
        value = "openai/gpt-oss-120b"
      }
      # Mismos límites que en Render; los dos comparten la cuota diaria de Groq.
      env {
        name  = "ASK_PER_IP_PER_MINUTE"
        value = "5"
      }
      env {
        name  = "ASK_PER_DAY"
        value = "50"
      }
      env {
        name  = "AGENT_PER_IP_PER_MINUTE"
        value = "1"
      }
      env {
        name  = "AGENT_PER_DAY"
        value = "15"
      }

      # Las mismas comprobaciones que en el chart de Helm, todas sobre /health.
      startup_probe {
        transport               = "HTTP"
        port                    = 8000
        path                    = "/health"
        initial_delay           = 10
        interval_seconds        = 5
        failure_count_threshold = 10 # hasta ~60 s para arrancar con 0,25 vCPU
      }
      readiness_probe {
        transport        = "HTTP"
        port             = 8000
        path             = "/health"
        interval_seconds = 10
      }
      liveness_probe {
        transport               = "HTTP"
        port                    = 8000
        path                    = "/health"
        interval_seconds        = 10
        timeout                 = 5
        failure_count_threshold = 3
      }
    }
  }
}

output "url" {
  description = "Dirección pública de la API en Azure"
  value       = "https://${azurerm_container_app.api.ingress[0].fqdn}"
}
