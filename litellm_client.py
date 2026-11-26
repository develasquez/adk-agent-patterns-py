import os
from google.adk.models.lite_llm import LiteLlm


class OrbitLiteLLM:
    """Gestiona la conexión de agentes ADK con Orbit v2 mediante LiteLLM."""

    def __init__(
        self,
        model: str = "DeepSeek-V4.1-Flash-1017",
        base_url: str = "https://orbit.dev.core.axmos.tech/openai",
        api_key: str | None = None,
    ):
        self.model = f"openai/{model}"
        self.base_url = base_url
        self.api_key = api_key or os.getenv("LITELLM_API_KEY")

    def get_model(self) -> LiteLlm:
        return LiteLlm(model=self.model, api_base=self.base_url, api_key=self.api_key)
