"""Application settings loaded from environment variables."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # GitHub
    github_token: str = ""
    github_webhook_secret: str = ""
    github_api_base: str = "https://api.github.com"

    # LLM provider selection: "openai" | "gemini"
    llm_provider: str = "openai"

    # Gemini
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"

    # OpenAI-compatible (Azure / custom endpoint)
    openai_api_key: str = ""
    openai_base_url: str = "https://omium-ai-2-resource.services.ai.azure.com/openai/v1"
    openai_model: str = "gpt-5.4-mini"

    # Shared LLM
    llm_timeout: float = 45.0

    # Search
    tavily_api_key: str = ""

    # Observability (optional)
    omium_api_key: str = ""

    # HTTP
    http_timeout: float = 20.0

    # Workflow tuning
    confidence_threshold: float = 0.6
    log_level: str = "INFO"
    default_base_branch: str = "main"
    live_writes_enabled: bool = False

    @property
    def llm_available(self) -> bool:
        """True when the active provider has an API key configured."""
        if self.llm_provider == "openai":
            return bool(self.openai_api_key)
        if self.llm_provider == "gemini":
            return bool(self.gemini_api_key)
        return False


settings = Settings()
