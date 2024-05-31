from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    backend_url: str = "http://localhost:8000/"
    model_config = SettingsConfigDict(env_file=".env")


def config_loader():
    return Config()
