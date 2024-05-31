from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_path: str = "/models/model.cbm"
    model_config = SettingsConfigDict(env_file=".env")


def config_loader():
    return Config()
