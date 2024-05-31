from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    path_to_model: str = "/models/model.cbm"
    model_config = SettingsConfigDict(env_file=".env")


def config_loader():
    return Config()
