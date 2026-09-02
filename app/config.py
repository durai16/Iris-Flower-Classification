from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    MODEL_PATH: str = "models/iris_model.pkl"

    LOG_LEVEL: str = "INFO"

    MAX_BATCH_SIZE: int = 100

    API_TITLE: str = "Iris Flower Classification API"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()