import os
from dotenv import load_dotenv

load_dotenv()


class Settings:

    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

    MODEL_NAME = os.getenv("MODEL_NAME")

    VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH")

    METADATA_PATH = os.getenv("METADATA_PATH")

    DATA_PATH = os.getenv("DATA_PATH")

    IMAGE_PATH = os.getenv("IMAGE_PATH")

    LANGSMITH= os.getenv('LANGSMITH_DR.AI_TEST')


settings = Settings()