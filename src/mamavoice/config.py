from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MV_", env_file=".env", extra="ignore")

    ollama_url: str = "http://localhost:11434"
    llm_model: str = "qwen2.5:3b-instruct"      # small, quantized, runs on CPU
    embed_model: str = "intfloat/multilingual-e5-small"
    nllb_model: str = "facebook/nllb-200-distilled-600M"
    nllb_adapter: str | None = None             # path/HF id of your LoRA adapter
    whisper_size: str = "small"
    index_dir: str = "data/index"
    guidelines_dir: str = "data/guidelines"
    top_k: int = 4
    min_retrieval_score: float = 0.35           # below this => refuse to answer, refer to clinic
    max_audio_seconds: int = 60


settings = Settings()

# ISO-ish app codes -> model-specific codes
LANGS = {
    "en": {"nllb": "eng_Latn", "whisper": "en", "mms_asr": "eng", "mms_tts": "facebook/mms-tts-eng"},
    "yo": {"nllb": "yor_Latn", "whisper": "yo", "mms_asr": "yor", "mms_tts": "facebook/mms-tts-yor"},
    "ha": {"nllb": "hau_Latn", "whisper": "ha", "mms_asr": "hau", "mms_tts": "facebook/mms-tts-hau"},
    "ig": {"nllb": "ibo_Latn", "whisper": None, "mms_asr": "ibo", "mms_tts": "facebook/mms-tts-ibo"},
}
