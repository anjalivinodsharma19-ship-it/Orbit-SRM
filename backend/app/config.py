from pathlib import Path
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    app_name: str = "OrbitSRM"
    app_version: str = "1.1.0"
    data_dir: Path = Path(__file__).resolve().parents[1] / "data"
    max_upload_mb: int = 200
    max_raster_width: int = Field(default=20000, gt=0)
    max_raster_height: int = Field(default=20000, gt=0)
    max_raster_pixels: int = Field(default=100000000, gt=0)
    max_baseline_output_pixels: int = Field(default=100000000, gt=0)
    metric_chunk_size: int = Field(default=512, gt=0)
    cors_origins: str = "http://localhost:3000,http://localhost:5173"
    model_checkpoint: str = ""
    device: str = "auto"
    # Optional OpenSR (LDSR-S2) inference; inert unless opensr_enabled is set.
    opensr_enabled: bool = False
    opensr_checkpoint: str = ""
    opensr_device: str = "auto"
    opensr_sampling_steps: int = 100
    opensr_max_sampling_steps: int = Field(default=100, gt=0)
    opensr_window: int = 128
    opensr_max_window_size: int = Field(default=256, ge=128)
    opensr_overlap: int = 12
    opensr_batch_size: int = 1
    opensr_max_batch_size: int = Field(default=4, gt=0)
    opensr_dn_scale: float = 10000.0
    opensr_max_tiles: int = 4096
    opensr_torch_threads: int = 0
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @model_validator(mode="after")
    def validate_opensr_window_limit(self):
        if self.opensr_max_window_size < self.opensr_window:
            raise ValueError("OPENSR_MAX_WINDOW_SIZE must be >= OPENSR_WINDOW")
        return self

    @property
    def origins(self):
        return [x.strip() for x in self.cors_origins.split(",") if x.strip()]

settings = Settings()
for folder in ("uploads", "outputs", "previews", "metadata", "temporary"):
    (settings.data_dir / folder).mkdir(parents=True, exist_ok=True)
