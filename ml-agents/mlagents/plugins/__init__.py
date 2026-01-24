from typing import Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from mlagents.trainers.settings import TrainerSettings

ML_AGENTS_STATS_WRITER = "mlagents.stats_writer"
ML_AGENTS_TRAINER_TYPE = "mlagents.trainer_type"

# Type annotations: Dict[str, TrainerSettings] at runtime, but use Any to avoid circular imports
all_trainer_types: Dict[str, Any] = {}
all_trainer_settings: "Dict[str, TrainerSettings]" = {}
