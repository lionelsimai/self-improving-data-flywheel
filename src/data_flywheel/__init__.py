"""Self-improving data flywheel: feedback -> synthetic data -> LoRA -> eval."""
from .evaluate import HeldOutItem, default_held_out, score_model
from .feedback import Feedback, FeedbackStore
from .synthetic import generate_synthetic

__all__ = ["FeedbackStore", "Feedback", "generate_synthetic",
           "HeldOutItem", "default_held_out", "score_model"]
__version__ = "0.1.1"
