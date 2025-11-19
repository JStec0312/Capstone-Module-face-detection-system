import numpy as np
from deepface import DeepFace
from typing import Optional
def extract_vector(
    img_path: str,
    model_name: str = "Facenet",
    detector_backend: str = "retinaface",
    enforce_detection: bool = False,
    normalization: str = "Facenet",
    model: Optional[object] = None
) -> np.ndarray:
    
    if not img_path:
        raise ValueError("No img_path")

    reps = DeepFace.represent(
        img_path=img_path,
        model_name=model_name if model is None else None, 
        detector_backend=detector_backend,
        enforce_detection=enforce_detection,
        normalization=normalization
    )
    return np.array(reps[0]["embedding"], dtype=np.float32)