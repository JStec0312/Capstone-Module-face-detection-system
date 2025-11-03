from deepface import DeepFace 
import numpy as np
from extract_vector import extract_vector
#--vector -img:<image_path> -model_name:<embeding generator model default ArcFace> -detector_backend=<face detector backend default retinaface> -enforce_detection=<default false>
def do_vector(args):
    img_path = args.get('img')
    print(img_path)
    if not img_path:
        raise ValueError("No img")
    model_name = args.get('model_name', 'ArcFace')
    detector_backend = args.get("detector_backend", 'retinaface')
    enforce_detection = args.get('enforce_detection', False)

    print(extract_vector(img_path=img_path, model_name=model_name, detector_backend=detector_backend, enforce_detection=enforce_detection))
    return


