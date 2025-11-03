import numpy as np
from extract_vector import extract_vector
from cosine_similarity import cosine_similarity
def do_cos_sim(args):
    img_path1 = args.get('img1')
    img_path2 =  args.get('img2')
    if not img_path1:
        raise ValueError("No img_path1")
    if not img_path2:
        raise ValueError("No img path2")
    model_name = args.get('model_name', 'ArcFace')
    detector_backend = args.get("detector_backend", 'retinaface')
    enforce_detection = args.get('enforce_detection', False)
    if enforce_detection in ['true', 'True', '1']:
        enforce_detection = True
    else:
        enforce_detection = False
        
    vec1 = extract_vector(img_path=img_path1, model_name=model_name, detector_backend=detector_backend, enforce_detection=enforce_detection)
    vec2=extract_vector(img_path=img_path2, model_name=model_name, detector_backend=detector_backend, enforce_detection=enforce_detection)
    print("Results for model ", model_name, " and detector ", detector_backend)
    print(cosine_similarity(vec1, vec2))