# Model artifacts

The full fitted hybrid_model.pkl is approximately 49 MB and is not committed to this Git repository. The local project bundle delivered with the prototype run contains this file, along with the scaler and encoder. For a reproducible Git-based release, publish the model through Git LFS or a versioned release asset after verifying the artifact and its provenance.

models/metadata.json records the preview model configuration and held-out metrics. To regenerate artifacts locally, install requirements.txt, place the required datasets under dataset/, and run python train.py.

Do not use model predictions as proof of criminal activity.
