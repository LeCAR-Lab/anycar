import os

# from .models import TorchTransformer, TorchTransformerDecoder, TorchGPT2
# from .jax_models import JaxLearnedPositionalEncoding, JaxTransformerDecoder

CAR_FOUNDATION_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
CAR_FOUNDATION_MODEL_DIR = os.path.join(CAR_FOUNDATION_ROOT_DIR, "car_foundation", 'models')
CAR_FOUNDATION_DATA_DIR = os.path.join(CAR_FOUNDATION_ROOT_DIR, "car_foundation", 'data')