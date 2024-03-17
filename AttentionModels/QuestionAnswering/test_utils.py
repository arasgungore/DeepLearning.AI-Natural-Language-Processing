import numpy as np
from termcolor import colored

# +
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

from utils import PositionalEmbedding
from dlai_grader.grading import test_case, object_to_grade
from types import ModuleType, FunctionType

# Compare the two inputs

def comparator(learner, instructor, modelId):
    cases = []
    t = test_case()
    if len(learner) != len(instructor):
        t.failed = True
        t.msg = f"{modelId}: The number of layers in the proposed model does not agree with the expected model"
        t.want = len(instructor)
        t.got = len(learner)
    cases.append(t)
    index_layer = 1

    for a, b in zip(learner, instructor):
        t = test_case()
        if tuple(a) != tuple(b):
            t.failed = True
            t.msg = f"{modelId}: Test failed in layer {index_layer}"
            t.want = b
            t.got = a
        cases.append(t)
        index_layer = index_layer + 1
    return cases


def summary(model):
    result = []
    for layer in model.layers:
        descriptors = [layer.__class__.__name__,
                       layer.output_shape, layer.count_params()]
        if (type(layer) == Dense):
            descriptors.append(layer.activation.__name__)
        if (type(layer) == Dropout):
            descriptors.append(f"rate={layer.rate}")
        if (type(layer) == GRU):
            descriptors.append(f"return_sequences={layer.return_sequences}")
            descriptors.append(f"return_state={layer.return_state}")
        if (type(layer) == PositionalEmbedding):
            descriptors.append(f"vocab_size={layer.vocab_size}")
            descriptors.append(f"d_model={layer.d_model}")
            descriptors.append(f"max_length={layer.max_length}")
        if hasattr(layer, 'd_model'):
            descriptors.append(f"d_model={layer.d_model}")
        if hasattr(layer, 'd_ff'):
            descriptors.append(f"d_ff={layer.d_ff}")
        if hasattr(layer, 'n_heads'):
            descriptors.append(f"n_heads={layer.n_heads}")
        if hasattr(layer, 'dropout'):
            descriptors.append(f"dropout={layer.dropout}")
        if hasattr(layer, 'ff_activation'):
            descriptors.append(f"ff_activation={layer.ff_activation}")
        
        result.append(descriptors)
    return result


