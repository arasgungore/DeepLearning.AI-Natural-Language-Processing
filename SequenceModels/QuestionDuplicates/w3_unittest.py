from types import ModuleType, FunctionType, FunctionType
from typing import Any, Callable, List, Optional, Union
from typing import Dict, List, Optional
from dlai_grader.grading import test_case, LearnerSubmission, object_to_grade
from dlai_grader.types import grading_function, grading_wrapper
from dlai_grader.grading import compute_grading_score

import pickle
import numpy as np
import random as rnd
import keras
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import TextVectorization
from tensorflow.data import Dataset
import os


grading_function = Callable[[Any], List[test_case]]
learner_submission = Union[ModuleType, LearnerSubmission]
grading_wrapper = Callable[[learner_submission, Optional[ModuleType]], grading_function]

def print_results(test_cases):
    num_cases = len(test_cases)
    failed_cases = [t for t in test_cases if t.failed == True]
    num_failed = len(failed_cases)
    if num_failed==0:
        print("\033[92mAll tests passed!")
    else:
        for failed_case in failed_cases:
            print(f"{failed_case.msg}\n\tExpected:{failed_case.want},\n\tGot:{failed_case.got}.\n")
        print(f"\033[92m{num_cases-num_failed} tests passed")
        print(f"\033[91m{num_failed} tests failed")    
		
# Test Siamese
def test_Siamese(learner_func):

    text_vectorization = TextVectorization()
    text_vectorization.adapt(['test vocabulary'])
    inpt = {"vocab_size": 41699, "d_feature": 128}
    
    t = test_case()
    try:
        model = learner_func(text_vectorization, **inpt)
    except Exception as e:
        t.failed = True
        t.msg = f"There was an error evaluating the `Siamese` function. "
        t.expected = "No exceptions"
        t.got = f"{str(e)}"
        return [t]
    
    cases: List[test_case] = []

    # Test output type
    t = test_case()
    if not isinstance(model, keras.models.Model):
        t.failed = True
        t.msg = "Model returned by `Siamese` has incorrect type."
        t.want = keras.models.Model
        t.got = type(model)
        return [t]
    cases.append(t)
    
		
    # Test layers types
    expected_layers_type = [
                keras.layers.InputLayer,
                keras.layers.InputLayer,
                keras.models.Sequential]
    for layer, expected_l_type in zip(model.layers, expected_layers_type):
        t = test_case()
        if not isinstance(layer, expected_l_type):
            t.failed = True
            t.msg = f"Layer '{layer.name}' has an incorrect type."
            t.want = expected_l_type
            t.got = type(layer)
        cases.append(t)

    # Test sequential layers type
    expected_sequential_layer_type = [keras.layers.TextVectorization,
                keras.layers.Embedding,
                keras.layers.LSTM,
                keras.layers.GlobalAveragePooling1D,
                keras.layers.Lambda]
    for (layer, expected_seq_type) in zip(model.get_layer(name='sequential').layers, expected_sequential_layer_type):
        t = test_case()
        if not isinstance(layer, expected_seq_type):
            t.failed = True
            t.msg = f"Sublayer {layer.name} has an incorrect type."
            t.want = expected_seq_type
            t.got = type(layer)
        cases.append(t)
    
    # Test expected input and output shapes
    expected_input_shape = [
                            [(None, 1)], 
                            [(None, 1)], 
                            (None,), 
                            [(None, 128), (None, 128)]
                            ]
    
    expected_output_shape = [
                            [(None, 1)], 
                            [(None, 1)], 
                            (None, 128), 
                            (None, 256)
                            ]
    for layer, input_shape, output_shape in zip(model.layers, expected_input_shape, expected_output_shape):
        t = test_case()
        if layer.input_shape != input_shape:
            t.failed = True
            t.msg = f"Layer '{layer.name}' has an incorrect input shape."
            t.want = input_shape
            t.got = layer.input_shape
        cases.append(t)
        t = test_case()
        if layer.output_shape != output_shape:
            t.failed = True
            t.msg = f"Layer '{layer.name}' has an incorrect output shape."
            t.want = output_shape
            t.got = layer.output_shape
        cases.append(t)

    # Test dimensions on a smaller model (d_feature=16)
    inpt = {"vocab_size": 200, "d_feature": 16}
    
    t = test_case()
    model = learner_func(text_vectorization, **inpt)


    expected_input_shape = [
                            [(None, 1)], 
                            [(None, 1)], 
                            (None,), 
                            [(None, 16), (None, 16)]
                            ]
    expected_output_shape = [
                            [(None, 1)], 
                            [(None, 1)], 
                            (None, 16), 
                            (None, 32)
                            ]
    for layer, input_shape, output_shape in zip(model.layers, expected_input_shape, expected_output_shape):
        t = test_case()
        if layer.input_shape != input_shape:
            t.failed = True
            t.msg = f"Layer '{layer.name}' has an incorrect input shape."
            t.want = input_shape
            t.got = layer.input_shape
        cases.append(t)
        t = test_case()
        if layer.output_shape != output_shape:
            t.failed = True
            t.msg = f"Layer '{layer.name}' has an incorrect output shape."
            t.want = output_shape
            t.got = layer.output_shape
        cases.append(t)
		
    print_results(cases)



# Test TripletLossFn
def test_TripletLoss(learner_func):

    t = test_case()
    
    cases: List[test_case] = []

    v1v2_list = [
        np.array([[0.26726124, 0.53452248, 0.80178373,0.26726124, 0.53452248, 0.80178373],
                  [0.5178918, 0.57543534, 0.63297887,-0.5178918, -0.57543534, -0.63297887],]),
        np.array([[0.26726124, 0.53452248, 0.80178373,0.32929278, 0.5488213, 0.76834982],
                    [0.64616234, 0.57436653, 0.50257071,0.64231723, 0.57470489, 0.50709255],
                    [-0.21821789, -0.87287156, -0.43643578,-0.20313388, -0.8802468, -0.42883819],
                    [0.13608276, -0.95257934, 0.27216553,0.09298683, -0.96971978, 0.22582515]]),
        np.array([[0.26726124, 0.53452248, 0.80178373,0.32929278, 0.5488213, 0.76834982],
                    [0.64616234, 0.57436653, 0.50257071,0.64231723, 0.57470489, 0.50709255],
                    [-0.21821789, -0.87287156, -0.43643578,-0.20313388, -0.8802468, -0.42883819],
                    [0.13608276, -0.95257934, 0.27216553,0.09298683, -0.96971978, 0.22582515]]),
        ]
    margin = [0.25,0.25,0.8]
    expected_loss = [0.7035077, 0.30219180776031007, 2.4262490547900892]
    
    for v1v2, m, expected in zip(v1v2_list, margin, expected_loss):
        t = test_case()
        try:
            loss = learner_func([1]*len(v1v2), v1v2, m)
        except Exception as e:
            t.failed = True
            t.msg = f"There was an error evaluating the TripletLoss function."
            t.expected = "No exceptions"
            t.got = f"{str(e)}"
            return [t]
        
        if not np.isclose(loss, expected):
            t.failed = True
            t.msg = f"Got a wrong triplet loss for inputs out: {v1v2}, and margin {m}"
            t.want = expected
            t.got = loss
        cases.append(t)
    print_results(cases)
	
# test trainer
def test_train_model(learner_func, model, lossfn):
    
    train_Q1_testing = np.array(
        [
            'Astrology : I am a Capricorn Sun Cap moon and cap rising ... what does that say about me?',
            'How can I be a good geologist?',
            'How do I read and find my YouTube comments ?',
        ],
        dtype=object,
    )

    train_Q2_testing = np.array(
        [
            "I'm a triple Capricorn (Sun, Moon and ascendant in Capricorn) What does this say about me?",
            'What should I do to be a great geologist?',
            'How can I see all my Youtube comments?',
        ],
        dtype=object,
    )
    text_vectorization = TextVectorization()
    text_vectorization.adapt(np.concatenate((train_Q1_testing, train_Q2_testing)))
    
    train_gen = Dataset.from_tensor_slices(((train_Q1_testing, train_Q2_testing),
                                        tf.constant([1]*len(train_Q1_testing)))).batch(batch_size=3)
    val_gen = Dataset.from_tensor_slices(((train_Q1_testing, train_Q2_testing),
                                        tf.constant([1]*len(train_Q1_testing)))).batch(batch_size=3)
    
    try: 
        trained_model = learner_func(model, lossfn, text_vectorization, train_gen, val_gen, train_steps = 0)
    except Exception as e:
        t.failed = True
        t.msg = f"There was an error evaluating the `train_model` function. "
        t.expected = "No exceptions"
        t.got = f"{str(e)}"
        return [t]
    
    cases: List[test_case] = []

    t = test_case()
    loss_fn = "TripletLoss"
    description = str(trained_model.loss.__name__)
    if not description == loss_fn:	
        t.failed = True
        t.msg = "fit method got wrong loss function"
        t.want = loss_fn
        t.got = description
    cases.append(t)

    t = test_case()
    if not isinstance(trained_model.optimizer, Adam):
        t.failed = True
        t.msg = "fit method got a wrong optimizer"
        t.want = Adam
        t.got = trained_model.optimizer
    cases.append(t)
    
    t = test_case()

    try: 
        trained_model = learner_func(model, lossfn, text_vectorization, train_gen, val_gen, d_feature =16, lr = 0.1, train_steps = 0)
    except Exception as e:
        t.failed = True
        t.msg = f"There was an error evaluating the `train_model` function."
        t.expected = "No exceptions"
        t.got = f"{str(e)}"
        return [t]
    
    t = test_case()
    if not np.isclose(trained_model.optimizer.learning_rate, 0.1):
        t.failed = True
        t.msg = "Wrong learning rate"
        t.want = 0.1
        t.got = trained_model.optimizer.learning_rate.numpy()
    cases.append(t)
	
    print_results(cases)


# test classification
def test_classify(learner_func, model):
    Q1_test = []
    with open("/tf/support_files/classify_fn/Q1_test_words.pkl", "rb") as f:
        while True:
            try:
                Q1_test.append(pickle.load(f))
            except EOFError:
                break  # Reached the end of the file
    Q2_test = []
    with open("/tf/support_files/classify_fn/Q2_test_words.pkl", "rb") as f:
        while True:
            try:
                Q2_test.append(pickle.load(f))
            except EOFError:
                break  # Reached the end of the file
    with open("/tf/support_files/classify_fn/y_test.pkl", "rb") as f:
        y_test = pickle.load(f)


    model = tf.keras.models.load_model('./model/trained_model.keras', safe_mode=False, compile=False)

    threshold = [0.7, 0.75, 0.7, 0.8,]
    batch_size = [512, 512, 256, 256]
    expected_acc = [0.7148571428571429, 0.7048571428571428, 0.7174285714285714, 0.682]
    expected_cm = [np.array([[1624 ,525],[ 473,  878]]),
                    np.array([[1747,  421], [ 612,  720]]),
                    np.array([[1647,  521], [ 468,  864]]),
                    np.array([[1857,  311], [ 802,  530]])]
    
    cases: List[test_case] = []
    kk = 0
    for th, bs, eacc, ecm in zip(threshold, batch_size, expected_acc, expected_cm):
        t = test_case()
        try: 
            pred_acc, cm = learner_func(Q1_test[kk:kk+3500], Q2_test[kk:kk+3500], 
                                    y_test[kk:kk+3500], th, model, bs, verbose=False)
        except Exception as e:
            t.failed = True
            t.msg = f"There was an error evaluating the `classify` function. "
            t.expected = "No exceptions"
            t.got = f"{str(e)}"
            return [t]
        
        t = test_case()
        if not np.isclose(pred_acc, eacc):
            t.failed = True
            t.msg = f"Wrong accuracy for threshold={th} and batch_size={bs}"
            t.want = eacc
            t.got = pred_acc
        cases.append(t)
        
        t = test_case()
        if not np.isclose(cm, ecm).all():
            t.failed = True
            t.msg = f"Wrong confusion matrix for threshold={th} and batch_size={bs}"
            t.want = ecm
            t.got = cm
        cases.append(t)
        
        kk =+ 1000
    
    print_results(cases)

# test predict
def test_predict(learner_func, model):
    cases: List[test_case] = []

    model = tf.keras.models.load_model('./model/trained_model.keras', safe_mode=False, compile=False)
    
    question1 = ["When will I see you?", "Do they enjoy eating the dessert?", 
                 "How does a long distance relationship work?", 
                 "How does a long distance relationship work?",
                 "Why don't we still do great music like in the 70's and 80's?", ]
    question2 = ["When can I see you again?", 
                 "Do they like hiking in the desert?", 
                 "How are long distance relationships maintained?", 
                 "How are long distance relationships maintained?", 
                 "Should I raise my young child on 80's music?",]
    threshold = [0.7, 0.7, 0.5, 0.75, 0.5]
    expected_label = [True, False, True, True, False]
    for q1, q2, th, lab in zip(question1, question2, threshold, expected_label):
        t = test_case()
        try:
            pred = learner_func(q1, q2, th, model, verbose=False)
        except Exception as e:
            t.failed = True
            t.msg = f"There was an error evaluating the `predict` function."
            t.expected = "No exceptions"
            t.got = f"{str(e)}"
            return [t]
        
        t = test_case()
        if not isinstance(pred, (np.bool_, np.ndarray)):
            t.failed = True
            t.msg = "There output of the function has wrong type"
            t.want = np.bool_
            t.got = type(pred)
            return [t]
        
        if pred != lab:
            t.failed=True
            t.msg = f"Wrong prediction for questions Q1: {q1}, Q2: {q2}"
            t.want = lab
            t.got = pred
        cases.append(t)
    print_results(cases)
