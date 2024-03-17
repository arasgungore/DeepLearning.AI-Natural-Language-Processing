import re
import numpy as np
import tensorflow as tf


def test_build_vocabulary(target):

    test_cases = [
        {
            "name": "simple_test_check1",
            "input": {
                "corpus": [['a']],
            },
            "expected": {
                "output_list": {'': 0, '[UNK]': 1, 'a': 2},
                "output_type": type(dict()),
            },
        },
        {
            "name": "simple_test_check2",
            "input": {
                "corpus": [['a', 'aa'], ['a', 'ab'], ['ccc']],
            },
            "expected": {
                "output_list": {'': 0, '[UNK]': 1, 'a': 2, 'aa': 3, 'ab': 4, 'ccc': 5},
                "output_type": type(dict()),
            },
        },
    ]

    failed_cases = []
    successful_cases = 0

    for test_case in test_cases:

        result = target(**test_case["input"])

        try:
            assert result == test_case["expected"]["output_list"]
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": "simple_test_check",
                    "expected": test_case["expected"]["output_list"],
                    "got": result,
                }
            )
            print(
                f"Output does not match with expected values. Maybe you can check the value you are using for unk_token variable. Also, try to avoid using the global dictionary Vocab.\n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:

            assert isinstance(result, test_case["expected"]["output_type"])
            successful_cases += 1

        except:
            failed_cases.append(
                {
                    "name": "simple_test_check",
                    "expected": test_case["expected"]["output_list"],
                    "got": result,
                }
            )
            print(
                f"Output object does not have the correct type.\n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")


def test_max_length(target):

    test_cases = [
        {
            "name": "simple_test_check1",
            "input": {
                "training_x": [['cccc']],
                "validation_x": [['a', 'aa'], ['a', 'ab'], ['cccc']],
            },
            "expected": {
                "output_list": 2,
                "output_type": type(1),
            },
        },
        {
            "name": "simple_test_check2",
            "input": {
                "training_x": [['a', 'aa'], ['a', 'ab'], ['ccc'], ['ddd']],
                "validation_x": [['a', 'aa'], ['a', 'ab', 'ac'], ['ccc']],
            },
            "expected": {
                "output_list": 3,
                "output_type": type(1),
            },
        },
    ]

    failed_cases = []
    successful_cases = 0

    for test_case in test_cases:

        result = target(**test_case["input"])

        try:
            assert result == test_case["expected"]["output_list"]
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": "simple_test_check",
                    "expected": test_case["expected"]["output_list"],
                    "got": result,
                }
            )
            print(
                f"Output does not match with expected values. Make sure you are measuring the length of tweets and not the length of datasets or words. Expected: {failed_cases[-1].get('expected')}.\n Got: {failed_cases[-1].get('got')}."
            )

        try:

            assert isinstance(result, test_case["expected"]["output_type"])
            successful_cases += 1

        except:
            failed_cases.append(
                {
                    "name": "simple_test_check",
                    "expected": test_case["expected"]["output_list"],
                    "got": result,
                }
            )
            print(
                f"Output object does not have the correct type.\n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")


def test_padded_sequence(target):

    test_cases = [
        {
            "name": "simple_test_check1",
            "input": {
                "tweet": ['a', 'a', 'aaa', 'cats'],
                "vocab_dict": {
                     '': 0,
                     '[UNK]': 1,
                     'a': 2,
                     'aa': 3,
                     'aaa': 4,
                     'aaaa': 5,
                },
                "max_len": 5,
                "unk_token": '[UNK]'
            },
            "expected": {
                "output_list": [2, 2, 4, 1, 0],
                "output_type": type([]),
            },
        },
    ]

    failed_cases = []
    successful_cases = 0

    for test_case in test_cases:

        result = target(**test_case["input"])

        try:
            assert result == test_case["expected"]["output_list"]
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": "simple_test_check",
                    "expected": test_case["expected"]["output_list"],
                    "got": result,
                }
            )
            print(
                f"Output does not match with expected values. Maybe you can check the value you are using for unk_token variable. Also, try to avoid using the global dictionary Vocab.\n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert isinstance(result, test_case["expected"]["output_type"])
            successful_cases += 1

        except:
            failed_cases.append(
                {
                    "name": "simple_test_check",
                    "expected": test_case["expected"]["output_list"],
                    "got": result,
                }
            )
            print(
                f"Output object does not have the correct type.\n Expected: {failed_cases[-1].get('expected')}.\n Got: {failed_cases[-1].get('got')}."
            )
            
        try:
            assert len(result) == len(test_case["expected"]["output_list"])
            successful_cases += 1

        except:
            failed_cases.append(
                {
                    "name": "simple_test_check",
                    "expected": test_case["expected"]["output_list"],
                    "got": result,
                }
            )
            print(
                f"Output object does not have the correct length.\n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")        


def test_relu(target):
    failed_cases = []
    successful_cases = 0

    test_cases = [
        {
            "name": "check_output1",
            "input": np.array([[-2.0, -1.0, 0.0], [0.0, 1.0, 2.0]], dtype=float),
            "expected": {
                "values": np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 2.0]]),
                "shape": (2, 3),
            },
        },
        {
            "name": "check_output2",
            "input": np.array(
                [
                    [-3.0, 1.0, -5.0, 4.0],
                    [-100.0, 3.0, -2.0, 0.0],
                    [-4.0, 0.0, 1.0, 5.0],
                ],
                dtype=float,
            ),
            "expected": {
                "values": np.array(
                    [[0.0, 1.0, 0.0, 4.0], [0.0, 3.0, 0.0, 0.0], [0.0, 0.0, 1.0, 5.0]]
                ),
                "shape": (3, 4),
            },
        },
    ]

    relu_layer = target

    for test_case in test_cases:
        result = relu_layer(test_case["input"])
        try:
            assert result.shape == test_case["expected"]["shape"]
            successful_cases += 1

        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["shape"],
                    "got": result.shape,
                }
            )
            print(
                f"Relu should not modify the input shape.\n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert np.allclose(result, test_case["expected"]["values"],)
            successful_cases += 1

        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["values"],
                    "got": result,
                }
            )
            print(
                f"Output from relu is incorrect.\n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")

    # return failed_cases, len(failed_cases) + successful_cases


def test_sigmoid(target):
    failed_cases = []
    successful_cases = 0

    test_cases = [
        {
            "name": "check_output1",
            "input": np.array([[-1000.0, -1.0, 0.0], [0.0, 1.0, 1000.0]], dtype=float),
            "expected": {
                "values": np.array([[0.0, 0.26894142, 0.5], [0.5, 0.73105858, 1.0]]),
                "shape": (2, 3),
            },
        },
        {
            "name": "check_output2",
            "input": np.array(
                [
                    [-3.0, 1.0, -5.0, 4.0],
                    [-100.0, 3.0, -2.0, 0.0],
                    [-4.0, 0.0, 1.0, 5.0],
                ],
                dtype=float,
            ),
            "expected": {
                "values": np.array(
                    [[4.74258732e-02, 7.31058579e-01, 6.69285092e-03, 9.82013790e-01],
                     [3.72007598e-44, 9.52574127e-01, 1.19202922e-01, 5.00000000e-01],
                     [1.79862100e-02, 5.00000000e-01, 7.31058579e-01, 9.93307149e-01]]
                ),
                "shape": (3, 4),
            },
        },
    ]

    relu_layer = target

    for test_case in test_cases:
        result = relu_layer(test_case["input"])
        try:
            assert result.shape == test_case["expected"]["shape"]
            successful_cases += 1

        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["shape"],
                    "got": result.shape,
                }
            )
            print(
                f"Sigmoid function should not modify the input shape.\n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert np.allclose(result, test_case["expected"]["values"],)
            successful_cases += 1

        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["values"],
                    "got": result,
                }
            )
            print(
                f"Output from sigmoid function is incorrect.\n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")

    # return failed_cases, len(failed_cases) + successful_cases


def test_Dense(target):
    failed_cases = []
    successful_cases = 0
    
    input1 = np.array([[2.0, 7.0]])
    input2 = np.array([[2.0, 7.0, -10], [-1, -2, -3]])
    
    test_cases = [
        {
            "name": "simple_test_check1",
            "input_init": {
                "n_units": 2, 
                "input_shape": input1.shape,
                "activation": lambda x: np.maximum(x, 0)
            },
            "input_forward": {
                "x": input1, 
            },
            "expected": {
                "weights": np.array(
                    [[ 0.03047171, -0.10399841],
                    [ 0.07504512,  0.09405647]]
                ),
                "output": np.array([[0.58625925, 0.45039848]]),
            },
        },
        {
            "name": "simple_test_check2",
            "input_init": {
                "n_units": 2, 
                "input_shape": input2.shape,
                "activation": lambda x: np.maximum(x, 0)
            },
            "input_forward": {
                "x": input2, 
            },
            "expected": {
                "weights": np.array(
                    [[ 0.03047171, -0.10399841],
                     [ 0.07504512,  0.09405647],
                     [-0.19510352, -0.13021795]]
                ),
                "output": np.array(
                    [[2.53729444, 1.75257799],
                     [0.40474861, 0.30653932]]),
            },
        },
        
    ]

    for test_case in test_cases:
        dense_layer = target(**test_case["input_init"])
        result = dense_layer(**test_case["input_forward"])
        
        try:
            assert dense_layer.weights.shape == test_case["expected"]["weights"].shape
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["weights"].shape,
                    "got": dense_layer.weights.shape,
                }
            )
            print(
                f"Weights matrix has the incorrect shape.\n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )
            
        try:
            assert np.allclose(dense_layer.weights, test_case["expected"]["weights"])
            successful_cases += 1

        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["weights"],
                    "got": dense_layer.weights,
                }
            )
            print(
                f"The weights did not initialize correctly. Make sure you did not change the random seed.\n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert np.allclose(result, test_case["expected"]["output"],)
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["output"],
                    "got": result,
                }
            )
            print(
                f"Dense layer produced incorrect output. Check your weights or your output computation.\n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert isinstance(result, type(test_case["expected"]["output"]))
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": type(test_case["expected"]["output"]),
                    "got": type(result),
                }
            )
            print(
                f"Output object has the incorrect type.\n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")

    # return failed_cases, len(failed_cases) + successful_cases


def test_model(target):
    successful_cases = 0
    failed_cases = []
    
    dummy_layers = [
        tf.keras.layers.Embedding(1, 2, input_length=3),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(2, activation='relu')
    ]
    dummy_model = tf.keras.Sequential(dummy_layers)
    dummy_model.compile()
    
    test_cases = [
        {
            "name": "simple_test_check1",
            "input": {"num_words": 20, "embedding_dim": 16, "max_len": 5},
            "expected": {
                "type": type(dummy_model),
                "no_layers": 3,
                "layer_1_type": type(dummy_layers[0]),
                "layer_1_input_dim": 20,
                "layer_1_input_length": 5,
                "layer_1_output_dim": 16,
                "layer_2_type": type(dummy_layers[1]),
                "layer_3_type": type(dummy_layers[2]),
                "layer_3_output_shape": (None, 1),
                "layer_3_activation": tf.keras.activations.sigmoid
            },
        },
        
    ]

    for test_case in test_cases:
        
        model = target(**test_case["input"])

        try:
            assert isinstance(model, test_case["expected"]["type"])
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["type"],
                    "got": type(model),
                }
            )

            print(
                f"Wrong type of the returned model.\n\tGot: {failed_cases[-1].get('got')},\n\tExpected: {failed_cases[-1].get('expected')}"
            )

        try:
            assert len(model.layers) == test_case["expected"]["no_layers"]
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["no_layers"],
                    "got": len(model.layers),
                }
            )
            print(
                f"The model has an incorrect number of layers.\n\tGot: {failed_cases[-1].get('got')},\n\tExpected:{failed_cases[-1].get('expected')}"
            )

        try:
            assert isinstance(model.layers[0], test_case["expected"]["layer_1_type"])
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["layer_1_type"],
                    "got": type(model.layers[0]),
                }
            )
            print(
                f"The first layer has incorrect type.\n\tGot: {failed_cases[-1].get('got')},\n\tExpected: {failed_cases[-1].get('expected')}"
            )
            
        try:
            assert model.layers[0].input_dim == test_case["expected"]["layer_1_input_dim"]
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["layer_1_input_dim"],
                    "got": model.layers[0].input_dim,
                }
            )
            print(
                f"The first layer has wrong input dimensions.\n\tGot: {failed_cases[-1].get('got')},\n\tExpected: {failed_cases[-1].get('expected')}"
            ) 
            
        try:
            assert model.layers[0].input_length == test_case["expected"]["layer_1_input_length"]
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["layer_1_input_length"],
                    "got": model.layers[0].input_length,
                }
            )
            print(
                f"The first layer has wrong input length.\n\tGot: {failed_cases[-1].get('got')},\n\tExpected: {failed_cases[-1].get('expected')}"
            )      
            
        try:
            assert model.layers[0].output_dim == test_case["expected"]["layer_1_output_dim"]
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["layer_1_output_dim"],
                    "got": model.layers[0].output_dim,
                }
            )
            print(
                f"The first layer has wrong output (embedding) dimension.\n\tGot: {failed_cases[-1].get('got')},\n\tExpected: {failed_cases[-1].get('expected')}"
            )

        try:
            assert isinstance(model.layers[1], test_case["expected"]["layer_2_type"])
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["layer_2_type"],
                    "got": type(model.layers[1]),
                }
            )
            print(
                f"The second layer has incorrect type.\n\tGot: {failed_cases[-1].get('got')},\n\tExpected: {failed_cases[-1].get('expected')}",
                test_case["expected"]["expected_type"],
            )
                              
        try:
            assert isinstance(model.layers[2], test_case["expected"]["layer_3_type"])
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["layer_3_type"],
                    "got": type(model.layers[2]),
                }
            )
            print(
                f"The third layer has incorrect type.\n\tGot: {failed_cases[-1].get('got')},\n\tExpected: {failed_cases[-1].get('expected')}"
            )

        try:
            assert model.layers[2].output_shape == test_case["expected"]["layer_3_output_shape"]
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["layer_3_output_shape"],
                    "got": model.layers[2].output_shape,
                }
            )
            print(
                f"The last layer has wrong output shape.\n\tGot: {failed_cases[-1].get('got')},\n\tExpected:{failed_cases[-1].get('expected')}"
            )
            
        try:
            assert model.layers[2].activation == test_case["expected"]["layer_3_activation"]
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["layer_3_activation"],
                    "got": model.layers[2].activation,
                }
            )
            print(
                f"The last layer has wrong output shape.\n\tGot: {failed_cases[-1].get('got')},\n\tExpected: {failed_cases[-1].get('expected')}"
            )
            
    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")

