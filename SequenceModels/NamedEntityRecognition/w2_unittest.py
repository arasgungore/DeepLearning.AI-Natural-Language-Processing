# -*- coding: utf-8 -*-
import os 
import numpy as np
import pandas as pd
import random as rnd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.nn import log_softmax
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import tensorflow.keras as keras
import tensorflow as tf
# UNIT TEST
# test data_generator


TAG_MAP_SET = set({'B-art',
 'B-eve',
 'B-geo',
 'B-gpe',
 'B-nat',
 'B-org',
 'B-per',
 'B-tim',
 'I-art',
 'I-eve',
 'I-geo',
 'I-gpe',
 'I-nat',
 'I-org',
 'I-per',
 'I-tim',
 'O'})

def test_get_sentence_vectorizer(target):
    
    test_cases = [

        {
            "name": "standardize_check",
            "input": [[" "]],
            "expected": {
                "expected_output_standardize": None

            }


        },
        {
         "name": "general_check_1",
         "input": [[['I like oranges'], ['Peter, son of Parker, is doing good']]],
         "expected": {
             "expected_output_size": 2,
             "expected_output_vocab_size": 12,
         }
        }
     ,
        {    
        "name": "general_check_2",
        "input":[[['Bananas, apples and oranges'], ['Grapefruit, blueberry and strawbarry']]],
        "expected": {
            "expected_output_size": 2,
            "expected_output_vocab_size": 9,
        }
        }
        
    ]
    
    successful_cases = 0
    failed_cases = []
    
    for test_case in test_cases:
        gen_result = target(*test_case["input"])

        if test_case['name'] == 'standardize_check':
            # Checking sdtandardize parameter
            try:
                assert gen_result[0]._standardize == test_case["expected"]["expected_output_standardize"]
                successful_cases += 1
            except:
                failed_cases.append(
                    {
                        "name": test_case["name"],
                        "expected": test_case["expected"]["expected_output_standardize"],
                        "got": gen_result[0]._standardize,
                    }
                )
                print(
                    f"Wrong standardize parameter.\n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}"
                    
                )
            continue
        
        # Checking output size:
        try:
            assert np.allclose(len(gen_result),test_case["expected"]["expected_output_size"])
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["expected_output_size"],
                    "got": len(gen_result),
                }
            )
            print(
                f"Wrong output size.\n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}"
            )
        # Checking output vocabulary size
        try:
            assert np.allclose(len(gen_result[1]), test_case["expected"]["expected_output_vocab_size"])
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["expected_output_vocab_size"],
                    "got": len(gen_result[1]),
                }
            )
            print(
                f"Wrong output vocabulary size.\n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}"
            )
            
    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")

    # return failed_cases, len(failed_cases) + successful_cases        
                    
def test_label_vectorizer(target):
    
    test_cases = [
        {
            "name": "default_input_check",
            "input":[['O O O O O O O O O', 'O O I-geo'], {'O':1, 'I-geo':2}],
            "expected": {
                "expected_output_type":np.ndarray ,
                "expected_output_dtype":type(np.dtype('int32')),
                "expected_output": np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1],
       [1, 1, 2, -1, -1, -1, -1, -1, -1]], dtype=np.int32)
            }
            
        }
        
    ]
    
    successful_cases = 0
    failed_cases = []
    
    for test_case in test_cases:
        
        try:
            gen_result = target(*test_case['input'])
        except KeyError:
            print("\033[91m", "Could not run the test due to an exception in function call. Please doublecheck how you are splitting the labels to map them using tag_map.")   
            return
        except Exception as e:
            print("\033[91m", f"There was a problem running your function. Please try to run it with some examples. The issue is: {e}")
            return
        try:
            assert isinstance(gen_result, test_case['expected']['expected_output_type'])
            successful_cases += 1
        except:
            failed_cases.append(
             {
                 "name": test_case['name'],
                 "expected": test_case['expected']['expected_output_type'],
                 "got": type(gen_result)
                 
             }
                
            )
            print(
             f"Incorrect output type.\n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}"   
            )
        
        try:
            assert isinstance(gen_result.dtype, test_case['expected']['expected_output_dtype'])
            successful_cases += 1
        except:
            failed_cases.append(
             {
                 "name": test_case['name'],
                 "expected": test_case['expected']['expected_output_dtype'],
                 "got": gen_result.dtype
                 
             }
                
            )
            print(
             f"Incorrect output data type.\n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."   
            )
            
        try:
            assert np.allclose(gen_result, test_case['expected']['expected_output'])
            successful_cases += 1
        except:
            
            failed_cases.append(
                {
                    "name": test_case['name'],
                    "expected": test_case['expected']['expected_output'],
                    "got": gen_result
           
                }
                
            )
            print(
                f"Wrong output. Please review your code. Remember to pass the argument padding = 'post'.\n\tExpected: {failed_cases[-1].get('expected')}.\n\t Got: {failed_cases[-1].get('got')}."
            )

            
    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")    
        
def test_NER(target):
    
    test_cases = [
        {
            "name": "check_embedding_layer",
            "input": {"len_tags":1,"vocab_size":10},
            "expected":
            {
                "expected_mask_zero": True,
                "expected_input_shape": (None, None),
                "expected_input_dim": 11,
                "expected_output_shape": (None, None, 50),
                "expected_output_dim": 50,
                "expected_layer_type": "embedding"                
            }
            
        },
        
        {
            "name": "check_lstm_layer",
            "input":{"len_tags":30,"vocab_size":31},
            "expected":
            {
                "expected_return_sequences": True,
                "expected_input_shape": (None, None, 50),
                "expected_output_shape": (None, None, 50),
                "expected_layer_type": "lstm"                
            }            
            
        },
        {
            "name": "check_dense_layer",
            "input": {"len_tags":20,"vocab_size":25},
            "expected":
            {
                "expected_input_shape": (None, None, 50),
                "expected_output_shape": (None, None, 20),
                "expected_activation": log_softmax,
                "expected_layer_type": "dense"                
            }
            
        }
    ]
    
    successful_cases = 0
    failed_cases = []
    
    for i, test_case in enumerate(test_cases):
        
        gen_result = target(**test_case['input']).layers[i]
        for expected_name, expected_value in test_case['expected'].items():
            attribute = expected_name.split("expected_")[-1]
            expected_name = expected_name.split('expected_')[-1]
            if 'layer' in attribute:
                attribute = 'name'
            if isinstance(expected_value, (type(True),type(1))):
                function_to_compare = np.allclose
            elif isinstance(expected_value, str):
                function_to_compare = lambda x,y: x in y
            else:
                function_to_compare = lambda x,y: x == y
            try:
                assert function_to_compare(expected_value, getattr(gen_result,attribute))
                successful_cases += 1
            except:
                failed_cases.append(
               {
                   "name": test_case['name'],
                   "expected": expected_value,
                   "got": getattr(gen_result,attribute)
               }
                )
                print(
                    f"Wrong {expected_name} in test {failed_cases[-1].get('name')}.\n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}"
                )
                

            
    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")    
                    
        

def test_masked_loss(target):
    np.random.seed(0)
    test_cases = [
     {
         "name": "default_check",
         "input": [[1,1],[[.1,.2],[.2,.3]]],
         "expected":
         {
             "expected_output_type":"tensor",
             "expected_output_shape":[],
             "expected_output_dtype":type(tf.float32),
             "expected_output":0.64439666
         }
     },
        {
            "name": "output_check_1",
            "input":[np.random.randint(0, 17, size = (1,10)),np.random.rand(1,10,17)],
            "expected":
            {
                "expected_output":2.9175894299246647
            },
             
        },
        {
            "name": "output_check_2",
            "input":[np.random.randint(0, 17, size = (1,10)),np.random.rand(1,10,17)],
            "expected":
            {
                "expected_output": 3.0363126032952374
            },
             
        },        
        {
            "name": "output_check_3",
            "input":[np.random.randint(0, 17, size = (1,10)),np.random.rand(1,10,17)],
            "expected": 
            {
                "expected_output": 2.9665916003363666
            },
             
        },              
        {
            "name": "output_check_4",
            "input":[np.random.randint(0, 17, size = (1,10)),np.random.rand(1,10,17)],
            "expected":
            {
                "expected_output": 2.955409114865017,
            }
             
        },
        {
            "name": "output_check_5",
            "input":[np.random.randint(0, 17, size = (1,10)),np.random.rand(1,10,17)],
            "expected": 
            {
                "expected_output": 2.8963096022811365
            },
             
        }
 
    ]
    
    successful_cases = 0
    failed_cases = []
    for test_case in test_cases:        
        gen_result = target(*test_case['input'])
        
        for expected_att, expected_val in test_case['expected'].items():
            try:
                if expected_val == 'tensor':
                    assert tf.is_tensor(gen_result)
                    successful_cases += 1
                if 'shape' in expected_att:
                    assert list(gen_result.shape.as_list()) == expected_val
                    successful_cases += 1
                if 'dtype' in expected_att:
                    assert isinstance(gen_result.dtype, expected_val)
                    successful_cases += 1
                if expected_att == 'expected_output':
                    assert np.allclose(gen_result, expected_val)
                    successful_cases += 1
            except:
                failed_cases.append(
                 {
                     "name": test_case['name'],
                     "expected": expected_val,
                     "got":type(gen_result) if expected_val == 'tensor' else gen_result.shape if 'shape' in expected_att else gen_result.dtype if 'dtype' in expected_att else gen_result
                     
                 }
                    
                )
                specific_test = expected_att.split("expected_")[-1]
                print(
                    f"Failed in test: {failed_cases[-1].get('name')}. Wrong {specific_test}.\n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}"
                )

                
            
    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")    
                    
        
def test_masked_accuracy(target):
    np.random.seed(1)
    test_cases = [
     {
         "name": "default_check",
         "input": [np.array([1,1]),[[.1,.2],[.2,.3]]],
         "expected":
         {
             "expected_output_type":"tensor",
             "expected_output_shape":[],
             "expected_output_dtype":type(tf.float32),
             "expected_output":1.0
         }
     },
        {
            "name": "output_check_1",
            "input":[np.random.randint(-1, 3, size = (1,5)),np.random.rand(1,5,3)],
            "expected":
            {
                "expected_output":0.33333334
            },
             
        },
        {
            "name": "output_check_2",
            "input":[np.random.randint(-1, 3, size = (1,5)),np.random.rand(1,5,3)],
            "expected":
            {
                "expected_output": 0.25
            },
             
        },        
        {
            "name": "output_check_3",
            "input":[np.random.randint(-1, 3, size = (1,5)),np.random.rand(1,5,3)],
            "expected": 
            {
                "expected_output": 0.33333334
            },
             
        },              
        {
            "name": "output_check_4",
            "input":[np.random.randint(-1, 3, size = (1,5)),np.random.rand(1,5,3)],
            "expected":
            {
                "expected_output": 0.75,
            }
             
        },
        {
            "name": "output_check_5",
            "input":[np.random.randint(-1, 3, size = (1,5)),np.random.rand(1,5,3)],
            "expected": 
            {
                "expected_output": 0.6666667
            },
             
        }
 
    ]
    
    successful_cases = 0
    failed_cases = []
    for test_case in test_cases:        
        gen_result = target(*test_case['input'])
        
        for expected_att, expected_val in test_case['expected'].items():
            try:
                if expected_val == 'tensor':
                    assert tf.is_tensor(gen_result)
                    successful_cases += 1
                if 'shape' in expected_att:
                    assert list(gen_result.shape.as_list()) == expected_val
                    successful_cases += 1
                if 'dtype' in expected_att:
                    assert isinstance(gen_result.dtype, expected_val)
                    successful_cases += 1
                if expected_att == 'expected_output':
                    assert np.allclose(gen_result, expected_val)
                    successful_cases += 1
            except:
                failed_cases.append(
                 {
                     "name": test_case['name'],
                     "expected": expected_val,
                     "got":type(gen_result) if expected_val == 'tensor' else gen_result.shape if 'shape' in expected_att else gen_result.dtype if 'dtype' in expected_att else gen_result
                     
                 }
                    
                )
                specific_test = expected_att.split("expected_")[-1]
                print(
                    f"Failed in test: {failed_cases[-1].get('name')}. Wrong {specific_test}.\n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}"
                )

                
            
    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")    
                    
        
            
           

#
# UNIT TEST
# test train_model

# like_pred: Creates a prediction like matrix base on a set of true labels
def test_predict(target, model, sentence_vectorizer, tag_map):
    np.random.seed(0)

    test_cases = [
                    {
                        "name": "default_check",
                        "input": 
                        {
                            "sentence": "Peter Navaro , is a great man !",
                            "model": model,
                            "sentence_vectorizer": sentence_vectorizer,
                            "tag_map": tag_map

                        },
                        "expected":

                        {
                            "expected_output_type": list,
                            "expected_output_size": 8,
                            "expected_output_values": TAG_MAP_SET,

                        }
                    }


    ]

    successful_cases = 0
    failed_cases = []

    break_test = False

    for test_case in test_cases:

        try:
            gen_result = target(**test_case['input'])
            try:
                assert isinstance(gen_result, test_case['expected']['expected_output_type'])
                successful_cases += 1
            except:
                failed_cases.append(
                    {
                        "name": test_case['name'],
                        "expected": test_case['expected']['expected_output_type'],
                        "got": type(gen_result)



                    }
                )
                print(f"Wrong output type.\n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}")
            
            try:
                assert np.allclose(len(gen_result), test_case['expected']['expected_output_size'])
                successful_cases += 1
            except:
                failed_cases.append(
                    {
                        "name": test_case['name'],
                        "expected": test_case['expected']['expected_output_size'],
                        "got": len(gen_result)
                    }
                )
                print(f"Wrong output size.\n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}.\nCheck the axis you are passing to compute the argmax and how you access the prediction values.")
            
            gen_result_set = set(gen_result)
            try:
                assert gen_result_set.issubset(test_case['expected']['expected_output_values'])
                successful_cases += 1
            except AssertionError:
                failed_cases.append(
                    {
                        "name": test_case['name'],
                        "expected": test_case['expected_output_values'],
                        "got": gen_result_set

                    }

                )
                print(f"Wrong output values. They must be a subset of {failed_caes[-1].get('expected')}. But they are elements of {failed_cases[-1].get('got')}.")
            except Exception as e:
                failed_cases.append(
                    {
                        "name": test_case['name'],
                        "expected": test_case['expected_output_values'],
                        "got": e

                    }
                )
                print("Expected output values test could not be performed. Please doublecheck your output. Output elements should be a subset of {TAG_MAP_SET}, but it was impossible to convert the output to a set. Error:\n\t{e}.")

        except ValueError as e:
            break_test = True
            if "Exception encountered when calling layer" in e.args[0]:
                failed_cases.append(
                    {
                        "name": "invalid_shape_error",
                        "expected": None,
                        "got": e
                    }

                )
                print(f"Your function could not be tested due an error. Please make sure you are passing the correct tensor to the model call. You need to expand its dimension before calling the model.")
            else:
                failed_cases.append(
                    {
                        "name": "invalid_shape_error",
                        "expected": None,
                        "got": e
                    }

                )
                print(f"Your function could not be tested due an error. The error is:\n\t{failed_cases[-1].get('got')}")

        except Exception as e:
            break_test = True
            failed_cases.append(
                {
                    "name": "invalid_shape_error",
                    "expected": None,
                    "got": e
                }

            )
            print(f"Your function could not be tested due an error. The exception is:\n\t{failed_cases[-1].get('got')}")
        
            
    if break_test:
        print("\033[91m", "Test failed.")
    else:
        if len(failed_cases) == 0:
            print("\033[92m All tests passed")
        else:
            print("\033[92m", successful_cases, " Tests passed")
            print("\033[91m", len(failed_cases), " Tests failed")

    #  return failed_cases, len(failed_cases) + successful_cases

