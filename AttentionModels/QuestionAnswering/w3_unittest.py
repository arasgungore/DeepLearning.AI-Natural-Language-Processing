import sys
import itertools
import numpy as np
import traceback
import test_utils
from utils import EncoderBlock

import tensorflow as tf
from tensorflow.keras.layers import MultiHeadAttention, ReLU, Attention, LayerNormalization, Input
import tensorflow_text as tf_text

from dlai_grader.grading import test_case, object_to_grade
from types import ModuleType, FunctionType
import transformer_utils

def testing_rnd():
    def dummy_generator():
        vals = np.linspace(0, 1, 10)
        cyclic_vals = itertools.cycle(vals)
        for _ in range(100):
            yield next(cyclic_vals)

    dumr = itertools.cycle(dummy_generator())

    def dummy_randomizer():
        return next(dumr)

    return dummy_randomizer

f = open("./models/sentencepiece.model", "rb")
tokenizer = tf_text.SentencepieceTokenizer(f.read(), out_type=tf.int32)

# +
def test_tokenize_and_mask(target):
    
    t = test_case()
    if not isinstance(target, FunctionType):
            t.failed = True
            t.msg = "target has incorrect type"
            t.want = FunctionType
            t.got = type(target)
            return [t]
    
    text1 = b"Beginners BBQ Class Taking Place in Missoula!"
    text2 = b"Foil plaid lycra and spandex shortall with metallic slinky insets."
    text3 = 'Beginners BBQ Class Taking Place in Missoula!\nDo you want to get better at making delicious BBQ? You will have the opportunity, put this on your calendar now. Thursday, September 22nd join World Class BBQ Champion, Tony Balay from Lonestar Smoke Rangers.'


    test_cases = [{"text": text1, "noise": 0,
              "expected_output":([12847, 277, 15068, 4501, 3, 12297, 3399, 16, 5964, 7115, 9, 55], [1])},
             {"text": text1, "noise": 0.2,
              "expected_output": ([31999, 15068, 4501, 3, 12297, 3399, 16, 5964, 7115, 31998], [31999, 12847, 277, 31998, 9, 55, 1])},
             {"text": text1, "noise": 0.5,
              "expected_output": ([31999, 12297, 3399, 16, 5964, 7115, 31998], [31999, 12847, 277, 15068, 4501, 3, 31998, 9, 55, 1])},
             {"text": text2, "noise": 0.1,
              "expected_output": ([31999, 173, 30772, 3, 120, 2935, 11, 8438, 26, 994, 31998, 1748, 28, 18813, 3, 7, 4907, 63, 16, 2244, 31997, 5], [31999, 4452, 31998, 710, 31997, 7, 1])},
             {"text": text2, "noise": 0.2,
              "expected_output": ([31999, 30772, 3, 120, 2935, 11, 8438, 26, 994, 31998, 28, 18813, 3, 7, 4907, 63, 16, 2244, 31997], [31999, 4452, 173, 31998, 710, 1748, 31997, 7, 5, 1])},
             {"text": text2, "noise": 1.0,
              "expected_output": ([31999, 994, 31998, 2244, 31997], [31999, 4452, 173, 30772, 3, 120, 2935, 11, 8438, 26, 31998, 710, 1748, 28, 18813, 3, 7, 4907, 63, 16, 31997, 7, 5, 1])},
             {"text": text3, "noise": 0.15, 
              "expected_output": ([31999, 15068, 4501, 3, 12297, 3399, 16, 5964, 7115, 31998, 531, 25, 241, 12, 129, 394, 44, 492, 31997, 58, 148, 56, 43, 8, 1004, 6, 474, 31996, 39, 4793, 230, 5, 2721, 6, 1600, 1630, 31995, 1150, 4501, 15068, 16127, 6, 9137, 2659, 5595, 31994, 782, 3624, 14627, 15, 12612, 277, 5], [31999, 12847, 277, 31998, 9, 55, 31997, 3326, 15068, 31996, 48, 30, 31995, 727, 1715, 31994, 45, 301, 1])}
            ]
    cases = []

    for case in test_cases:
        output = target(case.get('text'), 
                        noise=case.get('noise'),
                        randomizer=testing_rnd(),
                        tokenizer=tokenizer)
        t = test_case() #inps, targs
        if not isinstance(output[0], list):
            t.failed = True
            t.msg = "Wrong type. inps extected to be a list"
            t.want = tf.Tensor
            t.got = type(output[0])
        cases.append(t)
        
        t = test_case()
        if not isinstance(output[1], list):
            t.failed = True
            t.msg = "Wrong type. args extected to be a list"
            t.want = tf.Tensor
            t.got = type(output[1])
        cases.append(t)
        
        t = test_case()
        if len(case.get('expected_output')[0]) != len(output[0]):
            t.failed = True
            t.msg = "Wrong length for inps"
            t.want = len(case.get('expected_output')[0])
            t.got = len(output[0])
        cases.append(t)
        
        t = test_case()
        if len(case.get('expected_output')[1]) != len(output[1]):
            t.failed = True
            t.msg = "Wrong length for args"
            t.want = len(case.get('expected_output')[1])
            t.got = len(output[1])
        cases.append(t)

        t = test_case()
        if len(output[0])>0 and not (isinstance(output[0][0], (int, np.int32, np.int64, type(tf.constant(1.0))))):
            t.failed = True
            t.msg = "Wrong type. inps extected to be a int"
            t.want = type(1)
            t.got = type(output[0][0])
        cases.append(t)

        t = test_case()
        if len(output[1])>0 and not (isinstance(output[1][0], (int, np.int32, np.int64, type(tf.constant(1.0))))):
            t.failed = True
            t.msg = "Wrong type. args extected to be a int qq"
            t.want = type(1)
            t.got = type(output[1][0])
        cases.append(t)

        t = test_case()
        if not np.array_equal(output[0], case.get('expected_output')[0]):
            t.failed = True
            t.msg = f"Wrong values for inps for input: {case.get('text')}"
            t.want = case.get('expected_output')[0]
            t.got = output[0]
        cases.append(t)
    
        t = test_case()
        if not np.array_equal(output[1], case.get('expected_output')[1]):
            t.failed = True
            t.msg = f"Wrong values for args for input: {case.get('text')}"
            t.want = case.get('expected_output')[1]
            t.got = output[1]
        cases.append(t)

    for i in range(len(cases)):
        if cases[i].failed:
            print(f"\033[91mTese case {i} failed\n")
            print(cases[i])
            return
            
    print("\033[92m All tests passed")
    
def test_parse_squad(target):
    t = test_case()
    if not isinstance(target, FunctionType):
            t.failed = True
            t.msg = "target has incorrect type"
            t.want = FunctionType
            t.got = type(target)
            return [t]
    dataset1 =  [{"title": "t1", "paragraphs": [
        {"context": "very long context one", 
             "qas": [{ "question": "question is abc?",
                       "id": "1",
                       "answers": [
                             {
                               "text": "here is abc",
                               "answer_start": 8
                             },
                             {
                               "text": "abc here abc",
                               "answer_start": 0
                             }
                       ],
                       "is_impossible": False},
                     { "question": "unanswerable question?",
                       "id": "2",
                       "answers": [
                             {
                               "text": "what?",
                               "answer_start": 9
                             }
                       ],
                       "is_impossible": True},
                     { "question": "question is xyz?",
                       "id": "3",
                       "answers": [
                             {
                               "text": "here is xyz",
                               "answer_start": 9
                             }
                       ],
                       "is_impossible": False}
                      ]}]}]

    pairs = target(dataset1)   
    expected_pairs1 = (['question: question is abc? context: very long context one', 
                        'question: question is xyz? context: very long context one'], 
                       ['answer: here is abc', 'answer: here is xyz'])
    cases = []

    t = test_case()
    if not isinstance(pairs[0], list):
        t.failed = True
        t.msg = "Wrong type for returned inputs"
        t.want = list
        t.got = type(pairs[0])
    cases.append(t)
    
    t = test_case()
    if not isinstance(pairs[1], list):
        t.failed = True
        t.msg = "Wrong type for returned outputs"
        t.want = list
        t.got = type(pairs[1])
    cases.append(t)
    
    t = test_case() #inps, targs
    if len(pairs[0]) != 2:
        t.failed = True
        t.msg = "Wrong length for returned inputs"
        t.want = tf.Tensor
        t.got = type(pairs[0])
    cases.append(t)
    
    t = test_case() #inps, targs
    if len(pairs[1]) != 2:
        t.failed = True
        t.msg = "Wrong length for returned outputs"
        t.want = tf.Tensor
        t.got = type(pairs[1])
    cases.append(t)
        
    t = test_case() #inps, targs
    if not(pairs[0][0] == expected_pairs1[0][0]):
        t.failed = True
        t.msg = "Wrong input 0"
        t.want = expected_pairs1[0][0]
        t.got =  pairs[0][0]
    cases.append(t)
    
    t = test_case() #inps, targs
    if not(pairs[0][1] == expected_pairs1[0][1]):
        t.failed = True
        t.msg = "Wrong input 1"
        t.want = expected_pairs1[0][1]
        t.got =  pairs[0][1]
    cases.append(t)
    
    t = test_case() #inps, targs
    if not(pairs[1][0] == expected_pairs1[1][0]):
        t.failed = True
        t.msg = "Wrong output 0"
        t.want = expected_pairs1[1][0]
        t.got =  pairs[1][0]
    cases.append(t)
    
    t = test_case() #inps, targs
    if not(pairs[1][1] == expected_pairs1[1][1]):
        t.failed = True
        t.msg = "Wrong output 1"
        t.want = expected_pairs1[1][1]
        t.got =  pairs[1][1]
    cases.append(t)
    
    print("\033[92m All tests passed")
    
def test_answer_question(target):
    
    t = test_case()
    if not isinstance(target, FunctionType):
            t.failed = True
            t.msg = "target has incorrect type"
            t.want = FunctionType
            t.got = type(target)
            return [t]
    
    # Define the model parameters
    num_layers = 2
    embedding_dim = 128
    fully_connected_dim = 128
    num_heads = 2
    positional_encoding_length = 256

    encoder_vocab_size = int(tokenizer.vocab_size())
    decoder_vocab_size = encoder_vocab_size

    # Initialize the model
    modelx = transformer_utils.Transformer(
        num_layers, 
        embedding_dim, 
        num_heads, 
        fully_connected_dim,
        encoder_vocab_size, 
        decoder_vocab_size, 
        positional_encoding_length, 
        positional_encoding_length,
    )
    
    if False:
        print("Not all tests were performed due to missing files. Don't worry, this has no impact on the assignment and we are working to fix it.")
    else:
        modelx.load_weights('./pretrained_models/model_qa3')

        question = "question: How many are this? context: This is five."
        result = tokenizer.detokenize(target(question, modelx, tokenizer)).numpy()[0].decode()
        cases = []

        t = test_case() #inps, targs
        if not ("answer:" in result):
            t.failed = True
            t.msg = "Wrong preamble"
            t.want = "answer:"
            t.got =  result
        cases.append(t)

        if not ("five" in result):
            t.failed = True
            t.msg = "Wrong answer"
            t.want = "five"
            t.got =  result
        cases.append(t)


        question = "question: When did that happen? context: That happen on August 17, 1715"
        result = tokenizer.detokenize(target(question, modelx, tokenizer)).numpy()[0].decode()
        t = test_case() #inps, targs
        if not ("answer:" in result):
            t.failed = True
            t.msg = "Wrong preamble"
            t.want = "answer:"
            t.got =  result
        cases.append(t)

        if not ("August" in result):
            t.failed = True
            t.msg = "Wrong answer"
            t.want = "August"
            t.got =  result
        cases.append(t)

        question = "question: Who is the king? context: In this country the king is Charles from here in advance"
        result = tokenizer.detokenize(target(question, modelx, tokenizer)).numpy()[0].decode()

        if not ("answer:" in result):
            t.failed = True
            t.msg = "Wrong preamble"
            t.want = "answer:"
            t.got =  result
        cases.append(t)

        if not ("Charles V" in result):
            t.failed = True
            t.msg = "Wrong answer"
            t.want = "Charles V"
            t.got =  result
        cases.append(t)

        for i in range(len(cases)):
            if cases[i].failed:
                print(f"\033[91mTese case {i} failed\n")
                print(cases[i])
                return

    print("\033[92m All tests passed")
