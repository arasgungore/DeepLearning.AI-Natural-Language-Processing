import tensorflow as tf
from tensorflow.keras.layers import Input
import traceback

import numpy as np
from test_utils import comparator, summary

def test_line_to_tensor(target):
    line = '10101110'
    vocab = sorted(set(line))
    vocab.insert(0, "[UNK]") # Add a special character for any unknown
    vocab.insert(1,"") # Add the empty character for padding.
    
    ids = target(line, vocab)
    assert tf.is_tensor(ids), f"Wrong type, your function must return a Tensor"
    assert (ids.dtype == tf.int64), f"Wrong number type. Expected: {tf.int64} but got {ids.dtype}"
    assert len(ids) == len(line), f"Wrong length. Expected: {len(line)} but got {len(ids)}"
    assert tf.math.reduce_all(tf.equal(ids, [3, 2, 3, 2, 3, 3, 3, 2])), f"Unit test 1 failed. "
    
    line = "123"
    ids = target(line, vocab)
    assert tf.is_tensor(ids), f"Wrong type, your function must return a Tensor"
    assert len(ids) == len(line), f"Wrong length. Expected: {len(line)} but got {len(ids)}"
    assert tf.math.reduce_all(tf.equal(ids, [3, 0, 0])), f"Unit test 2 failed. "
    
    
    line = "123abc"
    vocab = sorted(set(line))
    vocab.insert(0, "[UNK]") # Add a special character for any unknown
    vocab.insert(1,"") # Add the empty character for padding.
    
    ids = target(line, vocab)
    assert tf.is_tensor(ids), f"Wrong type, your function must return a Tensor"
    assert len(ids) == len(line), f"Wrong length. Expected: {len(line)} but got {len(ids)}"
    assert tf.math.reduce_all(tf.equal(ids, [2, 3, 4, 5, 6, 7])), f"Unit test 1 failed. "
    
    line = "1234567"
    ids = target(line, vocab)
    assert tf.is_tensor(ids), f"Wrong type, your function must return a Tensor"
    assert len(ids) == len(line), f"Wrong length. Expected: {len(line)} but got {len(ids)}"
    assert tf.math.reduce_all(tf.equal(ids, [2, 3, 4, 0, 0, 0, 0])), f"Unit test 2 failed. "
    
    print("\033[92mAll test passed!")

def test_create_batch_dataset(target):
    BATCH_SIZE = 2
    SEQ_LENGTH = 8 
    lines = ['abc 123 xyz', 'Hello world!', '1011101']
    vocab = sorted(set('abcdefghijklmnopqrstuvwxyz12345'))
    
    tf.random.set_seed(272)
    dataset = target(lines, vocab, seq_length=SEQ_LENGTH, batch_size=BATCH_SIZE)
    exp_shape = (BATCH_SIZE, SEQ_LENGTH)
    outputs = dataset.take(1)
    assert len(outputs) > 0, f"Wrong length. First batch must have 1 element. Got {len(outputs)}"
    for in_line, out_line in dataset.take(1):
        assert tf.is_tensor(in_line), "Wrong type. in_line extected to be a Tensor"
        assert tf.is_tensor(out_line), "Wrong type. out_line extected to be a Tensor"
        assert in_line.shape == exp_shape, f"Wrong shape in in_line. Expected {in_line.shape} but got: {exp_shape}"
        assert out_line.shape == exp_shape, f"Wrong shape. Expected {in_line.shape} but got: {exp_shape}"

        expected_in_line = [[28, 20, 23, 17,  9,  0,  0,  1],
                            [30, 31,  0,  0, 10, 17, 17, 20]]
        expected_out_line = [[20, 23, 17,  9,  0,  0,  1,  0],
                             [31,  0,  0, 10, 17, 17, 20,  0]]
        
        assert tf.math.reduce_all(tf.equal(in_line, expected_in_line)), \
            f"Wrong values. Expected {expected_in_line} but got: {in_line.numpy()}"
        assert tf.math.reduce_all(tf.equal(out_line, expected_out_line)), \
            f"Wrong values. Expected {expected_out_line} but got: {out_line.numpy()}"
        
    BATCH_SIZE = 4
    SEQ_LENGTH = 8 
    lines = [ 'Hello world!', '1918', '1010101', 'deeplearning.ai']
    vocab = sorted(set('abcdefghijklmnopqrstuvwxyz012345'))
    vocab.insert(0, "[UNK]") # Add a special character for any unknown
    vocab.insert(1,"") # Add the empty character for padding.
        
    tf.random.set_seed(5)
    dataset = target(lines, vocab, seq_length=SEQ_LENGTH, batch_size=BATCH_SIZE)
    exp_shape = (BATCH_SIZE, SEQ_LENGTH)
    outputs = dataset.take(1)
    assert len(outputs) > 0, f"Wrong length. First batch must have 1 element. Got {len(outputs)}"
    for in_line, out_line in dataset.take(1):
        assert tf.is_tensor(in_line), "Wrong type. in_line extected to be a Tensor"
        assert tf.is_tensor(out_line), "Wrong type. out_line extected to be a Tensor"
        assert in_line.shape == exp_shape, f"Wrong shape in in_line. Expected {in_line.shape} but got: {exp_shape}"
        assert out_line.shape == exp_shape, f"Wrong shape. Expected {in_line.shape} but got: {exp_shape}"

        expected_in_line = [[19, 11,  0,  0,  3,  0,  3,  0],
                            [ 3,  2,  3,  2,  3,  2,  3,  0],
                            [ 0, 12, 19, 19, 22,  0, 30, 22],
                            [12, 12, 23, 19, 12,  8, 25, 21]]
        
        expected_out_line = [[11,  0,  0,  3,  0,  3,  0,  0],
                             [ 2,  3,  2,  3,  2,  3,  0, 11],
                             [12, 19, 19, 22,  0, 30, 22, 25],
                             [12, 23, 19, 12,  8, 25, 21, 16]]

        assert tf.math.reduce_all(tf.equal(in_line, expected_in_line)), \
            f"Wrong values. Expected {expected_in_line} but got: {in_line.numpy()}"
        assert tf.math.reduce_all(tf.equal(out_line, expected_out_line)), \
            f"Wrong values. Expected {expected_out_line} but got: {out_line.numpy()}"

    print("\n\033[92mAll test passed!")

def test_GRULM(target):
    batch_size = 64
    max_length = 128
    embedding_dim = 16
    vocab_size = 4
    rnn_units = 32
    modelw = target(
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                rnn_units = rnn_units)
    print("Test case 1:")
    try:
        modelw.build(input_shape=(batch_size, max_length))
        modelw.call(Input(shape=(max_length)))
        comparator(summary(modelw),
                   [['Embedding', (None, max_length, embedding_dim), 64], 
                    ['GRU', [(None, max_length, rnn_units), (None, rnn_units)], 4800, 'return_sequences=True', 'return_state=True'], 
                    ['Dense', (None, max_length, vocab_size), 132, 'log_softmax']])
    except:
        print("\033[91m\nYour model is not building")
        
    batch_size = 32
    max_length = 50
    embedding_dim = 400
    vocab_size = 52
    rnn_units = 101
    modelw = target(
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                rnn_units = rnn_units)
    print("Test case 2:")
    try:
        modelw.build(input_shape=(batch_size, max_length))
        modelw.call(Input(shape=(max_length)))
        comparator(summary(modelw),
                   [['Embedding', (None, max_length, embedding_dim), 20800], 
                    ['GRU', [(None, max_length, rnn_units), (None, rnn_units)], 152409, 'return_sequences=True', 'return_state=True'], 
                    ['Dense', (None, max_length, vocab_size), 5304, 'log_softmax']])

    except:
        print("\033[91m\nYour model is not building")
        traceback.print_exc()


def test_compile_model(target):
    
    model = target(tf.keras.Sequential())
    # Define the loss function. Use SparseCategoricalCrossentropy 
    loss = model.loss
    #loss = model.loss
    assert type(loss) == tf.losses.SparseCategoricalCrossentropy, f"Wrong loss function.  Expected {tf.losses.SparseCategoricalCrossentropy} but got {type(loss)}]"
    y_true = [1, 2]
    y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
    loss_y = loss(y_true, y_pred)
    assert np.isclose(loss_y, 0.9868951), f"Wrong value for loss. Expected {0.9868951} but got {loss_y}. Check from_logits parameter."
    optimizer = model.optimizer
    assert type(optimizer) == tf.keras.optimizers.Adam, "Wrong optimizer"
    assert np.isclose(optimizer.learning_rate.numpy(), 0.00125), f"Wrong learning_rate. Expected {0.00125} but got {optimizer.learning_rate.numpy()}."
    
    print("\n\033[92mAll test passed!")


def test_test_model(target):
    test_cases = [
        {
            "name": "example 1",
            "input": {
                "preds": tf.constant([[[0.1, 0.3, 0.7],
                                       [0.1, 0.3, 0.7],
                                       [0.1, 0.3, 0.7],
                                       [0.1, 0.3, 0.7],
                                       [0.1, 0.3, 0.7]]]),
                "target": tf.constant([[2, 2, 2, 2, 2]]),
            },
            "expected": -0.699999988079071,
        },
        {
            "name": "example 2",
            "input": {
                "preds": tf.constant([[[0.0, 0.0, 1.],
                                       [0.0, 0.0, 1.],
                                       [0.0, 0.0, 1.],
                                       [0.0, 0.0, 1.],
                                       [0.0, 0.0, 1.]]]),
                "target": tf.constant([[1, 1, 1, 1, 1]]),
            },
            "expected": float("nan"),
        },
                {
            "name": "example 3",
            "input": {
                "preds": tf.constant([[[0.0, 1.0, 0.],
                                       [0.0, 1.0, 0.],
                                       [0.0, 1.0, 0.],
                                       [0.0, 1.0, 0.],
                                       [0.0, 1.0, 0.]]]),
                "target": tf.constant([[1, 1, 1, 1, 1]]),
            },
            "expected": float("nan"),
        },
        {
            "name": "example 4",
            "input": {
                "preds": tf.constant([[[0.0, 0.0, 1.],
                                       [0.0, 0.0, 1.],
                                       [0.0, 0.0, 1.],
                                       [0.0, 0.0, 1.],
                                       [0.0, 0.0, 1.]]]),
                "target": tf.constant([[2, 2, 2, 2, 2]]),
            },
            "expected": -1.,
        },
        {
            "name": "example 5",
            "input": {
                "preds": tf.constant([[[1., 0., 0., 0., 0.],
                                       [0., 1., 0., 0., 0.],
                                       [0., 0., 1., 0., 0.],
                                       [0., 0., 0., 1., 0.],
                                       [0., 0., 1., 0., 1.],
                                       [0., 0., 0., 1., 0.],
                                       [0., 0., 1., 0., 0.],
                                       [0., 1., 0., 0., 0.]]]),
                "target": tf.constant([[0, 1, 2, 3, 4, 3, 2, 1]]),
            },
            "expected": -1.,
        },
        {
            "name": "example 6",
            "input": {
                "preds": tf.constant([[[1., 0., 0., 0., 0.],
                                       [0., 1., 0., 0., 0.],
                                       [0., 0., 1., 0., 0.],
                                       [0., 0., 0., 1., 0.],
                                       [0., 0., 1., 0., 1.],
                                       [0., 0., 0., 1., 0.],
                                       [0., 0., 1., 0., 0.],
                                       [0., 1., 0., 0., 0.]]]),
                "target": tf.constant([[0, 1, 2, 3, 4, 0, 1, 2]]),
            },
            "expected": -4./6.,
        },
        {
            "name": "example 7",
            "input": {
                "preds": tf.constant([[[1., 0., 0., 0., 0.],
                                       [0., 1., 0., 0., 0.],
                                       [0., 0., 1., 0., 0.],
                                       [0., 0., 0., 1., 0.],
                                       [0., 0., 1., 0., 1.],
                                       [0., 0., 0., 1., 0.],
                                       [0., 0., 1., 0., 0.],
                                       [0., 1., 0., 0., 0.]]]),
                "target": tf.constant([[1, 2, 3, 4, 0, 0, 0, 0]]),
            },
            "expected": 0,
        },
        {
            "name": "example 8, Batch of 1",
            "input": {
                "preds": tf.constant([[[0.1, 0.5, 0.4],
                                       [0.05, 0.9, 0.05],
                                       [0.2, 0.3, 0.5],
                                       [0.1, 0.2, 0.7],
                                       [0.2, 0.8, 0.1],
                                       [0.4, 0.4, 0.2],
                                       [0.5, 0.0, 0.5]]]),
                "target": tf.constant([[1, 2, 0, 2, 0, 2, 0]]),
            },
            "expected": -0.3083333329608043,
        },
        {
            "name": "Example 9. Batch of 2",
            "input": {
                "preds": tf.constant([[[[0.1, 0.5, 0.4],
                                       [0.05, 0.9, 0.05],
                                       [0.2, 0.3, 0.5],
                                       [0.1, 0.2, 0.7],
                                       [0.2, 0.8, 0.1],
                                       [0.4, 0.4, 0.2],
                                       [0.5, 0.0, 0.5]]],
                                     [[[0.1, 0.5, 0.4],
                                       [0.2, 0.8, 0.1],
                                       [0.4, 0.4, 0.2],
                                       [0.5, 0.0, 0.5],
                                       [0.05, 0.9, 0.05],
                                       [0.2, 0.3, 0.5],
                                       [0.1, 0.2, 0.7]]]]),
                "target": tf.constant([[1, 2, 0, 2, 0, 2, 0], [2, 1, 1, 2, 2, 0, 0]]),
            },
            "expected": -0.31333333427707355,
        }
    ]
    
    for testi in test_cases:
        test_in = testi["input"]
        expected = testi["expected"]
        output = target(test_in["preds"], test_in["target"])
        if np.isnan(expected):
            assert np.isnan(output), f"Fail in {testi['name']}. Expected {expected} but got {output}"
        else:
            assert np.allclose(output, expected), f"Fail in {testi['name']}. Expected {expected} but got {output}"

    print("\n\033[92mAll test passed!")


def test_GenerativeModel(target, model, vocab):
    tf.random.set_seed(272)
    gen = target(model, vocab, temperature=0.5)
    n_chars = 40
    pre = "SEFOE"
    text1 = gen.generate_n_chars(n_chars, pre)
    assert len(text1) == n_chars + len(pre) , f"Wrong length. Expected {n_chars + len(pre)} but got{len(text1)}"
    text2 = gen.generate_n_chars(n_chars, pre)
    assert len(text2) == n_chars + len(pre), f"Wrong length. Expected {n_chars + len(pre)} but got{len(text2)}"
    assert text1 != text2, f"Expected different strings since temperature is > 0.0"

    gen = target(model, vocab, temperature=0.0)
    n_chars = 40
    pre = "What is "
    text1 = gen.generate_n_chars(n_chars, pre)
    assert len(text1) == n_chars + len(pre) , f"Wrong length. Expected {n_chars + len(pre)} but got{len(text1)}"
    text2 = gen.generate_n_chars(n_chars, pre)
    assert len(text2) == n_chars + len(pre), f"Wrong length. Expected {n_chars + len(pre)} but got{len(text2)}"
    assert text1 == text2, f"Expected same strings since temperature is 0.0"
    
    n_chars = 100
    pre = "W"
    text_l = gen.generate_n_chars(n_chars, pre)
    used_voc = set(text_l)
    assert used_voc.issubset(set(vocab)), "Something went wrong. Only characters in vocab can be produced." \
    f" Unexpected characters: {used_voc.difference(set(vocab))}"
    
    print("\n\033[92mAll test passed!")
