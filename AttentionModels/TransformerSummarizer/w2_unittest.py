import tensorflow as tf
import numpy as np
from dlai_grader.grading import test_case, print_feedback

SEED = 10


# -

def print_feedback(test_cases):
    num_cases = len(test_cases)
    failed_cases = [t for t in test_cases if t.failed == True]
    feedback_msg = "\033[92m All tests passed!"
    if failed_cases:
        feedback_msg = ""
        for failed_case in failed_cases:
            feedback_msg += f"\033[91mFailed test case: {failed_case.msg}.\nExpected: {failed_case.want}\nGot: {failed_case.got}\n\n"
    print(feedback_msg)


def test_scaled_dot_product_attention(target):
    
    def g():
        q = np.array([[1, 1, 0, 1], [0, 1, 1, 1], [1, 0, 1, 1]]).astype(np.float32)
        k = np.array([[1, 1, 0, 1], [1, 0, 1, 1 ], [1, 1, 1, 0], [0, 0, 0, 1]]).astype(np.float32)
        v = np.array([[0, 0], [1, 0], [1, 0], [1, 1]]).astype(np.float32)
        mask = np.array([[[0, 1, 0, 1], [1, 0, 0, 1], [1, 1, 0, 1]]])

        unmasked_weights_solution = [[0.3874556,  0.23500372, 0.23500372, 0.14253697],
                                     [0.2772748,  0.2772748,  0.2772748,  0.16817565],
                                     [0.23500372, 0.3874556,  0.23500372, 0.14253697]]
        
        unmasked_output_solution = [[0.6125444,  0.14253697],
                                    [0.7227252,  0.16817565],
                                    [0.7649963,  0.14253697]]
        
        
        masked_weights_solution = [[0.0,        0.62245935, 0.0,        0.37754068],
                                   [0.62245935, 0.0,        0.0,        0.37754068],
                                   [0.30719587, 0.5064804,  0.0,        0.18632373]]
        
        
        masked_output_solution = [[1.0,        0.37754068],
                                  [0.37754068, 0.37754068],
                                  [0.6928041,  0.18632373]]
        
        cases = []
        
        # Test for unmasked
        attention, weights = target(q, k, v, None)

        t = test_case()
        if not tf.is_tensor(weights):
            t.failed = True
            t.msg = "Attention weights must be a tensor"
            t.want = "A tensor"
            t.got = type(weights)
            return [t]
        cases.append(t)

        t = test_case()
        if tuple(tf.shape(weights).numpy()) != (q.shape[0], k.shape[1]):
            t.failed = True
            t.msg = f"Wrong shape of attention weights. Expected shape: ({q.shape[0]}, {k.shape[1]})"
            t.want = (q.shape[0], k.shape[1])
            t.got = tuple(tf.shape(weights).numpy())
        cases.append(t)        

        t = test_case()
        if not np.allclose(weights, unmasked_weights_solution):
            t.failed = True
            t.msg = "Wrong unmasked attention weights"
            t.want = unmasked_weights_solution
            t.got = weights
        cases.append(t) 

        t = test_case()
        if not tf.is_tensor(attention):
            t.failed = True
            t.msg = "Output must be a tensor"
            t.want = "A tensor"
            t.got = type(attention)
            cases.append(t)
            return [t]
        cases.append(t)
        
        t = test_case()
        if tuple(tf.shape(attention).numpy()) != (q.shape[0], v.shape[1]):
            t.failed = True
            t.msg = f"Wrong shape of output. Expected shape: ({q.shape[0]}, {v.shape[1]})"
            t.want = (q.shape[0], v.shape[1])
            t.got = tuple(tf.shape(attention).numpy())
        cases.append(t)    

        t = test_case()
        if not np.allclose(attention, unmasked_output_solution):
            t.failed = True
            t.msg = "Wrong unmasked output."
            t.want = unmasked_output_solution
            t.got = attention
        cases.append(t) 
        
        # Test for masked
        attention, weights = target(q, k, v, mask)

        t = test_case()
        if not np.allclose(weights, masked_weights_solution):
            t.failed = True
            t.msg = "Wrong masked attention weights"
            t.want = masked_weights_solution
            t.got = weights
        cases.append(t)         
        
        t = test_case()
        if not np.allclose(attention, masked_output_solution):
            t.failed = True
            t.msg = "Wrong masked output"
            t.want = masked_output_solution
            t.got = attention
        cases.append(t)
        
        return cases
    
    cases = g()
    print_feedback(cases)


def test_encoderlayer(target):
    
    def g():
        tf.keras.utils.set_random_seed(SEED)
        q = np.array([[[1, 0, 1, 1], [0, 1, 0, 1], [1, 1, 0, 1]]]).astype(np.float32)
        encoder_layer1 = target(4, 2, 8)
        
        encoded_training_true = [[-0.6943532,  -1.1015786,   0.30987337,  1.4860584 ],
                                 [ 0.11547165,  0.8349443,  -1.6664022,   0.71598625],
                                 [ 0.37704456,  0.9044332,  -1.6940061,   0.4125284 ]]
        
        
        encoded_training_false = [[-0.6943532,  -1.1015786,   0.30987337,  1.4860584, ],
                                  [ 0.11547165,  0.8349443,  -1.6664022,   0.71598625,],
                                  [ 0.40035784,  0.8880369,  -1.6980288,   0.40963417,]]
        
        encoded = encoder_layer1(q, True, np.array([[1, 0, 1]]))
        
        cases = []
        
        t = test_case()
        if not tf.is_tensor(encoded):
            t.failed = True
            t.msg = "Wrong type. Output must be a tensor"
            t.want = "A tensor"
            t.got = type(encoded)
            return [t]
        cases.append(t)

        t = test_case()
        if tuple(tf.shape(encoded).numpy()) != (1, q.shape[1], q.shape[2]):
            t.failed = True
            t.msg = f"Wrong shape of output. Expected shape: (1, {q.shape[1]}, {q.shape[2]})"
            t.want = (1, q.shape[1], q.shape[2])
            t.got = tuple(tf.shape(encoded).numpy())
        cases.append(t)    
        
        t = test_case()
        if not np.allclose(encoded.numpy(), encoded_training_true):
            t.failed = True
            t.msg = "Wrong values when training=True"
            t.want = encoded_training_true
            t.got = encoded.numpy()
        cases.append(t)            

        encoded = encoder_layer1(q, True, np.array([[1, 0, 1]]))

        t = test_case()
        if not np.allclose(encoded.numpy(), encoded_training_false):
            t.failed = True
            t.msg = "Wrong values when training=False"
            t.want = encoded_training_false
            t.got = encoded.numpy()
        cases.append(t)            
        return cases

    cases = g()
    print_feedback(cases)


def test_encoder(target):
    def g():
        tf.keras.utils.set_random_seed(SEED)

        embedding_dim=4

        encoderq = target(num_layers=2,
                          embedding_dim=embedding_dim,
                          num_heads=2,
                          fully_connected_dim=8,
                          input_vocab_size=32,
                          maximum_position_encoding=5)

        x = np.array([[2, 1, 1], [0, 2, 1]])

        encoderq_output = encoderq(x, True, None)

        case_1_result = [[[-1.5521252e+00,  1.2275430e+00,  2.8751713e-01,  3.7065268e-02],
                          [ 9.4034457e-01, -6.4952612e-01, -1.2968377e+00,  1.0060191e+00],
                          [ 2.8042072e-01, -1.3738929e+00, -3.0059353e-01,  1.3940656e+00]],

                         [[-1.7030637e+00,  8.6491209e-01,  4.0807986e-01,  4.3007189e-01],
                          [-6.9350564e-01, -1.0131397e+00,  1.3350804e-01,  1.5731372e+00],
                          [-3.7993553e-01, -1.1846002e+00, -1.4650300e-03,  1.5660008e+00]]]
        
        case_2_result = [[[-1.5422106,   1.1804703,   0.4660407,  -0.10430057],
                          [-1.6235152,   1.051704,    0.5034224,   0.06838872],
                          [-0.12116455, -1.3260406,  -0.04442041,  1.4916254 ]],

                         [[-1.4519225,   1.3070422,   0.36660305, -0.22172265],
                          [ 1.4702849,  -0.7520739,  -1.0713955,   0.35318464],
                          [-1.4717183,  -0.0867067,   0.22596014,  1.3324649 ]]]
        
        case_3_result = [[[-1.6598201,   1.0180895,   0.28498387,  0.35674685],
                          [-0.10208954, -1.0945475,  -0.42039752,  1.6170347 ],
                          [-0.13384242, -1.2241358,  -0.20461643,  1.5625945 ]],

                         [[-1.6484364,   1.0502913,   0.3171085,   0.28103673],
                          [-0.74074364, -0.7831033,  -0.15376769,  1.6776146 ],
                          [-0.2733949,  -1.1927879,  -0.11179274,  1.5779753 ]]]
        
        cases = []
        
        t = test_case()
        if not tf.is_tensor(encoderq_output):
            t.failed = True
            t.msg = "Wrong type. Output must be a tensor"
            t.want = "A tensor"
            t.got = type(encoderq_output)
            return [t]
        cases.append(t)
        
        t = test_case()
        if tuple(tf.shape(encoderq_output).numpy()) != (x.shape[0], x.shape[1], embedding_dim):
            t.failed = True
            t.msg = f"Wrong shape of output. Expected shape: ({x.shape[0]}, {x.shape[1]}, {embedding_dim})"
            t.want = ({x.shape[0]}, {x.shape[1]}, {embedding_dim})
            t.got = tuple(tf.shape(encoderq_output).numpy())
        cases.append(t)    
        
        t = test_case()
        if not np.allclose(encoderq_output.numpy(), case_1_result):
            t.failed = True
            t.msg = "Wrong values for test case 1"
            t.want = case_1_result
            t.got = encoderq_output.numpy()
        cases.append(t)            

        encoderq_output = encoderq(x, True, np.array([[[[1., 1., 1.]]], [[[1., 1., 0.]]]]))

        t = test_case()
        if not np.allclose(encoderq_output.numpy(), case_2_result):
            t.failed = True
            t.msg = "Wrong values for test case 2"
            t.want = case_2_result
            t.got = encoderq_output.numpy()
        cases.append(t) 
        
        encoderq_output = encoderq(x, False, np.array([[[[1., 1., 1.]]], [[[1., 1., 0.]]]]))

        t = test_case()
        if not np.allclose(encoderq_output.numpy(), case_3_result):
            t.failed = True
            t.msg = "Wrong values for test case 3"
            t.want = case_3_result
            t.got = encoderq_output.numpy()
        cases.append(t) 
        
        return cases

    cases = g()
    print_feedback(cases)


def test_decoderlayer(target, create_look_ahead_mask):
    def g():
        num_heads=8
        tf.keras.utils.set_random_seed(SEED)

        decoderLayerq = target(
            embedding_dim=4, 
            num_heads=num_heads,
            fully_connected_dim=32, 
            dropout_rate=0.1, 
            layernorm_eps=1e-6)

        encoderq_output = tf.constant([[[-0.4,  0.4, -1.2,   1.5],
                                       [ 0.4,   0.2, -1.6,   0.9],
                                       [ 0.4,  -1.6,  0.1,   1.2]]])

        q = np.array([[[1, 0, 1, 1], [1, 0, 1, 1], [1, 0, 1, 1]]]).astype(np.float32)

        look_ahead_mask = create_look_ahead_mask(q.shape[1])

        padding_mask = None
        out, attn_w_b1, attn_w_b2 = decoderLayerq(q, encoderq_output, True, look_ahead_mask, padding_mask)
        
        shape1 = (q.shape[0], num_heads, q.shape[1], q.shape[1])
        
        cases = []
        
        t = test_case()
        if not tf.is_tensor(attn_w_b1):
            t.failed = True
            t.msg = "Wrong type for attn_w_b1. Output must be a tensor"
            t.want = "A tensor"
            t.got = type(attn_w_b1)
            return [t]
        cases.append(t)

        t = test_case()
        if not tf.is_tensor(attn_w_b2):
            t.failed = True
            t.msg = "Wrong type for attn_w_b2. Output must be a tensor"
            t.want = "A tensor"
            t.got = type(attn_w_b1)
            cases.append(t)
            return cases
        cases.append(t)
        
        t = test_case()
        if not tf.is_tensor(out):
            t.failed = True
            t.msg = "Wrong type for out. Output must be a tensor"
            t.want = "A tensor"
            t.got = type(attn_w_b1)
            cases.append(t)
            return cases
        cases.append(t)
    
        t = test_case()
        if tuple(tf.shape(attn_w_b1).numpy()) != shape1:
            t.failed = True
            t.msg = f"Wrong shape of attn_w_b1. Expected shape: {shape1}"
            t.want = shape1
            t.got = tuple(tf.shape(attn_w_b1).numpy())
        cases.append(t)  

        t = test_case()
        if tuple(tf.shape(attn_w_b2).numpy()) != shape1:
            t.failed = True
            t.msg = f"Wrong shape of attn_w_b2. Expected shape: {shape1}"
            t.want = shape1
            t.got = tuple(tf.shape(attn_w_b2).numpy())
        cases.append(t)  
        
        t = test_case()
        if tuple(tf.shape(out).numpy()) != q.shape:
            t.failed = True
            t.msg = f"Wrong shape of out. Expected shape: {q.shape}"
            t.want = q.shape
            t.got = tuple(tf.shape(out).numpy())
        cases.append(t)  

        t = test_case()
        if not np.allclose(attn_w_b1[0, 0, 1], [0.5, 0.5, 0.], atol=1e-2):
            t.failed = True
            t.msg = "Wrong values in 'attn_w_b1'. Check the call to self.mha1"
            t.want = [0.5, 0.5, 0.]
            t.got = attn_w_b1[0, 0, 1]
        cases.append(t)   
    
        t = test_case()
        if not np.allclose(attn_w_b2[0, 0, 1], [0.34003818, 0.32569194, 0.33426988]):
            t.failed = True
            t.msg = "Wrong values in 'attn_w_b2'. Check the call to self.mha2"
            t.want = [0.34003818, 0.32569194, 0.33426988]
            t.got = attn_w_b2[0, 0, 1]
        cases.append(t)      
    
        t = test_case()
        if not np.allclose(out[0, 0], [1.1810006, -1.5600019, 0.41289005, -0.03388882]):
            t.failed = True
            t.msg = "Wrong values in 'out'"
            t.want = [1.1810006, -1.5600019, 0.41289005, -0.03388882]
            t.got = out[0, 0]
        cases.append(t)  
        
        # Now let's try a example with padding mask
        padding_mask = np.array([[[1, 1, 0]]])
        out, attn_w_b1, attn_w_b2 = decoderLayerq(q, encoderq_output, True, look_ahead_mask, padding_mask)

        t = test_case()
        if not np.allclose(out[0, 0], [1.1297308, -1.6106694, 0.32352272, 0.15741566]):
            t.failed = True
            t.msg = "Wrong values in 'out' when we mask the last word. Are you passing the padding_mask to the inner functions?"
            t.want = [1.1297308, -1.6106694, 0.32352272, 0.15741566]
            t.got = out[0, 0]
        cases.append(t)  
        
        return cases

    cases = g()
    print_feedback(cases)


def test_decoder(target, create_look_ahead_mask, create_padding_mask):
    def g():
        tf.keras.utils.set_random_seed(SEED)
        
        num_layers=7
        embedding_dim=4 
        num_heads=3
        fully_connected_dim=8
        target_vocab_size=33
        maximum_position_encoding=6

        x = np.array([[3, 2, 1], [2, 1, 0]])


        encoderq_output = tf.constant([[[-0.2,  0.1, -1.3,  1.0],
                                        [ 0.4,  0.6, -1.1,  0.7],
                                        [ 0.1, -1.6,  0.3,  1.1]],
                                       [[-0.7,  0.2, -1.1,  1.0],
                                        [-0.2, -0.2, -1.0,  1.2],
                                        [ 0.8, -1.1,  0.4,  1.3]]])

        look_ahead_mask = create_look_ahead_mask(x.shape[1])

        decoderk = target(num_layers,
                        embedding_dim, 
                        num_heads, 
                        fully_connected_dim,
                        target_vocab_size,
                        maximum_position_encoding)
        outd, att_weights = decoderk(x, encoderq_output, False, look_ahead_mask, None)
        
        cases = []
        
        t = test_case()
        if not isinstance(att_weights, dict):
            t.failed = True
            t.msg = "Wrong type for attention_weights. Output must be a dictionary"
            t.want = type({1:2})
            t.got = type(att_weights)
            return [t]
        cases.append(t)

        t = test_case()
        if not tf.is_tensor(outd):
            t.failed = True
            t.msg = "Wrong type for x. Output must be a tensor"
            t.want = "A tensor"
            t.got = type(outd)
            cases.append(t)
            return cases
        cases.append(t)
        
        t = test_case()
        if not np.allclose(tf.shape(outd), tf.shape(encoderq_output)):
            t.failed = True
            t.msg = f"Wrong shape of x. Expected shape: {tf.shape(encoderq_output)}"
            t.want = tf.shape(encoderq_output)
            t.got = tf.shape(outd)
        cases.append(t)  

        t = test_case()
        if not np.allclose(outd[1, 1], [1.6461557, -0.7657816, -0.04255769, -0.8378165]):
            t.failed = True
            t.msg = "Wrong values in x"
            t.want = [1.6461557, -0.7657816, -0.04255769, -0.8378165]
            t.got = outd[1, 1]
        cases.append(t)   

        keys = list(att_weights.keys())

        t = test_case()
        if len(keys) != 2 * num_layers:
            t.failed = True
            t.msg = f"Wrong length for attention weights. It must be 2 x num_layers = {2*num_layers}"
            t.want = 2 * num_layers
            t.got = len(keys)
        cases.append(t)    
        
        t = test_case()
        if not tf.is_tensor(att_weights[keys[0]]):
            t.failed = True
            t.msg = f"Wrong type for att_weights[{keys[0]}]. Output must be a tensor"
            t.want = "A tensor"
            t.got = type(att_weights[keys[0]])
            cases.append(t)
            return cases
        cases.append(t)   
    
        shape1 = (x.shape[0], num_heads, x.shape[1], x.shape[1])
        
        t = test_case()
        if tuple(tf.shape(att_weights[keys[1]]).numpy()) != shape1:
            t.failed = True
            t.msg = f"Wrong shape of attention_weights[{keys[1]}]. Expected shape: {shape1}"
            t.want = shape1
            t.got = tf.shape(att_weights[keys[1]]).numpy()
        cases.append(t)  
        
        t = test_case()
        if not np.allclose(att_weights[keys[0]][0, 0, 1], [0.51728565, 0.48271435, 0.]):
            t.failed = True
            t.msg = f"Wrong values in att_weights[{keys[0]}]"
            t.want = [0.51728565, 0.48271435, 0.]
            t.got = att_weights[keys[0]][0, 0, 1]
        cases.append(t)
        
        outd, att_weights = decoderk(x, encoderq_output, True, look_ahead_mask, None)
        
        t = test_case()
        if not np.allclose(outd[1, 1], [1.6286429, -0.7686589, 0.00983591, -0.86982]):
            t.failed = True
            t.msg = "Wrong values in outd when training=True"
            t.want = [1.6286429, -0.7686589, 0.00983591, -0.86982]
            t.got = outd[1, 1]
        cases.append(t)
        
        outd, att_weights = decoderk(x, encoderq_output, True, look_ahead_mask, create_padding_mask(x))
        
        t = test_case()
        if not np.allclose(outd[1, 1], [1.390952, 0.2794097, -0.2910638, -1.3792979]):
            t.failed = True
            t.msg = "Wrong values in outd when training=True and use padding mask"
            t.want = [1.390952, 0.2794097, -0.2910638, -1.3792979]
            t.got = outd[1, 1]
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def test_transformer(target, create_look_ahead_mask, create_padding_mask):
    def g():
        tf.keras.utils.set_random_seed(SEED)

        num_layers = 6
        embedding_dim = 4
        num_heads = 4
        fully_connected_dim = 8
        input_vocab_size = 30
        target_vocab_size = 35
        max_positional_encoding_input = 5
        max_positional_encoding_target = 6

        transformer = target(num_layers, 
            embedding_dim, 
            num_heads, 
            fully_connected_dim, 
            input_vocab_size, 
            target_vocab_size, 
            max_positional_encoding_input,
            max_positional_encoding_target)
        
        # 0 is the padding value
        sentence_a = np.array([[2, 3, 4, 3, 0]])
        sentence__b = np.array([[1, 2, 4, 0, 0]])

        enc_padding_mask = create_padding_mask(sentence_a)
        dec_padding_mask = create_padding_mask(sentence__b)

        look_ahead_mask = create_look_ahead_mask(sentence_a.shape[1])

        summary, weights = transformer(
            sentence_a,
            sentence__b,
            True,  # Training
            enc_padding_mask,
            look_ahead_mask,
            dec_padding_mask
        )

        cases = []        
        
        t = test_case()
        if not tf.is_tensor(summary):
            t.failed = True
            t.msg = "Wrong type for summary. Output must be a tensor"
            t.want = "Tensor"
            t.got = type(summary)
            return [t]
        cases.append(t)
        
        shape1 = (sentence_a.shape[0], max_positional_encoding_input, target_vocab_size)
        
        t = test_case()
        if tuple(tf.shape(summary).numpy()) != shape1:
            t.failed = True
            t.msg = f"Wrong shape of summary. Expected shape: {shape1}"
            t.want = shape1
            t.got = tf.shape(summary).numpy()
        cases.append(t)

        summary_example_1 = [0.04855702, 0.03407773, 0.01294427, 0.05483282, 0.03182802, 0.01409046, 0.02963346, 0.04003222]
        
        t = test_case()
        if not np.allclose(summary[0, 0, 0:8], summary_example_1):
            t.failed = True
            t.msg = "Wrong values in summary"
            t.want = summary_example_1
            t.got = summary[0, 0, 0:8]
        cases.append(t)           

        t = test_case()
        if not isinstance(weights, dict):
            t.failed = True
            t.msg = "Wrong type for attention weights. It must be a dictionary"
            t.want = type({1:2})
            t.got = type(weights)
            cases.append(t) 
            return cases
        cases.append(t)        
        
        keys = list(weights.keys())        
        t = test_case()
        if len(keys) != 2 * num_layers:
            t.failed = True
            t.msg = f"Wrong length for attention weights. It must be 2 x num_layers = {2*num_layers}"
            t.want = 2 * num_layers
            t.got = len(keys)
        cases.append(t)    
        
        t = test_case()
        if not tf.is_tensor(weights[keys[0]]):
            t.failed = True
            t.msg = f"Wrong type for attention_weights[{keys[0]}]. Output must be a tensor"
            t.want = "A tensor"
            t.got = type(weights[keys[0]])
            cases.append(t)
            return cases
        cases.append(t)   

        shape2 = (sentence_a.shape[0], num_heads, sentence_a.shape[1], sentence_a.shape[1])
        
        t = test_case()
        if tuple(tf.shape(weights[keys[0]]).numpy()) != shape2:
            t.failed = True
            t.msg = f"Wrong shape of attention_weights[{keys[0]}]. Expected shape: {shape2}"
            t.want = shape2
            t.got = tf.shape(weights[keys[0]]).numpy()
        cases.append(t)
        
        t = test_case()
        if not np.allclose(weights[keys[0]][0, 0, 1], [0.481374, 0.51862603, 0.0, 0.0, 0.0]):
            t.failed = True
            t.msg = f"Wrong values in weights[{keys[0]}]"
            t.want = [0.481374, 0.51862603, 0.0, 0.0, 0.0]
            t.got = weights[keys[0]][0, 0, 1]
        cases.append(t)           

        tf.keras.utils.set_random_seed(SEED)
        summary, weights = transformer(
            sentence_a,
            sentence__b,
            False, # Training
            enc_padding_mask,
            look_ahead_mask,
            dec_padding_mask)
        
        summary_example_2 = [0.05015587, 0.02734077, 0.01308834, 0.04876801, 0.03092919, 0.02046618, 0.02923589, 0.03272967]

        t = test_case()
        if not np.allclose(summary[0, 0, 0:8], summary_example_2):
            t.failed = True
            t.msg = "Wrong values in summary"
            t.want = summary_example_2
            t.got = summary[0, 0, 0:8]
        cases.append(t)                
        
        return cases

    cases = g()
    print_feedback(cases)


def test_next_word(target, model, encoder_input, output):
    def g():
        
        next_word = target(model, encoder_input, output)

        cases = []        
        
        t = test_case()
        if not tf.is_tensor(next_word):
            t.failed = True
            t.msg = "Wrong type for predicted_id Output must be a tensor"
            t.want = "Tensor"
            t.got = type(next_word)
            return [t]
        cases.append(t)
        
        t = test_case()
        if next_word.dtype != tf.int32:
            t.failed = True
            t.msg = f"Returned tensor should contain tf.int32 type"
            t.want = tf.int32
            t.got = next_word.dtype
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)




