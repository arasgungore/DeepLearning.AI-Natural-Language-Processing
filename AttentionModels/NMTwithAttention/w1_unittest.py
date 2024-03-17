import math
from itertools import combinations
import tensorflow as tf
import numpy as np
from dlai_grader.grading import test_case, print_feedback
from utils import train_data

VOCAB_SIZE = 12000
UNITS = 256


def test_encoder(encoder_to_test):
    def g():
        vocab_sizes = [5, 20, 1000, 15000]
        units = [32, 64, 256, 512]

        cases = []

        encoder = encoder_to_test(vocab_sizes[0], units[0])

        t = test_case()
        if encoder.embedding.mask_zero != True:
            t.failed = True
            t.msg = "Embedding layer has incorrect value for 'mask_zero' attribute"
            t.want = True
            t.got = encoder.embedding.mask_zero
        cases.append(t)

        for vs, u in zip(vocab_sizes, units):
            encoder = encoder_to_test(vs, u)

            t = test_case()
            if encoder.embedding.input_dim != vs:
                t.failed = True
                t.msg = "Incorrect input dim of embedding layer"
                t.want = vs
                t.got = encoder.embedding.input_dim
            cases.append(t)

            t = test_case()
            if encoder.embedding.output_dim != u:
                t.failed = True
                t.msg = "Incorrect output dim of embedding layer"
                t.want = u
                t.got = encoder.embedding.output_dim
            cases.append(t)

        t = test_case()
        if not isinstance(encoder.rnn.layer, tf.keras.layers.LSTM):
            t.failed = True
            t.msg = "Incorrect type of layer inside Bidirectional"
            t.want = tf.keras.layers.LSTM
            t.got = type(encoder.rnn.layer)
            return [t]

        for u in units:
            encoder = encoder_to_test(vocab_sizes[1], u)
            t = test_case()
            if encoder.rnn.layer.units != u:
                t.failed = True
                t.msg = "Incorrect number of units in LSTM layer"
                t.want = u
                t.got = encoder.rnn.layer.units
            cases.append(t)

        t = test_case()
        if encoder.rnn.layer.return_sequences != True:
            t.failed = True
            t.msg = "LSTM layer has incorrect value for 'return_sequences' attribute"
            t.want = True
            t.got = encoder.rnn.layer.return_sequences
        cases.append(t)

        vocab_size = 16
        n_units = 8
        encoder = encoder_to_test(vocab_size, n_units)
        to_translate = np.array([[1, 2, 3, 4, 5, 6, 14, 0, 0, 0],
                               [2, 1, 1, 1, 1, 1, 8, 0, 0, 0],
                               [5, 4, 2, 3, 3, 15, 11, 0, 0, 0]])
        #for (to_translate, _), _ in train_data.take(3):
            
        first_dim_in, second_dim_in = to_translate.shape
        encoder_output = encoder(to_translate)
        t = test_case()
        if len(encoder_output.shape) != 3:
            t.failed = True
            t.msg = "Incorrect shape of encoder output"
            t.want = "a shape with 3 dimensions"
            t.got = encoder_output.shape
            return [t]

        first_dim_out, second_dim_out, third_dim_out = encoder_output.shape

        t = test_case()
        if first_dim_in != first_dim_out:
            t.failed = True
            t.msg = "Incorrect first dimension of encoder output"
            t.want = first_dim_in
            t.got = first_dim_out
        cases.append(t)

        t = test_case()
        if second_dim_in != second_dim_out:
            t.failed = True
            t.msg = "Incorrect second dimension of encoder output"
            t.want = second_dim_in
            t.got = second_dim_out
        cases.append(t)

        t = test_case()
        if third_dim_out != n_units:
            t.failed = True
            t.msg = "Incorrect third dimension of encoder output"
            t.want = units
            t.got = third_dim_out
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def test_cross_attention(cross_attention_to_test):
    def g():
        units = [32, 64, 256, 512]

        cases = []

        n_units = 512
        cross_attention = cross_attention_to_test(n_units)

        t = test_case()
        if not isinstance(cross_attention.mha, tf.keras.layers.MultiHeadAttention):
            t.failed = True
            t.msg = "Incorrect type of layer for Multi Head Attention"
            t.want = tf.keras.layers.MultiHeadAttention
            t.got = type(cross_attention.mha)
            return [t]

        #         for u in units:
        #             cross_attention = cross_attention_to_test(u)

        #             t = test_case()
        #             if cross_attention.mha.key_dim != u:
        #                 t.failed = True
        #                 t.msg = "Incorrect key dim of Multi Head Attention layer"
        #                 t.want = u
        #                 t.got = cross_attention.mha.key_dim
        #             cases.append(t)

        cross_attention = cross_attention_to_test(n_units)
        embed = tf.keras.layers.Embedding(VOCAB_SIZE, output_dim=UNITS, mask_zero=True)

        for (to_translate, sr_translation), _ in train_data.take(3):
            sr_translation_embed = embed(sr_translation)
            first_dim_in, second_dim_in, third_dim_in = sr_translation_embed.shape
            dummy_encoder_output = np.random.rand(64, 14, 512)
            cross_attention_output = cross_attention(
                dummy_encoder_output, sr_translation_embed
            )
            #             print(cross_attention_output.shape)

            t = test_case()
            if len(cross_attention_output.shape) != 3:
                t.failed = True
                t.msg = "Incorrect shape of cross_attention output"
                t.want = "a shape with 3 dimensions"
                t.got = cross_attention_output.shape
                return [t]

            first_dim_out, second_dim_out, third_dim_out = cross_attention_output.shape

            t = test_case()
            if first_dim_in != first_dim_out:
                t.failed = True
                t.msg = "Incorrect first dimension of cross_attention output"
                t.want = first_dim_in
                t.got = first_dim_out
            cases.append(t)

            t = test_case()
            if second_dim_in != second_dim_out:
                t.failed = True
                t.msg = "Incorrect second dimension of cross_attention output"
                t.want = second_dim_in
                t.got = second_dim_out
            cases.append(t)

            t = test_case()
            if third_dim_in != third_dim_out:
                t.failed = True
                t.msg = "Incorrect third dimension of cross_attention output"
                t.want = third_dim_in
                t.got = third_dim_out
            cases.append(t)

        _, n_heads, key_dim = cross_attention.mha.get_weights()[0].shape

        t = test_case()
        if n_heads != 1:
            t.failed = True
            t.msg = "Incorrect number of attention heads"
            t.want = 1
            t.got = n_heads
        cases.append(t)

        t = test_case()
        if key_dim != n_units:
            t.failed = True
            t.msg = f"Incorrect size of query and key for every attention head when passing {n_units} units to the constructor"
            t.want = n_units
            t.got = key_dim
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def test_decoder(decoder_to_test, CrossAttention):
    def g():
        vocab_sizes = [5, 20, 1000, 15000]
        units = [32, 64, 256, 512]

        cases = []

        vocab_size = 10000
        n_units = 512
        decoder = decoder_to_test(vocab_size, n_units)

        t = test_case()
        if not isinstance(decoder.embedding, tf.keras.layers.Embedding):
            t.failed = True
            t.msg = "Incorrect type of embedding layer"
            t.want = tf.keras.layers.Embedding
            t.got = type(decoder.embedding)
            return [t]

        t = test_case()
        if decoder.embedding.mask_zero != True:
            t.failed = True
            t.msg = "Embedding layer has incorrect value for 'mask_zero' attribute"
            t.want = True
            t.got = decoder.embedding.mask_zero
        cases.append(t)

        for vs, u in zip(vocab_sizes, units):
            decoder = decoder_to_test(vs, u)

            t = test_case()
            if decoder.embedding.input_dim != vs:
                t.failed = True
                t.msg = "Incorrect input dim of embedding layer"
                t.want = vs
                t.got = decoder.embedding.input_dim
            cases.append(t)

            t = test_case()
            if decoder.embedding.output_dim != u:
                t.failed = True
                t.msg = "Incorrect output dim of embedding layer"
                t.want = u
                t.got = decoder.embedding.output_dim
            cases.append(t)

        t = test_case()
        if not isinstance(decoder.pre_attention_rnn, tf.keras.layers.LSTM):
            t.failed = True
            t.msg = "Incorrect type of pre_attention_rnn layer"
            t.want = tf.keras.layers.LSTM
            t.got = type(decoder.pre_attention_rnn)
            return [t]

        for u in units:
            decoder = decoder_to_test(vocab_size, u)
            t = test_case()
            if decoder.pre_attention_rnn.units != u:
                t.failed = True
                t.msg = "Incorrect number of units in pre_attention_rnn layer"
                t.want = u
                t.got = decoder.pre_attention_rnn.units
            cases.append(t)

            #             t = test_case()
            #             if decoder.attention.units != u:
            #                 t.failed = True
            #                 t.msg = "Incorrect number of units in attention layer"
            #                 t.want = u
            #                 t.got = decoder.attention.units
            #             cases.append(t)

            t = test_case()
            if decoder.post_attention_rnn.units != u:
                t.failed = True
                t.msg = "Incorrect number of units in post_attention_rnn layer"
                t.want = u
                t.got = decoder.post_attention_rnn.units
            cases.append(t)

        t = test_case()
        if decoder.pre_attention_rnn.return_sequences != True:
            t.failed = True
            t.msg = "pre_attention_rnn layer has incorrect value for 'return_sequences' attribute"
            t.want = True
            t.got = decoder.pre_attention_rnn.return_sequences
        cases.append(t)

        t = test_case()
        if decoder.pre_attention_rnn.return_state != True:
            t.failed = True
            t.msg = "pre_attention_rnn layer has incorrect value for 'return_state' attribute"
            t.want = True
            t.got = decoder.pre_attention_rnn.return_state
        cases.append(t)

        t = test_case()
        if not isinstance(decoder.attention, CrossAttention):
            t.failed = True
            t.msg = "Incorrect type of attention layer"
            t.want = CrossAttention
            t.got = type(decoder.attention)
            return [t]

        t = test_case()
        if decoder.post_attention_rnn.return_sequences != True:
            t.failed = True
            t.msg = "post_attention_rnn layer has incorrect value for 'return_sequences' attribute"
            t.want = True
            t.got = decoder.post_attention_rnn.return_sequences
        cases.append(t)

        t = test_case()
        if not isinstance(decoder.post_attention_rnn, tf.keras.layers.LSTM):
            t.failed = True
            t.msg = "Incorrect type of pre_attention_rnn layer"
            t.want = tf.keras.layers.LSTM
            t.got = type(decoder.post_attention_rnn)
            return [t]

        t = test_case()
        if not isinstance(decoder.output_layer, tf.keras.layers.Dense):
            t.failed = True
            t.msg = "Incorrect type of output_layer layer"
            t.want = tf.keras.layers.Dense
            t.got = type(decoder.output_layer)
            return [t]

        t = test_case()
        if (
            "log" not in decoder.output_layer.activation.__name__
            or "softmax" not in decoder.output_layer.activation.__name__
        ):
            t.failed = True
            t.msg = "output_layer layer has incorrect activation function"
            t.want = "a log softmax activation function such as 'log_softmax_v2'"
            t.got = decoder.output_layer.activation.__name__
        cases.append(t)

        vocab_size = 6
        n_units = 4
        decoder = decoder_to_test(vocab_size, n_units)
        sr_translation = np.array([[3, 4, 5, 3, 3, 3, 5, 1, 1, 1, 1, 1], 
                                    [1, 2, 3, 4, 5, 1, 1, 0, 0, 0, 0, 0]])
        encoder_output = np.random.rand(2, 10, n_units)
        decoder_output = decoder(encoder_output, sr_translation)

        first_dim_in, second_dim_in = sr_translation.shape

        t = test_case()
        if len(decoder_output.shape) != 3:
            t.failed = True
            t.msg = "Incorrect shape of decoder output"
            t.want = "a shape with 3 dimensions"
            t.got = decoder_output.shape
            return [t]

        first_dim_out, second_dim_out, third_dim_out = decoder_output.shape

        t = test_case()
        if first_dim_in != first_dim_out:
            t.failed = True
            t.msg = "Incorrect first dimension of decoder output"
            t.want = first_dim_in
            t.got = first_dim_out
        cases.append(t)

        t = test_case()
        if second_dim_in != second_dim_out:
            t.failed = True
            t.msg = "Incorrect second dimension of decoder output"
            t.want = second_dim_in
            t.got = second_dim_out
        cases.append(t)

        t = test_case()
        if third_dim_out != vocab_size:
            t.failed = True
            t.msg = "Incorrect third dimension of decoder output"
            t.want = vocab_size
            t.got = third_dim_out
        cases.append(t)
        
        return cases

    cases = g()
    print_feedback(cases)


def test_translator(translator_to_test, Encoder, Decoder):
    def g():
        vocab_sizes = [5, 20, 1000, 15000]
        units = [32, 64, 256, 512]

        cases = []

        vocab_size = 10000
        n_units = 512
        translator = translator_to_test(vocab_size, n_units)

        t = test_case()
        if not isinstance(translator.encoder, Encoder):
            t.failed = True
            t.msg = "Incorrect type of encoder layer"
            t.want = Encoder
            t.got = type(translator.encoder)
            return [t]

        t = test_case()
        if not isinstance(translator.decoder, Decoder):
            t.failed = True
            t.msg = "Incorrect type of encoder layer"
            t.want = Decoder
            t.got = type(translator.decoder)
            return [t]

        vocab_size = 16
        n_units = 8
        translator = translator_to_test(vocab_size, n_units)

        to_translate = np. array([[1, 2, 3, 4, 5, 0, 0],
                                 [5, 2, 3, 4, 5, 6, 0],
                                 [6, 3, 3, 4, 5, 3, 3],
                                 [7, 9, 9, 6, 5, 3, 3]])

        sr_translation = np. array([[8, 1, 2, 3, 4, 5, 0, 0],
                                 [9, 5, 2, 3, 4, 5, 6, 0],
                                 [10, 6, 3, 3, 4, 5, 3, 3],
                                 [11, 7, 9, 9, 6, 5, 3, 3]])

        #for (to_translate, sr_translation), _ in train_data.take(3):
        first_dim_in, second_dim_in = sr_translation.shape
        translator_output = translator((to_translate, sr_translation))
        t = test_case()
        if len(translator_output.shape) != 3:
            t.failed = True
            t.msg = "Incorrect shape of translator output"
            t.want = "a shape with 3 dimensions"
            t.got = translator_output.shape
            return [t]

        first_dim_out, second_dim_out, third_dim_out = translator_output.shape

        t = test_case()
        if first_dim_in != first_dim_out:
            t.failed = True
            t.msg = "Incorrect first dimension of translator output"
            t.want = first_dim_in
            t.got = first_dim_out
        cases.append(t)

        t = test_case()
        if second_dim_in != second_dim_out:
            t.failed = True
            t.msg = "Incorrect second dimension of translator output"
            t.want = second_dim_in
            t.got = second_dim_out
        cases.append(t)

        t = test_case()
        if third_dim_out != vocab_size:
            t.failed = True
            t.msg = "Incorrect third dimension of translator output"
            t.want = vocab_size
            t.got = third_dim_out
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)

    

def test_translate(learner_func, model):
    def g():
        
        cases = []
        
        txt = "Hi, my name is Younes"
        try:
            translation, logit, tokens = learner_func(model, txt, temperature=0.9)
        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = "There was an exception when running your function"
            t.want = "No exceptions"
            t.got = f"{str(e)}"
            return [t]
        
        txt = "Hi, my name is Alejandra"
        translation, logit, tokens = learner_func(model, txt, temperature=0.0)
        
        t = test_case()
        
        if not isinstance(translation, str):
            t.failed = True
            t.msg = "'translation' has incorrect type"
            t.want = str
            t.got = type(translation)
        cases.append(t)
        
        if not isinstance(logit, np.number):
            t.failed = True
            t.msg = "'logit' has incorrect type"
            t.want = np.number
            t.got = type(logit)
        cases.append(t)
        
        if not isinstance(tokens, tf.Tensor):
            t.failed = True
            t.msg = "'tokens' has incorrect type"
            t.want = tf.Tensor
            t.got = type(tokens)
        cases.append(t)
        
        translation2, logit2, tokens2 = learner_func(model, txt, temperature=0.0)
        
        t = test_case()
        if translation != translation2:
            t.failed = True
            t.msg = "translate didn't return the same translation when using temperature of 0.0"
            t.want = translation
            t.got = translation2
        cases.append(t)
        
        t = test_case()
        if logit != logit2:
            t.failed = True
            t.msg = "translate didn't return the same logit when using temperature of 0.0"
            t.want = logit
            t.got = logit2
        cases.append(t)
        
        t = test_case()
        if not np.allclose(tokens, tokens2):
            t.failed = True
            t.msg = "translate didn't return the same tokens when using temperature of 0.0"
            t.want = tokens
            t.got = tokens2
        cases.append(t)

        # Check that function uses the model.decoder and model.enconder functions
        inputs = tf.keras.Input(shape=(37,))
        outputs = tf.keras.layers.Dense(5, activation="softmax")(inputs)
        model_fake = tf.keras.Model(inputs = inputs, outputs = outputs)
        
        model_fake.encoder = model.encoder
        model_fake.decoder = None
        t = test_case()
        try:
            ff = learner_func(model_fake, "Hello world", temperature=0.0)
            t.failed = True
            t.msg = "The translator is not using the internal model.decoder. You are probably using a global variable"
            t.want = "Fail translation"
            t.got = "Succeed translation with wrong decoder"
        except:
            None
            
        cases.append(t)
        
        model_fake.encoder = None
        model_fake.decoder = model.decoder
        t = test_case()
        try:
            ff = learner_func(model_fake, "Hello world", temperature=0.0)
            t.failed = True
            t.msg = "The translator is not using the internal model.encoder. You are probably using a global variable"
            t.want = "Fail translation"
            t.got = "Succeed translation with wrong encoder"
        except:
            None

        cases.append(t)
        
        return cases
    
    cases = g()
    print_feedback(cases)

    


def test_rouge1_similarity(learner_func):
    
    def g():
        
        tensors = [
            [0],
            [0, 1],
            [0, 1, 2],
            [1, 2, 4, 5],
            [5, 5, 7, 0, 232]
        ]
        
        expected = [0.6666666666666666, 0.5, 0, 0.33333333333333337, 0.8, 0.3333333333333333, 0.28571428571428575, 0.5714285714285715, 0.25]

        cases = []
        pairs = list(combinations(tensors, 2))
        
        for (candidate, reference), solution in zip(pairs, expected):
            answer = learner_func(candidate, reference)
            t = test_case()
            if not math.isclose(answer, solution):
                t.failed = True
                t.msg = f"Incorrect similarity for candidate={candidate} and reference={reference}"
                t.want = solution
                t.got = answer
            cases.append(t)

        return cases
    
    cases = g()
    print_feedback(cases)


def test_average_overlap(learner_func):

    def jaccard_similarity(candidate, reference):
        
        # Convert the lists to sets to get the unique tokens
        candidate_set = set(candidate)
        reference_set = set(reference)
        
        # Get the set of tokens common to both candidate and reference
        common_tokens = candidate_set.intersection(reference_set)
        
        # Get the set of all tokens found in either candidate or reference
        all_tokens = candidate_set.union(reference_set)
        
        # Compute the percentage of overlap (divide the number of common tokens by the number of all tokens)
        overlap = len(common_tokens) / len(all_tokens)
            
        return overlap
    
    def g():
        
        l1 = [1, 2, 3]
        l2 = [1, 2, 4]
        l3 = [1, 2, 4, 5]
        l4 = [5,6]

        elements = [l1, l2, l3, l4]

        all_combinations = []

        for r in range(2, len(elements) + 1):
            # Generate combinations of length r
            combinations_r = combinations(elements, r)
            
            # Append the combinations to the result list
            all_combinations.extend(combinations_r)
        
        expected = [{0: 0.5, 1: 0.5},
                     {0: 0.4, 1: 0.4},
                     {0: 0.0, 1: 0.0},
                     {0: 0.75, 1: 0.75},
                     {0: 0.0, 1: 0.0},
                     {0: 0.2, 1: 0.2},
                     {0: 0.45, 1: 0.625, 2: 0.575},
                     {0: 0.25, 1: 0.25, 2: 0.0},
                     {0: 0.2, 1: 0.3, 2: 0.1},
                     {0: 0.375, 1: 0.475, 2: 0.1},
                     {0: 0.3, 1: 0.417, 2: 0.45, 3: 0.067}]

        cases = []
        
        for combination, solution in zip(all_combinations, expected):
            answer = learner_func(combination, jaccard_similarity)
            t = test_case()
            if answer != solution:
                t.failed = True
                t.msg = f"Incorrect overlap for lists={combination}"
                t.want = solution
                t.got = answer
            cases.append(t)

        return cases
    
    cases = g()
    print_feedback(cases)
