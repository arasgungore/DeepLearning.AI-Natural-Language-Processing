import numpy as np
import tensorflow as tf
import tensorflow_text as tf_text
import pathlib

path_to_file = pathlib.Path("por-eng/por.txt")

np.random.seed(1234)
tf.random.set_seed(1234)

def load_data(path):
    text = path.read_text(encoding="utf-8")

    lines = text.splitlines()
    pairs = [line.split("\t") for line in lines]

    context = np.array([context for target, context, _ in pairs])
    target = np.array([target for target, context, _ in pairs])

    return context, target


portuguese_sentences, english_sentences = load_data(path_to_file)

sentences = (portuguese_sentences, english_sentences)

BUFFER_SIZE = len(english_sentences)
BATCH_SIZE = 64

is_train = np.random.uniform(size=(len(portuguese_sentences),)) < 0.8

train_raw = (
    tf.data.Dataset.from_tensor_slices(
        (english_sentences[is_train], portuguese_sentences[is_train])
    )
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE)
)
val_raw = (
    tf.data.Dataset.from_tensor_slices(
        (english_sentences[~is_train], portuguese_sentences[~is_train])
    )
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE)
)


def tf_lower_and_split_punct(text):
    text = tf_text.normalize_utf8(text, "NFKD")
    text = tf.strings.lower(text)
    text = tf.strings.regex_replace(text, "[^ a-z.?!,¿]", "")
    text = tf.strings.regex_replace(text, "[.?!,¿]", r" \0 ")
    text = tf.strings.strip(text)
    text = tf.strings.join(["[SOS]", text, "[EOS]"], separator=" ")
    return text


max_vocab_size = 12000

english_vectorizer = tf.keras.layers.TextVectorization(
    standardize=tf_lower_and_split_punct, max_tokens=max_vocab_size, ragged=True
)

english_vectorizer.adapt(train_raw.map(lambda context, target: context))

portuguese_vectorizer = tf.keras.layers.TextVectorization(
    standardize=tf_lower_and_split_punct, max_tokens=max_vocab_size, ragged=True
)

portuguese_vectorizer.adapt(train_raw.map(lambda context, target: target))


def process_text(context, target):
    context = english_vectorizer(context).to_tensor()
    target = portuguese_vectorizer(target)
    targ_in = target[:, :-1].to_tensor()
    targ_out = target[:, 1:].to_tensor()
    return (context, targ_in), targ_out


train_data = train_raw.map(process_text, tf.data.AUTOTUNE)
val_data = val_raw.map(process_text, tf.data.AUTOTUNE)

del train_raw
del val_raw


def masked_loss(y_true, y_pred):
    
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    loss = loss_fn(y_true, y_pred)
    
    # Check which elements of y_true are padding
    mask = tf.cast(y_true != 0, loss.dtype)
    
    loss *= mask
    # Return the total.
    return tf.reduce_sum(loss)/tf.reduce_sum(mask)


def masked_acc(y_true, y_pred):
    y_pred = tf.argmax(y_pred, axis=-1)
    y_pred = tf.cast(y_pred, y_true.dtype)
    match = tf.cast(y_true == y_pred, tf.float32)
    mask = tf.cast(y_true != 0, tf.float32)

    return tf.reduce_sum(match)/tf.reduce_sum(mask)


def tokens_to_text(tokens, id_to_word):
    words = id_to_word(tokens)
    result = tf.strings.reduce_join(words, axis=-1, separator=" ")
    return result
