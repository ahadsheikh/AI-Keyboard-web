import numpy as np
import tensorflow as tf
from pathlib import Path
from transformers import AutoTokenizer, TFGPT2LMHeadModel, AutoConfig
import pickle
import pandas as pd

project_root = Path(__file__).resolve().parent
gpt2_model_dir = project_root / 'mlmodels/bn_gpt2'
latm_model_dir = project_root / 'mlmodels/bn_lstm'

tokenizer_dir = gpt2_model_dir / 'tokenizer'
model_weights_path = gpt2_model_dir / 'gpt2_model_weights.h5'
config_path = gpt2_model_dir / 'config.json'

MAX_WORDS = 10000
MAX_SEQUENCE_LENGTH = 200

def predict_gpt2(text):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    model = TFGPT2LMHeadModel.from_pretrained(str(model_weights_path), config=str(config_path))

    input_ids = tokenizer.encode(text, return_tensors='tf')
    outputs = model.predict(input_ids).logits

    beam_outputs = model.generate(
        input_ids,
        max_length=100,
        num_beams=5,
        no_repeat_ngram_size=2,
        num_return_sequences=5,
        early_stopping=True,
        do_sample=True,
        top_k=50,
        top_p=0.95,
    )

    res = []

    for i, beam_output in enumerate(beam_outputs):
        res.append("{}".format(tokenizer.decode(
            beam_output, skip_special_tokens=True)))
    
    return res


def predict_lstm(text: pd.DataFrame):
    model = tf.keras.models.load_model(str(latm_model_dir))
    with open(str(latm_model_dir / f'tokenizer_{MAX_WORDS}_words.pickle'), 'rb') as f:
        tokenizer = pickle.load(f)

    # tokenize the text
    tokenized_sequence = tokenizer.texts_to_sequences([text])
    # pre-pad with 0's to make it of size MAX_SEQUENCE_LENGTH - 1
    input_sequences = tf.keras.preprocessing.sequence.pad_sequences(tokenized_sequence,
                                                                maxlen=MAX_SEQUENCE_LENGTH - 1,
                                                                padding='pre')
    # predict using model
    predictions = model.predict(input_sequences)
    ids = np.argsort(predictions, axis=1)[:,-10:] # indices of the top 5 predictions
    # print next word with score

    words = []
    probs = []

    for id in ids[0]:
        # print(tokenizer.index_word[id], "->", predictions[:, id].squeeze())
        words.append(tokenizer.index_word[id])
        probs.append(str(predictions[:, id].squeeze()))

    words.reverse()
    probs.reverse()

    if '<oov>' in words:
        i = words.index('<oov>')
        words.pop(i)
        probs.pop(i)

    return pd.DataFrame({'Word': words, 'Probability': probs})
