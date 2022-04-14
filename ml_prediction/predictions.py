import numpy as np
import tensorflow as tf
from pathlib import Path
from transformers import AutoTokenizer, TFGPT2LMHeadModel, AutoConfig
import pickle

project_root = Path(__file__).resolve().parent
gpt2_model_dir = project_root / 'mlmodels/bn_gpt2'
latm_model_dir = project_root / 'mlmodels/bn_lstm-2'

tokenizer_dir = gpt2_model_dir / 'tokenizer'
model_weights_path = gpt2_model_dir / 'gpt2_model_weights.h5'

MAX_WORDS = 10000
MAX_SEQUENCE_LENGTH = 200

def predict_gpt2(text):
    config = AutoConfig.from_pretrained("flax-community/gpt2-bengali")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    model = TFGPT2LMHeadModel.from_pretrained(str(model_weights_path), config=config)

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


def predict_lstm(text: str):
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
    ids = np.argsort(predictions, axis=1)[:,-6:] # indices of the top 5 predictions
    # print next word with score

    res = []

    for id in ids[0][0:len(ids[0])-1]:
        # print(tokenizer.index_word[id], "->", predictions[:, id].squeeze())
        res.append(f"{tokenizer.index_word[id]} -> {predictions[:, id].squeeze()}")

    return res