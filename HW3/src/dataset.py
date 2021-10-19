import os, re, gc
import pandas as pd

import tensorflow as tf
import tensorflow_datasets as tfds


def preprocess_text_nonbreaking(corpus, non_breaking_prefixes):
    corpus_cleaned = corpus
    # Add the string $$$ before the non breaking prefixes
    # To avoid remove dots from some words
    for prefix in non_breaking_prefixes:
        corpus_cleaned = corpus_cleaned.replace(prefix, prefix + '$$$')
    # Remove dots not at the end of a sentence
    corpus_cleaned = re.sub(r"\.(?=[0-9]|[a-z]|[A-Z])", ".$$$", corpus_cleaned)
    # Remove the $$$ mark
    corpus_cleaned = re.sub(r"\.\$\$\$", '', corpus_cleaned)
    # Remove multiple white spaces
    corpus_cleaned = re.sub(r"  +", " ", corpus_cleaned)

    return corpus_cleaned

def subword_tokenize(corpus, vocab_size, max_length, return_as_dict=False):
    # Create the vocabulary using Subword tokenization
    tokenizer_corpus = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(corpus, target_vocab_size=vocab_size)
    # Get the final vocab size, adding the eos and sos tokens
    num_words = tokenizer_corpus.vocab_size + 2
    # Set eos and sos token
    sos_token = [num_words-2]
    eos_token = [num_words-1]
    # Tokenize the corpus
    sentences = [sos_token + tokenizer_corpus.encode(sentence) + eos_token
                 for sentence in corpus]
    # Identify the index of the sentences longer than max length
    idx_to_remove = [count for count, sent in enumerate(sentences)
                    if len(sent) > max_length]
    #Pad the sentences
    sentences = tf.keras.preprocessing.sequence.pad_sequences(sentences, value=0, padding='post', maxlen=max_length)

    if return_as_dict:
        return dict(
            sentences           = sentences,
            tokenizer_corpus    = tokenizer_corpus,
            num_words           = num_words,
            sos_token           = sos_token,
            eos_token           = eos_token,
            idx_to_remove       = idx_to_remove
        )

    return sentences, tokenizer_corpus, num_words, sos_token, eos_token, idx_to_remove

def load_datasets(data_files, num_samples=80000, max_vocab_size=2**14, max_length=15, batch_size=64):

    main_keys = ['input', 'target']
    df = pd.read_csv(data_files['train'], sep="\t", header=None, names=main_keys, usecols=[0,1], nrows=num_samples)

    token_dset = dict()
    len_data = dict()

    for k_dat in main_keys:
        with open(data_files[k_dat], mode = "r", encoding = "utf-8") as f:
            non_breaking_prefix = f.read()
        non_breaking_prefix = [' ' + pref + '.' for pref in non_breaking_prefix.split("\n")]

        pre_data = df[k_dat].apply(lambda x : preprocess_text_nonbreaking(x, non_breaking_prefix)).tolist()

        len_data[k_dat] = len(pre_data)
        token_dset[k_dat] = subword_tokenize(pre_data, max_vocab_size, max_length, return_as_dict=True)

    del df
    gc.collect()

    dataset = tf.data.Dataset.from_tensor_slices((token_dset['input']['sentences'], token_dset['target']['sentences']))
    dataset = dataset.shuffle(len_data['input'], reshuffle_each_iteration=True).batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset, token_dset