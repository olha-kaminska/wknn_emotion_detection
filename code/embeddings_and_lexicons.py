import json
import numpy as np
import tensorflow_hub as hub
import torch

from scipy.spatial.distance import cosine
from collections import defaultdict

# torchmoji 
from torchmoji.sentence_tokenizer import SentenceTokenizer
from torchmoji.model_def import torchmoji_feature_encoding 
from torchmoji.global_variables import PRETRAINED_PATH, VOCAB_PATH
from transformers import AutoTokenizer, AutoModel, TFAutoModel
# roBERTa model 
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
# Word2Vec
from gensim.models import KeyedVectors
# SBERT
from sentence_transformers import SentenceTransformer
# BERT
from transformers import BertTokenizer, BertModel

# Uncomment to preload all models here
'''
# path to the pre-loaded roBERTa model 
MODEL_path_roberta = r"..\model\twitter-roberta-base-emotion"
tokenizer_roberta = AutoTokenizer.from_pretrained(MODEL_path_roberta)
model_roberta = TFAutoModel.from_pretrained(MODEL_path_roberta)
# load pre-trained model tokenizer (vocabulary)
tokenizer_bert = BertTokenizer.from_pretrained('bert-base-uncased')
# upload BERT model 
model_bert = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True)
# upload Sentence-BERT model from the 'sentence_transformers' package 
model_sbert = SentenceTransformer('distilbert-base-nli-mean-tokens')
# upload the big Universal Sentence Encoder model from HTTPS domain 
model_use = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
# path to the pre-loaded Word2Vec model 
w2v_path = '../model/GoogleNews-vectors-negative300.bin'
w2v_model = KeyedVectors.load_word2vec_format(w2v_path, binary=True)
'''

def get_vector_roberta(text, tokenizer_roberta, model_roberta):
    '''
    This function provides Twitter-roBERTa-based embedding for the tweet.
    
    Input: tweet as a string, preloaded tokenizer_roberta and model_roberta
    Output: 768-dimentional vector  
    '''
    encoded_input = tokenizer_roberta(text, return_tensors='tf')
    features = model_roberta(encoded_input)
    features = features[0].numpy()
    features_mean = features[0][-1]
    return features_mean
    
def get_vector_bert(text, tokenizer_bert, model_bert):
    '''
    This function provides BERT embedding for the tweet.
    
    It uses "torch" library to work with tensors.
    
    Input: tweet as a string, preloaded tokenizer_bert and model_bert
    Output: 768-dimentional vector as 
    '''
    marked_text = "[CLS] " + text + " [SEP]"
    # Tokenize tweet with the BERT tokenizer
    tokenized_text = tokenizer_bert.tokenize(marked_text)
    indexed_tokens = tokenizer_bert.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1] * len(tokenized_text)
    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    # Make vector 
    with torch.no_grad():
        outputs = model_bert(tokens_tensor, segments_tensors)
        hidden_states = outputs[2]
    token_vecs = hidden_states[-2][0]
    sentence_embedding = torch.mean(token_vecs, dim=0)
    return sentence_embedding.cpu().detach().numpy()
    
def get_vector_sbert(tweet, model_sbert):
    '''
    This function provides Sentence-BERT embedding for the tweet.
    
    Input: tweet as a string, preloaded model_sbert
    Output: 768-dimentional vector as a list 
    '''
    return model_sbert.encode(tweet)

def get_vectors_deepmoji(tweets):
    '''
    This function provides DeepMoji (torchmoji) embedding for the tweet.
    
    It uses "json" library to upload DeepMoji vocabulary.
    
    Input: tweets - a list of strings
    Output: 2304-dimentional vector as a list 
    '''
    maxlen = 30
    batch_size = 32
    with open(VOCAB_PATH, 'r') as f:
        vocabulary = json.load(f)
    st = SentenceTokenizer(vocabulary, maxlen)
    tokenized, _, _ = st.tokenize_sentences(tweets)
    model = torchmoji_feature_encoding(PRETRAINED_PATH)
    encoding = model(tokenized)
    vectors = []
    for i in range(len(encoding)):
        vectors.append(encoding[i,:])
    return vectors
    
def get_vector_use(tweet, model_use):
    '''
    This function provides Universal Sentence Encoder embedding for the tweet.
    
    It uses "numpy" library as "np" to process vectors.
    
    Input: tweet as a string, preloaded model_use
    Output: 512-dimentional vector as a list 
    '''
    message_embeddings = model_use([tweet])
    mess_list = np.array(message_embeddings).tolist()
    return mess_list[0]
    
def get_vector_w2v(tweet, w2v_model):
    '''
    This function provides Word2Vec embedding for the tweet.
    
    It uses "numpy" library as "np" to create zeros vectors.
    
    Input: tweet as a string, preloaded Word2Vec model
    Output: 300-dimentional vector as a list 
    '''
    vectors = []
    for w in tweet.split():
        if w in w2v_model.wv.vocab:
            vectors.append(w2v_model.wv.__getitem__(w))
        else:
            vectors.append(np.zeros(300))
    return np.mean(vectors, axis = 0)

def get_lexicon_scores(word, lexicon, lexicon_words, size):
    '''
    This function provides lexicon scores for a word.
    
    It uses "numpy" library as "np".
    
    Input: - word: string, the target word for which we want to have lexicon scores
           - lexicon: lexicon as a DataFrame, with "lexicon_words" columns 
           - lexicon_words: string, name of the column in lexicon dataframe, which corresponds to the column with a list of words
           - size: int, number of columns in lexicon dataframe with numerious scores 
    Output: vector with size "size" with lexicon scores or zeros, if "word" is not in lexicon
    '''
    if word in lexicon[lexicon_words].to_list():
        word_values = lexicon[lexicon[lexicon_words]==word]
        ind = word_values.index.values.astype(int)[0]
        results = []
        for col in lexicon.columns.to_list():
            results.append(word_values[col][ind])
        return np.array(results[1:])+1
    else:
        return np.ones(size)
        
def append_lexicon_scores(sent, vec, lexicon_data): 
    '''
    This function appends one lexicon scores to the embedding vector.
    
    It uses "numpy" library as "np".
    
    Input: - sent: string, the sentence with the target word for which we want to have lexicon scores, separated by spaces
           - vec: array, the embedding vector 
           - lexicon_data: array size 5, contains the information about lexicon in a format:
                           - DataFrame with lexicon
                           - string with the name of column from the DataFrame with list of words
                           - int, number of numerical columns with lexicon scores in DataFrame, all columns besides the one with the list of words  
                           - int, the lower margin of the lexicon scores 
                           - int, the upper margin of the lexicon scores 
    Output: concataneted vector of embedding and lexicon scores 
    '''
    words_vectors = []
    for w in sent.split():
        words_vectors.append(normalize(get_lexicon_scores(w, lexicon_data[0], lexicon_data[1], lexicon_data[2]), lexicon_data[3], lexicon_data[4]))
    return np.append(normalize(vec, -1, 1), np.mean(words_vectors, axis=0))
    
def append_two_lexicon_scores(sent, vec, lexicon_data1, lexicon_data2):
    '''
    This function appends two lexicons scores to the embedding vector.
    
    It uses "numpy" library as "np".
    
    Input: - sent: string, the sentence with the target word for which we want to have lexicon scores, separated by spaces
           - vec: array, the embedding vector 
           - lexicon_data1, lexicon_data2: array size 5, contains the information about lexicon in a format:
                           - DataFrame with lexicon
                           - string with the name of column from the DataFrame with list of words
                           - int, number of numerical columns with lexicon scores in DataFrame, all columns besides the one with the list of words  
                           - int, the lower margin of the lexicon scores 
                           - int, the upper margin of the lexicon scores 
    Output: concataneted vector of embedding and lexicon scores 
    '''
    words_vectors = []
    for w in sent.split():
        words_vectors.append(np.append(normalize(get_lexicon_scores(w, lexicon_data1[0], lexicon_data1[1], lexicon_data1[2]), lexicon_data1[3], lexicon_data1[4]), normalize(get_lexicon_scores(w, lexicon_data2[0], lexicon_data2[1], lexicon_data2[2]), lexicon_data2[3], lexicon_data2[4])))
    return np.append(normalize(vec, -1, 1), np.mean(words_vectors, axis=0))

def append_three_lexicon_scores(sent, vec, lexicon_data1, lexicon_data2, lexicon_data3):
    '''
    This function appends three lexicons scores to the embedding vector.
    
    It uses "numpy" library as "np".
    
    Input: - sent: string, the sentence with the target word for which we want to have lexicon scores, separated by spaces
           - vec: array, the embedding vector 
           - lexicon_data1, lexicon_data2, lexicon_data3: array size 5, contains the information about lexicon in a format:
                           - DataFrame with lexicon
                           - string with the name of column from the DataFrame with list of words
                           - int, number of numerical columns with lexicon scores in DataFrame, all columns besides the one with the list of words  
                           - int, the lower margin of the lexicon scores 
                           - int, the upper margin of the lexicon scores 
    Output: concataneted vector of embedding and lexicon scores 
    '''
    words_vectors = []
    for w in sent.split():
        words_vectors.append(np.append(normalize(get_lexicon_scores(w, lexicon_data1[0], lexicon_data1[1], lexicon_data1[2]), lexicon_data1[3], lexicon_data1[4]), np.append(normalize(get_lexicon_scores(w, lexicon_data2[0], lexicon_data2[1], lexicon_data2[2]), lexicon_data2[3], lexicon_data2[4]), normalize(get_lexicon_scores(w, lexicon_data3[0], lexicon_data3[1], lexicon_data3[2]), lexicon_data3[3], lexicon_data3[4]))))
    return np.append(normalize(vec, -1, 1), np.mean(words_vectors, axis=0))

def append_four_lexicon_scores(sent, vec, lexicon_data1, lexicon_data2, lexicon_data3, lexicon_data4):
    '''
    This function appends four lexicons scores to the embedding vector.
    
    It uses "numpy" library as "np".
    
    Input: - sent: string, the sentence with the target word for which we want to have lexicon scores, separated by spaces
           - vec: array, the embedding vector 
           - lexicon_data1, lexicon_data2, lexicon_data3, lexicon_data4: array size 5, contains the information about lexicon in a format:
                           - DataFrame with lexicon
                           - string with the name of column from the DataFrame with list of words
                           - int, number of numerical columns with lexicon scores in DataFrame, all columns besides the one with the list of words  
                           - int, the lower margin of the lexicon scores 
                           - int, the upper margin of the lexicon scores 
    Output: concataneted vector of embedding and lexicon scores 
    '''
    words_vectors = []
    for w in sent.split():
        words_vectors.append(np.append(normalize(get_lexicon_scores(w, lexicon_data1[0], lexicon_data1[1], lexicon_data1[2]), lexicon_data1[3], lexicon_data1[4]), np.append(normalize(get_lexicon_scores(w, lexicon_data2[0], lexicon_data2[1], lexicon_data2[2]), lexicon_data2[3], lexicon_data2[4]), np.append(normalize(get_lexicon_scores(w, lexicon_data3[0], lexicon_data3[1], lexicon_data3[2]), lexicon_data3[3], lexicon_data3[4]), normalize(get_lexicon_scores(w, lexicon_data4[0], lexicon_data4[1], lexicon_data4[2]), lexicon_data4[3], lexicon_data4[4])))))
    return np.append(normalize(vec, -1, 1), np.mean(words_vectors, axis=0))

def append_five_lexicon_scores(sent, vec, lexicon_data1, lexicon_data2, lexicon_data3, lexicon_data4, lexicon_data5):
    '''
    This function appends five lexicons scores to the embedding vector.
    
    It uses "numpy" library as "np".
    
    Input: - sent: string, the sentence with the target word for which we want to have lexicon scores, separated by spaces
           - vec: array, the embedding vector 
           - lexicon_data1, lexicon_data2, lexicon_data3, lexicon_data4, lexicon_data5: array size 5, contains the information about lexicon in a format:
                           - DataFrame with lexicon
                           - string with the name of column from the DataFrame with list of words
                           - int, number of numerical columns with lexicon scores in DataFrame, all columns besides the one with the list of words  
                           - int, the lower margin of the lexicon scores 
                           - int, the upper margin of the lexicon scores 
    Output: concataneted vector of embedding and lexicon scores 
    '''
    words_vectors = []
    for w in sent.split():
        words_vectors.append(np.append(normalize(get_lexicon_scores(w, lexicon_data1[0], lexicon_data1[1], lexicon_data1[2]), lexicon_data1[3], lexicon_data1[4]), 
                                       np.append(normalize(get_lexicon_scores(w, lexicon_data2[0], lexicon_data2[1], lexicon_data2[2]), lexicon_data2[3], lexicon_data2[4]), 
                                         np.append(normalize(get_lexicon_scores(w, lexicon_data3[0], lexicon_data3[1], lexicon_data3[2]), lexicon_data3[3], lexicon_data3[4]), 
                                           np.append(normalize(get_lexicon_scores(w, lexicon_data4[0], lexicon_data4[1], lexicon_data4[2]), lexicon_data4[3], lexicon_data4[4]),normalize(get_lexicon_scores(w, lexicon_data5[0], lexicon_data5[1], lexicon_data5[2]), lexicon_data5[3], lexicon_data5[4]))))))
    return np.append(normalize(vec, -1, 1), np.mean(words_vectors, axis=0))