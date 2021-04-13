# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 15:53:38 2021

@author: Clement_Bureau
"""


"""
Ce script sert de module pour utiliser un modele.
Ce script est long a charger, il est conseiller de ne faire l'import qu'une seule fois.


    Comment utiliser ce module ?

Apres avoir import le module, il faut charger un modele :
    
    Utiliser la fonction charge_model('path) : 'path' le chemin jusqu'au dossier du modele 


La fonction à utiliser pour tester une phrase :
    
    evaluate_sentiment(list_sentences) : retourne les sentiments (liste de int) de la liste de phrases 'list_sentences' (liste de chaine de caractere) 
    
    


Si il vous manque un module lors du chargement du script, utilisez la commande dans la console "!pip install nom_du_module_manquant" 

Pour bert : !pip install bert-for-tf2
"""



import tensorflow_hub as hub
from tensorflow import keras
import nltk
import re
nltk.download("stopwords")
from nltk.corpus import stopwords
import bert
import numpy as np




# Nom du Model :    
model=None
# Le Pad :
pad=None
#pad =44

def charge_model(nom_model):
    global model
    model=keras.models.load_model(nom_model)
 

#########################################################################################
################################ NE PAS MODIFIER LA SUITE ###############################
#########################################################################################

BertTokenizer = bert.bert_tokenization.FullTokenizer
bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
                            trainable=False)
vocabulary_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
to_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = BertTokenizer(vocabulary_file, to_lower_case)

def text_preprocessing(s):
    """
    - Lowercase the sentence
    - Change "'t" to "not"
    - Remove "@name"
    - Isolate and remove punctuations except "?"
    - Remove other special characters
    - Remove stop words except "not" and "can"
    - Remove trailing whitespace
    """
    s = s.lower()
    # Change 't to 'not'
    s = re.sub(r"\'t", " not", s)
    # Remove @name
    s = re.sub(r'(@.*?)[\s]', ' ', s)
    # Isolate and remove punctuations except '?'
    s = re.sub(r'([\'\"\.\(\)\!\?\\\/\,])', r' \1 ', s)
    s = re.sub(r'[^\w\s\?]', ' ', s)
    # Remove some special characters
    s = re.sub(r'([\;\:\|•«\n])', ' ', s)
    # Remove stopwords except 'not' and 'can'
    s = " ".join([word for word in s.split()
                  if word not in stopwords.words('english')
                  or word in ['not', 'can']])
    # Remove trailing whitespace
    s = re.sub(r'\s+', ' ', s).strip()
    
    return s


def tokenize_reviews(text_reviews):
    return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text_reviews))

def PaddingX(M,pad=-1):
  if pad==-1: 
    pad = len(max(M, key=len))
  return np.array([i + [0]*(pad-len(i)) for i in M]),pad


def evaluate_proba(sentences):
  sentences_preprocessed = [text_preprocessing(s) for s in sentences]
  sentences_tokenized = [tokenize_reviews(s) for s in sentences_preprocessed]
  sentences_padded,p =PaddingX(sentences_tokenized)
  return model.predict(sentences_padded)



# Donne le sentiment donné par le modele 'model' sur la phrase 'sentence'   (ATTENTION A COMMENT SONT NUMEROTES LES SENTIMENTS)
# Exemple : sentiment = evaluate_sentiment('The flight is good')
def evaluate_sentiment(sentences):
    proba=evaluate_proba(sentences)
    return [np.argmax(p) for p in proba]