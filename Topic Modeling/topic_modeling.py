# import libraries needed
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import re

from nltk.stem import PorterStemmer
from gensim import corpora, models
from gensim.utils import simple_preprocess
from nltk.stem import WordNetLemmatizer
import nltk

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation

import pyLDAvis
import pyLDAvis.sklearn

from collections import Counter
import itertools
from wordcloud import WordCloud, STOPWORDS
from nltk.corpus import wordnet


nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
sns.set_style('whitegrid')

from read_file import * # import functions used to read data files


def cleanTxt(text):
    """
    Function to clean str text
    
    Parameters
    ----------
    text : str
        text before cleaning.

    Returns
    -------
    text : str
        text after cleaning.

    """
    # regular expressions to recognize 'USA' (to prevent the confusion of us and US)
    text = re.sub(r'@US+([^a-z]|$)','usa ',text)
    text = re.sub(r'@us+([^a-z]|$)','usa ',text)
    text = re.sub(r'@Us+([^a-z]|$)','usa ',text)
    text = re.sub(r'@uS+([^a-z]|$)','usa ',text)
    text = re.sub(r'#US+([^a-z]|$)','usa ',text)
    text = re.sub(r'#us+([^a-z]|$)','usa ',text)
    text = re.sub(r'#Us+([^a-z]|$)','usa ',text)
    text = re.sub(r'#uS+([^a-z]|$)','usa ',text)
    text = re.sub(r'#U.S.A+([^a-z]|$)','usa ',text)
    text = re.sub(r'#U.S+([^a-z]|$)','usa ',text)
    text = re.sub(r'#u.s+([^a-z]|$)','usa ',text)
    text = re.sub(r'#u.S+([^a-z]|$)','usa ',text)
    text = re.sub(r'#U.s+([^a-z]|$)','usa ',text)
    text = re.sub(r'US+([^a-z]|$)','usa ',text)
    text = re.sub(r'U.S+([^a-z]|$)','usa ',text)
    text = re.sub(r'U.s+([^a-z]|$)','usa ',text)
    text = re.sub(r'u.s+([^a-z]|$)','usa ',text)
    text = re.sub(r'u.S+([^a-z]|$)','usa ',text)
    

    text = re.sub(r'@','',text) # Remove @ without the word 
    text = re.sub(r'&.+;','',text) #remove html special entities (&amp; ...)
    text = re.sub(r'\n',' ',text) # Remove \n (newline)
    text = re.sub(r'#','',text) # Remove the # symbol
    text = re.sub(r'RT[\s]+','',text) # Remove RT (retweet)
    text = re.sub(r'http?:\/\/\S+','',text) # Remove the hyper link
    text = re.sub(r'https?:\/\/\S+','',text)
    text = re.sub(' +', ' ', text) # remove multiple whitespace from text 
    text = re.sub(r'travel+([^a-z]|$)',' ',text) # Remove 'travel' the principal keyword in our data only
    text = re.sub(r'travelling','traveling',text) # travelling -> traveling
    text = re.sub(r'coronavirus','covid 19',text) # coronavirus -> covid 19
    text = re.sub(r'covid19','covid 19',text) # covid19 -> covid 19
    text = re.sub(' +', ' ', text) # remove multiple whitespace from text
    return text



def cleanData(df, col):
    """
    Function to clean df : pandas.core.frame.DataFrame
    
    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        data before cleaning.
    col : str
        column where tweets are stored.

    Returns
    -------
    df : pandas.core.frame.DataFrame
        data after cleaning.

    """
    
    
    
    df[col] = df[col].map(lambda x: re.sub('[,\.!?"]', '', x)) # Remove punctuation
    df[col]= df[col].map(lambda x: x.lower()) # Convert the titles to lowercase
    
    # apply clenTxt defined before on the data
    df[col] = df[col].apply(cleanTxt)
    
    # remove emojis
    df[col] = df[col].apply(lambda x: x.encode('ascii', 'ignore').decode('ascii'))
    
    
    # Note : We don't drop NaN and Empty rows to conserve the original IDs of tweets
    # the commented code below drops NaN rows
    """
    # Drop NaN
    #nan_value = float("NaN")
    #df.replace("", nan_value, inplace=True)
    #df.dropna(subset = [col], inplace=True)
    #df = df.reset_index(drop=True)
    """
    
    return df
    


def wordcloud_data(df, col):
    """
    Shows the wordcloud of the data
    
    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        data.
    col : str
        column where tweets are stored.

    Returns
    -------
    None.

    """
    
    # Join the different processed titles together.
    long_string = ','.join(list(df[col].values))
    # Create a WordCloud object
    wordcloud = WordCloud(background_color="white", max_words=10000, contour_width=10, contour_color='steelblue')
    # Generate a word cloud
    wordcloud.generate(long_string)
    #wordcloud.to_image()
    plt.figure( figsize=(20,10), facecolor='k')
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()


def wordBarGraphFunction(df,column):
    """

    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        data (preferd cleaned one).
    col : str
        column where tweets are stored.

    Returns
    -------
    None.

    """
    topic_words = [ z.lower() for y in
                       [ x.split() for x in df[column] if isinstance(x, str)]
                       for z in y]
    word_count_dict = dict(Counter(topic_words))
    popular_words = sorted(word_count_dict, key = word_count_dict.get, reverse = True)
    popular_words_nonstop = [w for w in popular_words if w not in STOPWORDS]
    plt.barh(range(20), [word_count_dict[w] for w in reversed(popular_words_nonstop[0:20])])
    plt.yticks([x + 1 for x in range(20)], reversed(popular_words_nonstop[0:20]))
    plt.title("Most common words")
    plt.show()
    



def lemmatize(text):
    """
    Stemmers remove morphological affixes from words, leaving only the word stem.
    <Lemmatization is similar to stemming but it brings context to the words>
    <Lemmatization gives interesting results but still very slow to use>
    <we don't use this function in our implementation of models!!>
    # https://www.geeksforgeeks.org/python-lemmatization-approaches-with-examples/


    Parameters
    ----------
    text : str
    Returns
    -------
    str
        text after lemmatizing.
    """
      
    wnl = WordNetLemmatizer()
    stemmer = PorterStemmer()
    wnl = WordNetLemmatizer()
      
    # Define function to lemmatize each word with its POS tag 
      
    # POS_TAGGER_FUNCTION : TYPE 1 
    def pos_tagger(nltk_tag): 
        if nltk_tag.startswith('J'): 
            return wordnet.ADJ 
        elif nltk_tag.startswith('V'): 
            return wordnet.VERB 
        elif nltk_tag.startswith('N'): 
            return wordnet.NOUN 
        elif nltk_tag.startswith('R'): 
            return wordnet.ADV 
        else:           
            return None
            
    # tokenize the sentence and find the POS tag for each token 
    pos_tagged = nltk.pos_tag(nltk.word_tokenize(text))   
    # we use our own pos_tagger function to make things simpler to understand. 
    wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged)) 
      
    lemmatized_sentence = [] 
    for word, tag in wordnet_tagged: 
        if tag is None: 
            # if there is no available tag, append the token as is 
            lemmatized_sentence.append(word) 
        else:         
            # else use the tag to lemmatize the token 
            lemmatized_sentence.append(wnl.lemmatize(word, tag)) 
    lemmatized_sentence = " ".join(lemmatized_sentence) 
      
    return lemmatized_sentence

    

def lemmatize_steming(text):
    """
    Stemmers remove morphological affixes from words, leaving only the word stem.
    <Lemmatization is similar to stemming but it brings context to the words>
    
    Fast way to lemmatizing and stemming : (we only lemmitize verbs)

    Parameters
    ----------
    text : str
           (only one word !!!)
    Returns
    -------
    str
        text after stemming.
        (only one word !!!)
    """
      
    stemmer = PorterStemmer()
    wnl = WordNetLemmatizer()
    return stemmer.stem(wnl.lemmatize(text, pos='v')) #v -> verb

def preprocess(text):
    """
    Function that we can apply to DataFrame to clean text

    Parameters
    ----------
    text : str

    Returns
    -------
    result : str
        string after applying lemmatize_steming

    """
    result = ""
    for token in simple_preprocess(text):
        if token not in STOPWORDS and len(token) > 3:
            result = result + " " + lemmatize_steming(token)
    return result


def NMF_model(df, col, no_features):
    """
    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        data (preferd cleaned one).
    col : str
        column where tweets are stored.
    no_features : int
        number of clusters (number of topics).

    Returns
    -------
    Nmf_model : sklearn.decomposition._nmf.NMF
        NMF model
    Nmf_model.fit(tfidf) : sklearn.decomposition._nmf.NMF
        NMF model
    tfidf_feature_names : list
        list of words in data
    tfidf : scipy.sparse.csr.csr_matrix
        matrix storing tfidf of the our data
    """
    # NMF is able to use tf-idf
    tfidf_vectorizer = TfidfVectorizer(max_df=0.9, min_df=2, max_features=no_features, stop_words='english')
    tfidf = tfidf_vectorizer.fit_transform(df[col])
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    Nmf_model = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd')
    
    return Nmf_model, Nmf_model.fit(tfidf), tfidf_feature_names, tfidf
    
    
def LDA_model(df, col, no_features):
    """
    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        data (preferd cleaned one).
    col : str
        column where tweets are stored.
    no_features : int
        number of clusters (number of topics).

    Returns
    -------
    Lda_model.fit(tf): sklearn.decomposition._lda.LatentDirichletAllocation
        NMF model
    tf_feature_names : list
        list of words in data
    tf : scipy.sparse.csr.csr_matrix
        matrix storing tf of the our data
    """
    # LDA can only use raw term counts for LDA because it is a probabilistic graphical model
    tf_vectorizer = CountVectorizer(max_df=0.9, min_df=5, max_features=no_features, stop_words='english')
    tf = tf_vectorizer.fit_transform(df[col])
    tf_feature_names = tf_vectorizer.get_feature_names()
    Lda_model = LatentDirichletAllocation(n_components=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0)
    
    return Lda_model.fit(tf), tf_feature_names, tf
    
    

def display_topics(model, feature_names, no_top_words):
    """
    display topics results of LDA or NMF

    Parameters
    ----------
    model : 
        NMF or LDA model
    feature_names : list
        
    no_top_words : TYPE
        number of words to show in a topic

    Returns
    -------
    None.

    """
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % (topic_idx))
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))
        
        
def get_clusters(model, tf, save, name="clusters"):
    """
    this function calculates a matrix where (i,j) is true if the tweet i is in the topic j otherwise it's false
    shape of the matrix is (number of tweets, number of topics)
    Parameters
    ----------
    model : 
        NMF or LDA model.
    tf : scipy.sparse.csr.csr_matrix
        matrix storing tf of the our data
    save : bool
        if save is true we save results in a file name.csv
    name : str, optional
        the name of the file where we save the results. The default is "clusters".

    Returns
    -------
    T : numpy array.

    """
    m = model.transform(tf)
    maxx = np.max(m, axis=1)
    T = m==maxx[:,None]
    
    for i in range(len(T)):
        if np.sum(T[i])>1 :
            T[i] = [0]*len(T[0])
            
    if save:
        np.savetxt("results/" + name+".csv", T, delimiter=",")
        
    return T
    
def get_topics(df,clusters,n,save):
    """
    This function saves the tweets of each topic in separate files 

    Parameters
    ----------
    df : pandas.core.frame.DataFrame
         data not cleaned prefered.
    clusters : numpy array
        numpy array results of the function get_clusters().
    n : int
        number of topics.
    save : bool
        if save is true we save results in a files

    Returns
    -------
    l : TYPE
        DESCRIPTION.

    """
    clusters = pd.DataFrame(clusters,columns = ['Topic_'+str(i) for i in range(n)])
    l = []
    for i in range(n):
        l.append(df[clusters['Topic_'+str(i)] ==1])
        if save:
            l[i].to_csv('results/Topic_'+str(i)+".csv")
    
    l.append(df[np.sum(clusters.iloc[:],axis = 1) == 0])
    l[-1].to_csv("results/Non_classified.csv")




# get topics with their terms and weights
def get_topics_terms_weights(weights, feature_names):
    feature_names = np.array(feature_names)
    sorted_indices = np.array([list(row[::-1]) for row in np.argsort(np.abs(weights))])
    sorted_weights = np.array([list(wt[index]) for wt, index in zip(weights, sorted_indices)])
    sorted_terms = np.array([list(feature_names[row]) for row in sorted_indices])

    topics = [np.vstack((terms.T, term_weights.T)).T for terms, term_weights in zip(sorted_terms, sorted_weights)]

    return topics


# prints components of all the topics
# obtained from topic modeling
def print_topics_udf(topics, total_topics=1,
                     weight_threshold=0.0001,
                     display_weights=False,
                     num_terms=None):

    for index in range(total_topics):
        topic = topics[index]
        topic = [(term, float(wt))
                 for term, wt in topic]
        #print(topic)
        topic = [(word, round(wt,2))
                 for word, wt in topic
                 if abs(wt) >= weight_threshold]

        if display_weights:
            print('Topic #'+str(index)+' with weights')
            print(topic[:num_terms]) if num_terms else topic
        else:
            print('Topic #'+str(index)+' without weights')
            tw = [term for term, wt in topic]
            print(tw[:num_terms]) if num_terms else tw

# prints components of all the topics
# obtained from topic modeling
def get_topics_udf(topics, total_topics=1,
                     weight_threshold=0.0001,
                     num_terms=None):

    topic_terms = []

    for index in range(total_topics):
        topic = topics[index]
        topic = [(term, float(wt))
                 for term, wt in topic]
        #print(topic)
        topic = [(word, round(wt,2))
                 for word, wt in topic
                 if abs(wt) >= weight_threshold]

        topic_terms.append(topic[:num_terms] if num_terms else topic)

    return topic_terms


#Load the csv file containing the different countries :
def load_tweets(no_topics,path_to_folder="results/"):
  #num_top is the number of topics
  #path_to_folder : the path to the folder containing the different tweets of each topic
  #returns a list where each element is a dataframe containing the tweets of a certain topic
  #to_countries is a pandas.core.frame.DataFrame : Each element
  Lists_Topics = [[] for i in range(no_topics)]
  for i in range (no_topics):
    Lists_Topics[i] = pd.read_csv(path_to_folder+"Topic_"+str(i)+".csv")[['text']]
  return Lists_Topics


#find the different distinations mentionned in the different topics
def find_destinations (Lists_Topics,countries):
    #Lists_Topics is a list : each element is a dataframe containing the tweets of a certain topic
    #to_countries is a pandas.core.frame.DataFrame : Each element
    #of the column "Name" is a 'to + country'

    #returns a list of lists : each element(list) contains tuples (a,b) where a is the number of 
    #occurences of b ( to + country) in the topic corresponding to the elemnt 
    
    destinations = [[] for i in range(len(Lists_Topics))]
    # We only need the column "Name" containing the names of the countries 
    countries = countries[['Name']]
    to_countries = countries.Name 
    for e in to_countries:
        for i in range(len(Lists_Topics)):
              if e == ' to United States' : 
                  destinations[i].append((len(Lists_Topics[i][Lists_Topics[i]['text'].str.contains('to the united states| to the usa')]),e))
              elif e == ' to United Kingdom' :
                  destinations[i].append((len(Lists_Topics[i][Lists_Topics[i]['text'].str.contains('to the united kingdom| to the uk')]),e))

              else :
                  destinations[i].append((len(Lists_Topics[i][Lists_Topics[i]['text'].str.contains(e.lower())]),e))
    return destinations


#Plots a histogram showing the top destinations in each topic
def plot_destinations(destinations):
      #destinations is  a list of lists : each element(list) contains tuples (a,b) where a is the number of 
      #occurences of b ( to + country) in the topic corresponding to the elemnt 
      counters = [{} for i in range(len(destinations))]
      for i in range (len(destinations)) :
          for j in range (len(destinations[i])):
               counters[i] = Counter({destinations[i][j][1]:destinations[i][j][0] for j in range(len(destinations[i]))})
      for elt in counters :    
        Top_destinations=dict(itertools.islice(dict(elt.most_common()[:9]).items(),10))
        Top_destinations_df = pd.DataFrame.from_dict(Top_destinations, orient='index')
        Top_destinations_df.plot(kind='bar')
        

from os import listdir
from os.path import isfile, join
# fusion multiple csv files
def fusion(path,title):

    files = [f for f in listdir(path) if isfile(join(path, f))]
    
    data = pd.DataFrame()
    for file in files:
        data = pd.concat([data, dataframe_from_file(path + file)])
    
    data.to_csv("data/" + title + ".csv")
    return data
    

if __name__ == "__main__":
    #fusion data
    path = "../data/"
    title = "data"
    
    df = fusion(path,title)
    
    # reading the data
        #df = dataframe_from_file("stream_tweets_Week4_181k.csv") # put the path corresponding to the file
        #df = pd.read_csv('results/Topic 0_4.csv')
    col = 'text' # the column where tweets are sotored
    col_processed = col+"_processed"
    
    # Cleaning the data
    df = cleanData(df, col)
    
    # data processed (lemmatize_stemming)
    df_processed = df.copy()
    df_processed[col_processed] = df_processed[col].apply(preprocess)
    
    # Visualise the most common words
    wordBarGraphFunction(df, col)
    wordcloud_data(df, col)
    
    # paramerters
    no_topics = 5
    no_features = 1000
    no_top_words = 10
    save = True
    model = "NMF"
    
    
    
    if model == "NMF":
        # display topics
        print("TOPICS FOUND USING NMF")
        Nmf_model, Nmf_model_fitted, tfidf_feature_names, tfidf = nmf = NMF_model(df_processed, col_processed, no_features)
        display_topics(Nmf_model_fitted, tfidf_feature_names, no_top_words)
        clusters = get_clusters(Nmf_model_fitted, tfidf,save)
        get_topics(df, clusters,no_topics, save)
        nmf_weights = Nmf_model.components_
        topics = get_topics_terms_weights(nmf_weights, tfidf_feature_names)
        print_topics_udf(topics, total_topics=no_topics, num_terms=30, display_weights=True)
        topics_display_list = get_topics_udf(topics, total_topics=2, num_terms=30)
        
    
    elif model == "LDA":
        print("TOPICS FOUND USING LDA")
        Lda_model_fitted, tf_feature_names, tf = LDA_model(df, col_processed, no_features)
        topic_of_tweets = np.argmax(Lda_model_fitted.transform(tf), axis=1)
        display_topics(Lda_model_fitted, tf_feature_names, no_top_words)
        
        pyLDAvis.sklearn.prepare(lda, tf, tf_vectorizer)

    
    
    #destinations :
    ##Load the csv file containing the different countries :
    countries = pd.read_csv("data_csv.csv")# put the path corresponding to the file
    
    ## TO + Country gives more insights about the destinations 
    countries['Name'] = 'to ' + countries['Name'] 
    
    ##Each element of Lists_Topics is a dataframe containing a given topic
    Lists_Topics = load_tweets(no_topics)
    
    ##List of lists : each element(list) contains tuples (a,b) where a is the number of occurences of b ( to + country) in the topic corresponding to the elemnt
    destinations = find_destinations(Lists_Topics,countries)
    
    ##Plotting the top destinations in each topic
    plot_destinations(destinations)
    

    
    
    