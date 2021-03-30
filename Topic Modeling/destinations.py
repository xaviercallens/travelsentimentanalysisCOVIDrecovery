import os 
import pandas as pd
from collections import Counter
import itertools


#Load the csv file containing the different countries :
def load_tweets(num_top,path_to_folder):
  #num_top is the number of topics
  #path_to_folder : the path to the folder containing the different tweets of each topic
  #returns a list where each element is a dataframe containing the tweets of a certain topic
  #to_countries is a pandas.core.frame.DataFrame : Each element
  Lists_Topics = [[] for i in range(num_top)]
  for i in range (num_top):
    Lists_Topics[i] = pd.read_csv(path_to_folder+"Topic_"+str(i)+"_allweeks.csv")[['text']]
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
        


if __name__ == "__main__":
    # paramerters
    num_top = 5
    path_to_folder = "Desktop\\"
    #Load the csv file containing the different countries :
    countries = pd.read_csv("Desktop\\data_csv.csv")# put the path corresponding to the file
    
    # TO + Country gives more insights about the destinations 
    countries['Name'] = 'to ' + countries['Name'] 
    
    #Each element of Lists_Topics is a dataframe containing a given topic
    Lists_Topics = load_tweets(num_top,path_to_folder)
    
    #List of lists : each element(list) contains tuples (a,b) where a is the number of occurences of b ( to + country) in the topic corresponding to the elemnt
    destinations = find_destinations(Lists_Topics,countries)
    
    #Plotting the top destinations in each topic
    plot_destinations(destinations)
  
 
