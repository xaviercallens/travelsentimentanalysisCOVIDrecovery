# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 17:03:24 2021

@author: Clement_Bureau
"""


# Import (l'import est long, vous pouvez le faire 1 seule fois au lancement de python)
import Load_Model


# Chargement d'un modele :
Load_Model.charge_model('model_Tweets_Flights')

# Test d'une phrase
sentiment=Load_Model.evaluate_sentiment(['the flight was the best','the flight is bad'])
print(sentiment)

