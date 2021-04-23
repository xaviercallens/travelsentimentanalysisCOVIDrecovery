Les notebooks "Download ..." téléchargent des fichiers stockés dans Google Cloud Storage.
Les tweets sont téléchargés à partir du dossier "Collected tweets/".
"model_Tweets_Flights/" contient un modèle LSTM de Tensorflow pour l'analyse de sentiment.

## Data streaming

La récolte de tweets se fait en exécutant le fichier configuration.py. Les différents paramètres de streaming peuvent être modifiés dans la classe Configuration.

## Analyse de sentiment

L'analyse de sentiment est effectué dans le dossier "Sentiment/Génération des résultats/". Il y a 3 modèles, chacun donnant son nom à un notebook du dossier : Multinomial Naive Bayes, TextBlob et LSTM1. Un notebook d'un modèle prédit les sentiments de chaque tweets stockés dans le fichier spécifié (le nom est à spécifier dans le notebook). Ces sentiments sont sauvegardés dans un fichier sous forme d'un tableau numpy.

Le notebook "Generate sentiment analysis" générent différents résultats d'analyse de sentiment à partir des sentiments sauvegardés dans les fichiers. Différents paramètres sont à spécifier dans ce notebook (si besoin, des informations sur ces paramètres sont disponibles dans les fichiers contenant les fonctions et classes concernées). Les différents types de résultat sont : la distribution des sentiments, l'évolution du sentiment au cours du temps, l'évolution du nombre de tweets au cours du temps, et des échantillons de tweets pour chaque sentiment. Chacun de ces résultats sont sauvegardés dans des fichiers csv. Ces fichiers pourront ensuite être ouverts pour afficher les résultats sur un dashboard.

Le notebook "View sentiment analysis" permet de visualiser les résultats de l'analyse de sentiment sauvegardés dans les fichiers, sans passer par un dashboard. Différents paramètres sont là aussi à spécifier.
