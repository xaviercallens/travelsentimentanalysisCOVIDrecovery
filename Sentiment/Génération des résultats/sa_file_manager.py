import pandas as pd

def load_data(filename):
    return pd.read_csv(filename, index_col=0, squeeze=True)

def get_sa_filename(folder, tweets_filename, model_name, analysis_name):
    return folder + f'{tweets_filename} - {model_name} - {analysis_name}.csv'

def save_sa(graph, folder, tweets_filename, model_name, analysis_name):
    '''Save a sentiment analysis result (pandas series or dataframe) in the file named with get_sa_filename function.'''
    graph.to_csv(get_sa_filename(folder, tweets_filename, model_name, analysis_name))

def load_sa(folder, tweets_filename, model_name, analysis_name):
    '''Load the file containing the pandas series or dataframe named with get_sa_filename function.'''
    return load_data(get_sa_filename(folder, tweets_filename, model_name, analysis_name))

def load_sa_from_models_tweets(folder, models_names, tweets_filenames, analysis_name, concatenate):
    '''Load multiple files each containing one pandas series or dataframe associated to one model and one tweet file.
    Those series/dataframe are then concatenated along the tweets files for each model.

    @param concatenate: a function concatenating a list of series or dataframes.
    The concatenation depends on the conducted analysis.'''
    
    dict_sa = {}
    for model_name in models_names:
        data_list = [load_sa(folder, tw_f, model_name, analysis_name) for tw_f in tweets_filenames]
        dict_sa[model_name] = concatenate(data_list)
    return dict_sa

def get_sa_param_filename(param):
    '''Return the string that must be added at the end of a sa filename to specify the parameters of the sa.
    @param (dict): map parameter's names to their corresponding values (only string)'''
    str_param = ' - '.join(f'{k}={v}' for k, v in param.items())
    return f' ({str_param})'