import pandas as pd
import csv


def decode_utf8(s):
    if s: return eval(s).decode('utf-8')
    else: return s

def decode_strlist_utf8(lst):
    return [s.decode('utf-8') for s in eval(lst)]

def dataframe_from_file(filename):
    with open(filename, 'r') as f:
        df = pd.read_csv(f, error_bad_lines=False, converters={
            'text': decode_utf8,
            'country': decode_utf8,
            'user_screen_name': decode_utf8,
            'hashtags': decode_strlist_utf8,
            'mentions': decode_strlist_utf8
        })
    return df

def dataframe_from_mult_files(filenames):
    """@param filenames (List[Str]): list of filenames"""
    
    dfs = []
    for filename in filenames:
        dfs.append(dataframe_from_file(filename))
    return pd.concat(dfs, axis=0)