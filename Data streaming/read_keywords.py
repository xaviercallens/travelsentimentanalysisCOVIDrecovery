import codecs

def get_keywords(filename):
    """Read the keywords contained in the file and put them
    in a list for Twitter stream filter.
    Each line in the file must be one phrase of keywords or
    an empty line.

    @ param filename (Str): txt file name
    """

    keywords = []
    with codecs.open(filename, encoding='utf-8') as f:
        for line in f.readlines():
            words = ' '.join(line.split())
            if words is not '':
                keywords.append(words)
    return keywords