import os

def path_to_list(path, key_word=None):
    filenames = []
    for r,d,f in os.walk(path):
        for x in f:
            a_file = None
            if key_word != None:
                if key_word not in x:
                    continue
            a_file = r+'/'+x
            filenames.append(a_file)
    return filenames