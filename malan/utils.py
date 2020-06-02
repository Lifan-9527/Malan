import os

def path_to_list(path):
    filenames = []
    for r,d,f in os.walk(path):
        for x in f:
            a_file = None
            if 'context' in x:
                a_file = r+'/'+x
                filenames.append(a_file)
    return filenames