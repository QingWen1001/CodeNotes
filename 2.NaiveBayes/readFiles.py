import os
import pandas as pd
def get_filename_list(path):
    '''

    :param path: type:string : The path of document where data files stale
    :return: type:list : The list of filename
    '''
    filename_list = os.listdir(path)
    return filename_list
def read_file(path):
    with open(path) as f:
        content = f.read()
    return content
def readline(path):
    content = []
    for i in open(path):
        content.append(i.strip())
    return content
def read_csv_file(path):
    content = pd.read_csv(path)
    return content
def get_data_from_document(path):
    '''

    :param path: type:string : The path of document where data files stale
    :return: type:list : The list of data
    '''
    filename_list = get_filename_list(path)
    text = []
    for filename in filename_list:
        file_path = path +"/" + filename
        content = read_file(file_path)
        text.append(content)
    return text
