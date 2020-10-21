'''
Created on 16 de jan de 2019
@author: gusta
'''


import os
from streams.generators.__init__ import *
from streams.generators.virtual_stream import VirtualStream

def caminho(nome, i):
    stream_name = nome
    project_path = "../../data_streams/_synthetic/" + stream_name + "/"
    if not os.path.exists(project_path):
        os.makedirs(project_path)
    file_path = project_path + stream_name + "_"+ str(i)
    
    return file_path

for i in range(0, 31):

    file_path = caminho('circles', i)
    stream_generator = CIRCLES(concept_length=2000, random_seed=i)
    stream_generator.generate(file_path)
    
    file_path = caminho('sine1', i)
    stream_generator = SINE1(concept_length=2000, random_seed=i)
    stream_generator.generate(file_path)
    
    file_path = caminho('sine2', i)
    stream_generator = SINE2(concept_length=2000, random_seed=i)
    stream_generator.generate(file_path)
    
    file_path = caminho('SEA', i)
    stream_generator = SEA(concept_length=2000, thresholds=[1, 9, 2, 6], random_seed=i)
    stream_generator.generate(file_path)
    
    file_path = caminho('SEARec', i)
    stream_generator = SEA(concept_length=2000, thresholds=[1, 9, 2, 6, 1, 9, 2, 6], random_seed=i)
    stream_generator.generate(file_path)
    
    vt = VirtualStream()
    vt.generate_stream_5changes(4, i)
    vt.write_to_arff('../data_streams/_synthetic/virtual_5changes/')
        
    vt = VirtualStream()
    vt.generate_stream_9changes(4, i)
    vt.write_to_arff('../data_streams/_synthetic/virtual_9changes/')
