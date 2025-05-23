import numpy as np
from basic import flip_dict
import time as ti
from tqdm import tqdm
from collections import defaultdict
import networkx as nx
from basic import read_txt_as_list, read_json, write_txt, write_txt_ans
from id_words import convert_dataset
import os, glob
import argparse
from concurrent.futures import ProcessPoolExecutor


class Graph_tkg:
    def __init__(self, 
                 test, all_facts, 
                 entities, relations, times_id, 
                 num_relations, rel_keys, dataset, retrieve_type='weighted',):
        self.retrieve_type = retrieve_type
        self.dataset = dataset
        self.test = test
        self.all_facts = all_facts
        
        self.entities = entities
        self.relations = relations
        self.times_id = times_id
        self.num_relations = num_relations
        
        self.entities_flip = flip_dict(self.entities) # id2word
        self.relations_flip = flip_dict(self.relations)
        col_sub = []
        col_rel = []
        col_obj = []
        col_time = []
        for row in all_facts:
            row = row.strip().split('\t')
            col_sub.append(row[0]) #take sub
            col_rel.append(row[1]) #Get the relation column of all facts
            col_obj.append(row[2]) #Take obj
            col_time.append(row[3]) #Time in Str form
        self.col_obj = np.array(col_obj) #To get cand and then search for facts use
        self.col_sub = np.array(col_sub) #To get cand and then search for facts use
        self.col_time = np.array(col_time) #time, str form
        self.col_rel = np.array(col_rel) #relation, str form
        all_facts_array = np.array(all_facts)
        
        self.rel_keys = np.array(rel_keys)

    def create_graph(self,current_time):
        # Create the graph structure and calculate the number of hops
        self.graph = nx.Graph()
        # edges = [(sub, obj) for sub, obj in zip(self.col_sub, self.col_obj)]
        edges = [(sub, obj) for sub, obj, time in zip(self.col_sub, self.col_obj, self.col_time) if self.times_id[time] < current_time]
        self.graph.add_edges_from(edges)
    
    def entity_hop_numbers(self, query_subject):
        # Calculate the hop count from all entities to query_subject
        # lengths = nx.single_source_shortest_path_length(self.graph, query_subject)
        try:
            lengths = nx.single_source_shortest_path_length(self.graph, query_subject)
            # print("Have Neighbor")
        except:
            # print("No neighbor")
            lengths = {}
        return lengths
    
    def hops(self):
        results = defaultdict(int)
        hop_indices = defaultdict(list)  # Used to record the index list corresponding to each hop

        for i in tqdm(range(0, len(self.test))):
        # for i in tqdm(range(0, 50)):
            test_sub, test_rel, test_obj, test_time, _ = self.test[i].strip().split("\t")
            current_time = self.times_id[test_time]
            self.create_graph(current_time)
            hop_dict = self.entity_hop_numbers(test_sub)
            hop = hop_dict.get(test_obj, np.inf)
            results[hop] += 1
            hop_indices[hop].append(i)  # Record the index

        return results, hop_indices



def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", default="icews14", type=str)
    parser.add_argument("--retrieve_type", "-t", default="weighted", type=str)
    parser.add_argument("--name_of_rules_file", "-r", default="", type=str)
    parsed = vars(parser.parse_args())
    return parsed
    
if __name__ == "__main__":
    parsed = parser()
    retrieve_type = parsed["retrieve_type"]
    type_dataset = parsed["dataset"]
    name_rules = parsed["name_of_rules_file"]
    
    # path_workspace = "./data/"+type_dataset+"/" #Icew s14 /icews14
    path_workspace = "./data/original/"+type_dataset+"/" #Icew s14 /icews14
    path_out_tl = "./data_utils/rules_output/"+type_dataset+"/"
    print(path_out_tl)
    
    path_save = "./data/processed/"+type_dataset+"/"
    if not os.path.exists(path_save):
            os.makedirs(path_save)
        
    period = 1
    if type_dataset == "icews18":
        num_relations = 256 #for ICEWS18 #set before np.array
        period = 24
    elif type_dataset == "icews14":
        num_relations = 230
        period = 24
    elif type_dataset == "icews0515":
        num_relations = 251
    elif type_dataset == "GDELT":
        num_relations = 238 #GDELT and
    else:
        num_relations = 24 # YAGO
        
    test_ans = []
    
    #open files:
        
    li_files = ['test']#  ['train','valid','test'] or  ['test'] when only test set is needed
    
    for files in li_files:
        test_ans = read_txt_as_list(path_workspace+files+'.txt')
        
        relations = read_json(path_workspace+'relation2id.json')
        entities = read_json(path_workspace+'entity2id.json')
        times_id = read_json(path_workspace+'ts2id.json')
        test_ans = convert_dataset(test_ans, path_workspace, period=1)
            
        rel_keys = list(relations.keys())
        # ent_idx = list(entities.keys()) # [0, 1, ...]
        # times_id_keys = list(times_id.keys())
        all_facts = []
        with open(path_workspace+"all_facts.txt", "r", encoding='utf-8') as f:
            all_facts = f.readlines()
        graph1 = Graph_tkg(test_ans, all_facts, entities, relations, times_id, num_relations, rel_keys, dataset=type_dataset, retrieve_type=retrieve_type)
        print(graph1.hops())

