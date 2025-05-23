import numpy as np
from basic import flip_dict
import time as ti
from tqdm import tqdm
from collections import defaultdict
import networkx as nx
from math import exp, log
from joblib import Parallel, delayed

class Retriever:
    def __init__(self, gamma,
                 test, all_facts, 
                 entities, relations, times_id, 
                 num_relations, chains, rel_keys, dataset, retrieve_type='weighted'):
        self.gamma = gamma

        self.retrieve_type = retrieve_type
        self.dataset = dataset
        self.test = test
        self.all_facts = all_facts
        
        self.entities = entities
        self.relations = relations
        self.times_id = times_id
        self.num_relations = num_relations
        self.chains = chains
        
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
    
    def create_time_index(self, current_time):
        # Store the number of connections (s, o) before current_time
        self.time_index = defaultdict(int)
        for sub, obj, time in zip(self.col_sub, self.col_obj, self.col_time):
            if self.times_id[time] < current_time:
                self.time_index[(sub, obj)] += 1

    def create_graph(self,current_time):
        # Create the graph structure and calculate the number of hops
        self.graph = nx.Graph()
        # edges = [(sub, obj) for sub, obj in zip(self.col_sub, self.col_obj)]
        edges = [(sub, obj) for sub, obj, time in zip(self.col_sub, self.col_obj, self.col_time) if self.times_id[time] <= current_time]
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

    def connection_counts_before_T(self, sub, obj):
        # Return the number of connections between sub and obj before T
        return self.time_index.get((sub, obj), 0) + self.time_index.get((obj, sub), 0)
    
    def frequency_counts_before_T(self, current_time):
        triple_keys = [(sub, rel, obj) for sub, rel, obj, time in zip(self.col_sub, self.col_rel, self.col_obj, self.col_time) if self.times_id[time] < current_time]
        unique, counts = np.unique(triple_keys, return_counts=True, axis=0)
        self.triple_to_count = {tuple(k): v for k, v in zip(unique, counts)}

    
    def neighbor_weight(self, hop_s, hop_o):
        g1 = self.gamma[0]
        return exp(-g1 * (hop_s + hop_o - 1))
    
    def frequency_weight(self, i):
        g2 = self.gamma[1]
        key = (self.col_sub[i], self.col_rel[i], self.col_obj[i])
        ns = self.triple_to_count.get(key,1)
        return 1 / (g2 * np.log(ns) + 1)
    
    def time_weight(self, event_time:str, current_time:int):
        g3 = self.gamma[2]
        return exp(-g3 * ((current_time - self.times_id[event_time])//24))

    def connection_weight(self, count_so):
        g4 = self.gamma[3]
        return log(1 + g4 * count_so) / (1 + log(1 + g4 * count_so))
    


    def total_weight(self, i, sub, obj, event_time, current_time, hop_dict, vip_neighbors):
        hop_s = hop_dict.get(sub, np.inf)
        hop_o = hop_dict.get(obj, np.inf)

        if np.isinf(hop_s) or np.isinf(hop_o):
            return 0

        tw = self.time_weight(event_time, current_time)
        nw = self.neighbor_weight(hop_s, hop_o)
        fw = self.frequency_weight(i)
        cw = self.connection_weight(self.connection_counts_before_T(sub, obj))
        vw = 1 if sub in vip_neighbors or obj in vip_neighbors else 0

        return nw * fw * (tw + cw + vw)

    def sample_ids_by_top_weights(self, total_weight, sample_size, replace=False, seed=42, top_multiplier=10):
        total_weight = np.array(total_weight)

        if seed is not None:
            np.random.seed(seed)

        # Step 1: Retrieve the indices of the top_k largest weights
        top_k = min(len(total_weight), top_multiplier * sample_size)
        top_indices = np.argpartition(-total_weight, top_k - 1)[:top_k]

        # Step 2: Extract the corresponding weights
        top_weights = total_weight[top_indices]

        # Step 3: Construct normalized sampling probabilities
        probs = top_weights / np.sum(top_weights)
        actual_size = min(sample_size, np.count_nonzero(probs))
        if actual_size == 0:
            print("No non-zero weights available. Returning empty list.")
            return []

        # Step 4: Perform weighted sampling from top_indices
        sampled_sub_indices = np.random.choice(
            len(top_indices),
            size=actual_size,
            replace=replace,
            p=probs
        )

        # Step 5: Map back to the original indices of total_weight
        sampled_ids = top_indices[sampled_sub_indices]

        return sampled_ids.tolist()

    def prepare_comp(self, i):
        test_sub, test_rel, _, test_time, _ = self.test[i].strip().split("\t")
        #First of all, there must be a time premise of retrieve s_t
        #Here we need to find out the idx of test in all_facts so that it can be removed
        idx_test = len(self.all_facts)- (len(self.test)-1) + i -1
        # #The major premise is that retrieval must be performed from those ranges earlier than test_time
        idx_t = np.where(self.col_time < test_time)[0]
        s_t = set(idx_t)
        if idx_test in s_t:
                s_t.remove(idx_test) #Remove the test item itself
        #Second, the search for the beginning of the chain needs to be restricted to rel==test_sub
        idx_test_sub = np.where(self.col_sub == test_sub)[0]
        s_test_sub = set(idx_test_sub)
        s_0 = s_t & s_test_sub #Get: major premise
        head_rel = self.relations[test_rel] #Get the idx corresponding to test_relation: 0,1,2,...
        time = self.times_id[test_time] #To move forward the time according to the id corresponding relationship of time. Get int from str
        return s_0, s_t, head_rel, time, test_sub, test_rel

    def one_case_weighted(self, i):
        num_facts = 50
        s_0, s_t, head_rel, current_time, test_sub, test_rel = self.prepare_comp(i)
        facts = []
        idx = []

        if not str(head_rel) in self.chains:
            history_query = [str(current_time)+': ['+ test_sub +', '+ test_rel+',\n']
            return [], history_query

        s_0 = np.array(list(s_0))
        for k in range(len(self.chains[str(head_rel)])):
            idx_case = []
            body_rel_len = len(self.chains[str(head_rel)][k]['body_rels'])
            if body_rel_len != 1:
                continue
            rel = self.chains[str(head_rel)][k]['body_rels'][-1] % self.num_relations
            idx_rel = np.where(self.col_rel == self.rel_keys[rel])[0]
            idx_rel = np.intersect1d(idx_rel, s_0)
            idx_case = idx_rel.tolist()
            if len(idx_case) != 0:
                idx = list(set(idx + idx_case))
            if len(idx) >= num_facts:
                break

        self.frequency_counts_before_T(current_time)

        if len(idx) < num_facts:
            vip_neighbors = self.col_obj[np.array(idx)].tolist() if len(idx) > 0 else []
            self.create_time_index(current_time)
            self.create_graph(current_time)
            hop_dict = self.entity_hop_numbers(test_sub)
            if hop_dict:
                weights = []
                for sub, obj, time in zip(self.col_sub, self.col_obj, self.col_time):
                    if self.times_id[time] < current_time and sub != test_sub:
                        weight = self.total_weight(i, sub, obj, time, current_time, hop_dict, vip_neighbors)
                        weights.append(weight)
                    else:
                        weights.append(0)
                resid = num_facts - len(idx)
                if np.sum(weights) != 0:
                    sampled_ids = self.sample_ids_by_top_weights(weights, resid)
                    idx = list(set(idx) | set(sampled_ids))

        idx.sort(reverse=True)
        idx = idx[0:num_facts]
        for a in idx:
            facts.append(self.all_facts[a])
        if len(facts) == 0:
            history_query = self.build_history_query(current_time, test_sub, test_rel)
            return idx, history_query
        if num_facts >= len(facts):
            num_facts = len(facts)
        histories = self.collect_hist(i, facts, num_facts)
        history_query = self.build_history_query(current_time, test_sub, test_rel, histories=histories)
        return idx, history_query

    def build_weighted(self, num_processes=100):
        print("Starting to build weighted retrieval...")
        results = Parallel(n_jobs=num_processes)(
            delayed(self.one_case_weighted)(i) for i in tqdm(range(len(self.test)))
        )

        test_idx, test_text = zip(*results)
        return list(test_idx), list(test_text)

    def tlogic_prepro(self, i):
        test_sub, test_rel, _, test_time, _ = self.test[i].strip().split("\t")
        #First of all, there must be a time premise of retrieve s_t
        #Here we need to find out the idx of test in all_facts so that it can be removed
        idx_test = len(self.all_facts)- (len(self.test)-1) + i -1
        # #The major premise is that retrieval must be performed from those ranges earlier than test_time
        idx_t = np.where(self.col_time < test_time)[0] 
        s_t = set(idx_t)
        if idx_test in s_t:
                s_t.remove(idx_test) #Remove the test item itself
                #Second, the search for the beginning of the chain needs to be restricted to rel==test_sub
        idx_test_sub = np.where(self.col_sub == test_sub)[0]
        s_test_sub = set(idx_test_sub)
        s_0 = s_t & s_test_sub #Get: major premise
        head_rel = self.relations[test_rel] #Get the idx corresponding to test_relation: 0,1,2,...
        time = self.times_id[test_time] #To move forward the time according to the id corresponding relationship of time. Get int from str
        return s_0, s_t, head_rel, time, test_sub, test_rel
    
    def one_case_TLR(self,i):
        num_facts = 50 #Set it again for each question, as the following may change.
        s_0, s_t, head_rel, time, test_sub, test_rel = self.tlogic_prepro(i)
        facts = []
        idx = []

        if not str(head_rel) in self.chains: #If the test relation has no chain, nothing will be done.
            #l = ['Just repeat "No Chains."\n']
            history_query = [str(time)+': ['+ test_sub +', '+ test_rel+',\n']
            return [], history_query
        s_0 = np.array(list(s_0))  #Convert collection to NumPy array
        #After the above preparations, start searching for facts by chain.
        idx_chain = []
        for k in range( 0,len(self.chains[str(head_rel)]) ): #There are len(chains[str(head_rel)]) chains
            body_rel_len = len( self.chains[str(head_rel)][k]['body_rels'] ) #The length of this chain
            if body_rel_len==1:
                idx_chain.append(k)
        for k in idx_chain: #TLogic (or TLogic-3), as long as the shortest len=1 chain
            idx_case = []
            body_rel_len = len( self.chains[str(head_rel)][k]['body_rels'] ) #The length of this chain
            rel = self.chains[str(head_rel)][k]['body_rels'][-1] %self.num_relations 
            idx_rel = np.where(self.col_rel == self.rel_keys[rel])[0]
            idx_rel = np.intersect1d(idx_rel, s_0)  #Using NumPy functions for intersection operations
            idx_case = idx_rel
            idx_case.tolist()
            if len(idx_case) != 0: #If it is not empty, retrieve it.
                idx_case = list( set(idx_case) )
                idx = list( set( idx + idx_case) )
            else: 
                continue #If no such chain exists, jump to the next chain
            if len(idx)>=num_facts: 
                break #Break out of the loop on chains and go to the next test
        #time reordering
        idx.sort(reverse=True)
        #Idx with chain.sort(reverse=true)
        if len(idx)>num_facts:
            idx = idx[0:num_facts] 
            for a in idx:         
                facts.append(self.all_facts[a])
        else: 
            for a in idx:         
                facts.append(self.all_facts[a])
        if len(facts)==0: #If the test relation has no chain, nothing will be done.
            history_query = self.build_history_query(time, test_sub, test_rel) #Self.times id[time]
            return idx, history_query
        if num_facts >= len(facts):
            num_facts = len(facts)
        histories = self.collect_hist(i, facts, num_facts)
        history_query = self.build_history_query(time, test_sub, test_rel, histories=histories)
        return idx, history_query

    def build_TLR(self, num_processes=100):
        print("Starting to build TLR retrieval...")
        results = Parallel(n_jobs=num_processes)(
            delayed(self.one_case_TLR)(i) for i in tqdm(range(len(self.test)))
        )

        test_idx, test_text = zip(*results)
        return list(test_idx), list(test_text)

    def one_case_ISI(self,i):
        num_facts = 50 #Set it again for each question, as the following may change.
        s_0, s_t, head_rel, time, test_sub, test_rel = self.tlogic_prepro(i)
        facts = []
        idx = []

        #time reordering
        idx = list(s_0)
        idx.sort(reverse=True)
        #Idx with chain.sort(reverse=true)
        if len(idx)>num_facts:
            idx = idx[0:num_facts] 
            for a in idx:         
                facts.append(self.all_facts[a])
        else: 
            for a in idx:         
                facts.append(self.all_facts[a])
        if len(facts)==0: #If the test relation has no chain, nothing will be done.
            history_query = self.build_history_query(time, test_sub, test_rel) #Self.times id[time]
            return idx, history_query
        if num_facts >= len(facts):
            num_facts = len(facts)
        histories = self.collect_hist(i, facts, num_facts)
        history_query = self.build_history_query(time, test_sub, test_rel, histories=histories)
        return idx, history_query

    def build_ISI(self, num_processes=100):
        print("Starting to build ISI retrieval...")
        results = Parallel(n_jobs=num_processes)(
            delayed(self.one_case_ISI)(i) for i in tqdm(range(len(self.test)))
        )
        test_idx, test_text = zip(*results)
        return list(test_idx), list(test_text)
    
    def collect_hist(self, i, facts, num_facts):
        period = 1
        if self.dataset == "icews14" or self.dataset == "icews18":
            period = 24
        histories = []
        facts = facts[0:num_facts] # 
        facts.reverse() #Replace the order so that the last output is the one closest in time.
        for b in range(num_facts): 
            fact = facts[b].strip().split('\t')
            time_in_id = self.times_id[fact[3]]
            sub_in_word = fact[0]
            rel_in_word = fact[1]
            
            obj_in_word = fact[2]
            id_obj = self.entities[obj_in_word]
            histories= histories+ [str(int(int(time_in_id)/int(period)))
                    +': [' + sub_in_word +', ' + rel_in_word +', ' 
                    + str(id_obj)+'.'+ obj_in_word +'] \n']
        return histories

    def build_history_query(self, time, test_sub, test_rel, histories=''):
        # print(time, test_sub, test_rel)
        period = 1
        if self.dataset == "icews14" or self.dataset == "icews18":
            period = 24
        # time_in_id = self.times_id[time]
        time_in_id = time
        return [''.join(histories)  + str(int(int(time_in_id)/int(period)))+': ['+ test_sub +', '+ test_rel+',\n'#times id[time]
                ]
    
    def call_function(self, func_name):
        func = getattr(self, func_name)
        if func and callable(func):
            test_idx, test_text = func()
        else:
            print("Retrieve function not found")
        return test_idx, test_text
    
    def get_output(self):
        if 'rbmh' in self.retrieve_type:
            type_retr = "weighted"
        elif 'tlr' in self.retrieve_type:
            type_retr = "TLR"
        elif 'isi' in self.retrieve_type:
            type_retr = "ISI"
        test_idx, test_text = self.call_function("build_"+type_retr)
        
        return test_idx, test_text
