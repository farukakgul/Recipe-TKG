from weighted_sampling import Retriever
from basic import read_txt_as_list, read_json, write_txt, write_txt_ans
from id_words import convert_dataset
import os, glob
import argparse

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", default="icews14", type=str)
    parser.add_argument("--retrieve_type", "-t", default="weighted", type=str)
    parser.add_argument("--name_of_rules_file", "-r", default="", type=str)

    parser.add_argument("--g1", type=float, default=0.6)
    parser.add_argument("--g2", type=float, default=0.6)
    parser.add_argument("--g3", type=float, default=0.01)
    parser.add_argument("--g4", type=float, default=0.1)

    parser.add_argument("--sensi_exp", default=False, action="store_true")
    parsed = vars(parser.parse_args())
    return parsed
    
if __name__ == "__main__":
    parsed = parser()
    retrieve_type = parsed["retrieve_type"]
    type_dataset = parsed["dataset"]
    name_rules = parsed["name_of_rules_file"]

    path_workspace = "./data/original/"+type_dataset+"/"
    path_out_tl = "./data_utils/rules_output/"+type_dataset+"/"
    print(path_out_tl)
    
    path_save = "./data/processed/"
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
        
    li_files = ['test','valid','train']# ['test'] when only test set is needed

    if parsed["sensi_exp"]:
        li_files = ['test']
    
    for files in li_files:
        print("exiting rules:", glob.glob(path_out_tl+'*rules.json'))
        print("files", files)
        test_ans = read_txt_as_list(path_workspace+files+'.txt')
        
        relations = read_json(path_workspace+'relation2id.json')
        entities = read_json(path_workspace+'entity2id.json')
        times_id = read_json(path_workspace+'ts2id.json')
        test_ans = convert_dataset(test_ans, path_workspace, period=1)
            
        if name_rules:
            chains = read_json(path_out_tl+name_rules)
        else:
            chains = None #isi
        rel_keys = list(relations.keys())

        all_facts = []
        with open(path_workspace+"all_facts.txt", "r", encoding='utf-8') as f:
            all_facts = f.readlines()
        
        gamma = [parsed["g1"], parsed["g2"], parsed["g3"], parsed["g4"]]

        rtr = Retriever(gamma, test_ans, all_facts, entities, relations, times_id, num_relations, chains, rel_keys, dataset=type_dataset, retrieve_type=retrieve_type)
        test_idx, test_text = rtr.get_output()
        # print(test_idx, test_text)

        if not parsed["sensi_exp"]:
            if files == 'test':
                path_file = path_save + "eval/" + type_dataset + "/"
            elif files == 'valid':
                path_file = path_save + "train/" + type_dataset + "/raw/valid/"
            else:
                path_file = path_save + "train/" + type_dataset + "/raw/train/"

            if not os.path.exists(path_file):
                os.makedirs(path_file)
            path_file_word = path_file + "history_facts_" + type_dataset + "_" + retrieve_type + ".txt"
            

            with open(path_file_word, 'w', encoding='utf-8') as f:
                for i in range(len(test_text)): 
                    f.write(test_text[i][0] + '\n')
            print("saved as ", path_file_word)
            
            path_answer = path_file + "test_ans_" + type_dataset + ".txt"
            write_txt_ans(path_answer, test_ans, head='')
            print("saved as ", path_answer)
        
        else:
            path_file = path_save + "eval/" + type_dataset + "sensitivity/"
            if not os.path.exists(path_file):
                os.makedirs(path_file)
            path_file_word = path_file + "history_facts_" + type_dataset + "_" + retrieve_type + str(parsed["g1"]) + str(parsed["g2"]) + str(parsed["g3"]) + str(parsed["g4"]) + ".txt"

            with open(path_file_word, 'w', encoding='utf-8') as f:
                for i in range(len(test_text)): 
                    f.write(test_text[i][0] + '\n')
            print("saved as ", path_file_word)

            path_answer = path_file + "test_ans_" + type_dataset + ".txt"
            write_txt_ans(path_answer, test_ans, head='')
            print("saved as ", path_answer)
    

