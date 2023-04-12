import pandas as pd
import spacy, benepar
from apted import APTED, apted, single_path_functions, node_indexer, resources, helpers
from collections import Counter 
import math
from scipy import spatial, stats
import pandas as pd
import numpy as np
import re

nlp = spacy.load('en_core_web_lg')
nlp.add_pipe(benepar.BeneparComponent("benepar_en3"))

def cont_parse(utt):
    try:
        return list(nlp(utt).sents)[0]._.parse_string
    except:
        return ''

def clean_str(string):
    return ''.join(e for e in string if e.isalnum())

def cont_tree_nodes_only(tree_parse_string):
    
    if tree_parse_string == '()':
        return '()'
    
    else:
    
        start = []
        end = []
        ch = ''
        ts = ''.join(list(reversed(tree_parse_string)))
        for i, x in enumerate(ts):
            ch += x
            if i != 0:

                if (x != ')') and (ts[i-1] == ')'):
                    start.append(i)
                    count = i
                    for y in ts[i:]:
                        count +=1
                        if y == ' ':
                            end.append(count)
                            break

        cords = list(zip(start, end))


        s = ''

        count = 0

        for cord in cords:

            count += 1

            if count > 1:
                seq_len = len(ts[cord[0]: cord[1]])

                seq_ = ''.join(['_' for x in range(seq_len)])

                s = s[:cord[0]] + seq_ + s[cord[1]:]

            else:


                seq_len = len(ts[cord[0]: cord[1]])

                seq_ = ''.join(['_' for x in range(seq_len)])



                s = ts[:cord[0]] + seq_ + ts[cord[1]:]

        return ''.join(list(reversed(s.replace('_', ''))))

def js_syntax_convert(s):
    return s.replace(')', ']').replace('(', '[')

def buildVector(iterable1, iterable2):
    
    counter1 = Counter(iterable1)
    counter2= Counter(iterable2)
    all_items = set(counter1.keys()).union( set(counter2.keys()) )
    vector1 = [counter1[k] for k in all_items]
    vector2 = [counter2[k] for k in all_items]
    return vector1, vector2

def pre_traversal_str(s):
    return [x.split()[0] for x in [x.strip() for x in re.split('[()]', s) if x != ''] if x != ''][:]
                
                
def tree_depth(s):
    li = re.split('[)]', s)
    li = [x.strip() for x in li]
    for i, x in enumerate(li):
        if x == '':
            
            
            if set(['']) == set(li[i:]):
                if len(li[i:]) == 1:
                    return 0
                else: 
                    if set(['']) == set(li[i:]):
                    
                    
                        return len(li[i:])
            
            else:
                continue
                
        else:
            continue



def apt_edit_tree_dist_format(tree_string):
    return tree_string.replace('(', '{').replace(')', '}').replace(' ', '')


def apt_edit_tree_dist_format_bt(tree_string, bracket_type1, bracket_type2):
    return tree_string.replace(bracket_type1, '{').replace(bracket_type2, '}')


def count_sub_trees_root(string_tree):
    
    string_tree = apt_edit_tree_dist_format(string_tree)
    
    num_sub_trees = 0
    
    try:
    
        for i, x in enumerate(string_tree):

            if (x == '}') and  (string_tree[i+1] == '}') and (string_tree[i+2] == '}'):
                num_sub_trees += 1
                break

            else:
                if (x == '}') and  (string_tree[i+1] == '}'):
                    num_sub_trees += 1

        return num_sub_trees
    except:
        return get_all_sub_trees_count(string_tree)[1]


def get_sub_trees_from_root(str_tree):
    
    str_tree = apt_edit_tree_dist_format(str_tree)
    
    
    sub_tree_preorder_strings = []
    start_sub_tree_indices = []
    end_sub_tree_indices = []
    start_sub_tree_indices.append(0)
    
    try:
        for i, x in enumerate(str_tree):


            if (x == '}') and  (str_tree[i+1] == '}') and (str_tree[i+2] == '}'):
                end_sub_tree_indices.append(i+1)
                break

            else:
                if (x == '}') and  (str_tree[i+1] == '}'):
                    end_sub_tree_indices.append(i+1)
                    start_sub_tree_indices.append(i+1)

        for cord in list(zip(start_sub_tree_indices, end_sub_tree_indices)):

            sub_tree_preorder_strings.append(str_tree[cord[0]:cord[1]])

        sub_tree_preorder_strings = [x.replace('}', ' ').replace('{', ' ').replace('  ', ' ').strip() for x in sub_tree_preorder_strings]

        return sub_tree_preorder_strings
    
    except:
        return get_all_sub_trees_count(str_tree)[0]
                



def get_all_sub_trees_count(str_tree):
    sub_trees = [x.replace('}', ' ').replace('{', ' ').strip()  for x in apt_edit_tree_dist_format(str_tree).split('}{')]
    num_sub_trees = len(sub_trees)
    
    return [sub_trees, num_sub_trees]


def compare_all_sub_trees(tree1, tree2):
    
    num_shared_sub_trees = 0
    
    prop_sub_tree_shared = 0
    
    prop_sub_tree = 0
    
    prop_sub_trees_li = []
    
    tree1_preorder_subtrees = get_all_sub_trees_count(tree1)[0]
    tree2_preorder_subtrees = get_all_sub_trees_count(tree2)[0]
    
    tree1_preorder_subtrees_num = get_all_sub_trees_count(tree1)[1]
    tree2_preorder_subtrees_num = get_all_sub_trees_count(tree2)[1]
    
    tree1_indices = [i for i in range(len(tree1_preorder_subtrees))]
    
    for ix, x in enumerate(tree2_preorder_subtrees):
        if x in tree1_preorder_subtrees:
            num_shared_sub_trees += 1
            
        else:
            if ix in tree1_indices:
                if x != tree1_preorder_subtrees[ix]:
                    for e in x.split():
                        if e in tree1_preorder_subtrees[ix].split():

                            prop_sub_tree += 1 

    
    
    try:                
        prop_sub_trees_shared = prop_sub_tree/sum([len(x.split()) for x in tree1_preorder_subtrees])
        
    except:
        prop_sub_trees_shared = 0
                     
    
    return [num_shared_sub_trees, prop_sub_trees_shared]



def compare_root_sub_trees(tree1, tree2):
    
    num_shared_sub_trees = 0
    
    prop_sub_tree_shared = 0
    
    prop_sub_tree = 0
    
    prop_sub_trees_li = []
    
    tree1_preorder_subtrees = get_sub_trees_from_root(tree1)
    tree2_preorder_subtrees = get_sub_trees_from_root(tree2)
    
    tree1_indices = [i for i in range(len(tree1_preorder_subtrees))]
    
    for ix, x in enumerate(tree2_preorder_subtrees):
        if x in tree1_preorder_subtrees:

            num_shared_sub_trees += 1
            
        else:

            if ix in tree1_indices:
                if x != tree1_preorder_subtrees[ix]:
                    for e in x.split():
                        if e in tree1_preorder_subtrees[ix].split():

                            prop_sub_tree += 1 
                            
                            
    try:                
        prop_sub_trees_shared = prop_sub_tree/sum([len(x.split()) for x in tree1_preorder_subtrees])
        
    except:
        prop_sub_trees_shared = 0
    
    return [num_shared_sub_trees, prop_sub_trees_shared]
    
    
def compare_preorder_traversal_differences(preorder_traversed_tree1, preordered_traversed_tree2):
    
    deviation_diff = 0
    num_remaining_differences = 0
    
    for i, x in enumerate(preorder_traversed_tree1):
        if i >= len(preordered_traversed_tree2):
    
            num_remaining_differences = len(preorder_traversed_tree1[i:])
            
            break
                
        
        elif x == preordered_traversed_tree2[i]:
                deviation_diff -= 1
        else:
            deviation_diff += 1
            
    
    return deviation_diff + num_remaining_differences


def get_bases_for_entropy(tree_preoder_traversals):
    bases = collections.Counter([tmp_base for tmp_base in tree_preoder_traversals])
    
    return bases

 
def estimate_shannon_entropy(tree_preoder_traversal, bases):
    
    m = len(tree_preoder_traversal)
    bases = bases

    shannon_entropy_value = 0
    for base in bases:
        # number of residues
        n_i = bases[base]
        # n_i (# residues type i) / M (# residues in column)
        p_i = n_i / float(m)
        entropy_i = p_i * (math.log(p_i, 2))
        shannon_entropy_value += entropy_i

    return shannon_entropy_value * -1



def compare_traversal_preorder_tree_deviation(preorder_traversed_tree1, preorder_traversed_tree2, tree1, tree2):
    
    ### num root sub trees
    ### num all sub trees
    ### compare all sub trees
    ### compare root sub trees
    ### tree order traversal
    
    
    v1 = preorder_traversed_tree1
    v2 = preorder_traversed_tree2
    
    nv = []
    
     
    deviation_diff = 0
    
    source_tree = 0
    
    source_tree_num_nodes = len(v1)
    
    target_tree_num_nodes = len(v2)
    
    set_diff_source_target  = len(set(v1) - (set(v2)))
    
    set_diff_target_source  = len(set(v2) - (set(v1)))
    
    tree1_num_root_sub_trees = count_sub_trees_root(tree1)
    
    tree1_num_all_sub_trees = get_all_sub_trees_count(tree1)[1]
    
    tree2_num_root_sub_trees = count_sub_trees_root(tree2)
    
    tree2_num_all_sub_trees = get_all_sub_trees_count(tree2)[1]
    
    prop_shared_all_sub_trees = compare_all_sub_trees(tree1, tree2)[1]
    
    prop_shared_root_sub_trees = compare_root_sub_trees(tree1, tree2)[1]
    
    tree1_tree2_num_shared_all_trees = compare_all_sub_trees(tree1, tree2)[0]
    
    tree1_tree2_num_shared_root_trees = compare_root_sub_trees(tree1, tree2)[0]
    
    tree2_tree1_num_shared_all_trees = compare_all_sub_trees(tree1, tree2)[0]
    
    tree2_tree1_num_shared_root_trees = compare_root_sub_trees(tree1, tree2)[0]
    
    
    

    symmetric_diff = len(set(v2).symmetric_difference(set(v1)))
    
    
    if v1 == v2:
        return {'deviation_diff': deviation_diff, 
                'source_tree':  source_tree,
                'set_diff_source_target': set_diff_source_target, 
                'set_diff_target_source': set_diff_target_source,
                'source_tree_num_nodes': source_tree_num_nodes, 
                'target_tree_num_nodes':target_tree_num_nodes,
                'tree1_num_root_sub_trees': tree1_num_root_sub_trees,
                'tree1_num_all_sub_trees': tree1_num_all_sub_trees,
                'tree2_num_root_sub_trees': tree2_num_root_sub_trees,
                'tree2_num_all_sub_trees': tree2_num_all_sub_trees,
                'prop_shared_all_sub_trees': prop_shared_all_sub_trees,
                'prop_shared_root_sub_trees': prop_shared_root_sub_trees,
                'tree1_tree2_num_shared_all_trees': tree1_tree2_num_shared_all_trees,
                'tree1_tree2_num_shared_root_trees': tree1_tree2_num_shared_root_trees,
                'tree2_tree1_num_shared_all_trees': tree2_tree1_num_shared_all_trees,
                'tree2_tree1_num_shared_root_trees': tree2_tree1_num_shared_root_trees
                }
    
    else:
        
        deviation_diff = compare_preorder_traversal_differences(v1, v2)
     

                    
        return {'deviation_diff': deviation_diff, 
                'source_tree':  source_tree,
                'set_diff_source_target': set_diff_source_target, 
                'set_diff_target_source': set_diff_target_source,
                'source_tree_num_nodes': source_tree_num_nodes, 
                'target_tree_num_nodes':target_tree_num_nodes,
                'tree1_num_root_sub_trees': tree1_num_root_sub_trees,
                'tree1_num_all_sub_trees': tree1_num_all_sub_trees,
                'tree2_num_root_sub_trees': tree2_num_root_sub_trees,
                'tree2_num_all_sub_trees': tree2_num_all_sub_trees,
                'prop_shared_all_sub_trees': prop_shared_all_sub_trees,
                'prop_shared_root_sub_trees': prop_shared_root_sub_trees,
                'tree1_tree2_num_shared_all_trees': tree1_tree2_num_shared_all_trees,
                'tree1_tree2_num_shared_root_trees': tree1_tree2_num_shared_root_trees,
                'tree2_tree1_num_shared_all_trees': tree2_tree1_num_shared_all_trees,
                'tree2_tree1_num_shared_root_trees': tree2_tree1_num_shared_root_trees}
    
    
    
def tree_sim_dev(v1, v2, tree1, tree2, tree1_depth, tree2_depth):
    
    tree_comp = compare_traversal_preorder_tree_deviation(v1, v2, tree1, tree2)
    
    tree1_depth = [tree1_depth]
    tree2_depth = [tree2_depth]
    

    source_tree_keys = ['source_tree', 'set_diff_source_target', 'source_tree_num_nodes', 
                        'tree2_tree1_num_shared_all_trees', 'tree2_tree1_num_shared_root_trees',
                       'tree1_tree2_num_shared_all_trees', 'tree1_tree2_num_shared_root_trees',
                       'tree1_num_root_sub_trees', 'tree1_num_all_sub_trees']
    
    target_tree_keys = ['deviation_diff', 'set_diff_target_source', 'target_tree_num_nodes',
                       'tree1_tree2_num_shared_root_trees', 'tree1_tree2_num_shared_all_trees',
                       'tree2_tree1_num_shared_all_trees', 'tree2_tree1_num_shared_root_trees',
                       'tree2_num_root_sub_trees', 'tree2_num_all_sub_trees']
    
    source_tree_vals = [tree_comp[k] for k in source_tree_keys]
    target_tree_vals = [tree_comp[k] for k in target_tree_keys]
    
    source_tree_node_type_counts = buildVector(v1, v2)[0]
    target_tree_node_type_counts = buildVector(v1, v2)[1]
    
    vs = buildVector(v1, v2)[0] + tree1_depth + source_tree_vals
    vt =  buildVector(v1, v2)[1] + tree2_depth + target_tree_vals
    
    
    r = 1 - spatial.distance.cosine(vs, vt)
    
    return {'cosine_sim': r, 'tree1_depth': tree1_depth[0], 
           'tree2_depth': tree2_depth[0],
           'source_tree_vals': source_tree_vals, 
           'target_tree_vals': target_tree_vals,
           'source_node_counts': source_tree_node_type_counts,
           'target_node_counts': target_tree_node_type_counts}


def calculate_tree_edit_distance(tree1_text, tree2_text):

    t1 = helpers.Tree.from_text(tree1_text)
    t2 = helpers.Tree.from_text(tree2_text)

    ted = APTED(t1, t2)

    return APTED.compute_edit_distance(ted)


def tree_edit_sim(tree_edit_distance):
    return 1/(1+tree_edit_distance)
    

def tree_edit_sim_sqrt(tree_edit_distance):
    
    return math.sqrt(1/(1+tree_edit_distance))
    
    
    
