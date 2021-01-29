import pandas as pd
from IPython.display import display, HTML
import gc
import random
import sys
import os
from functools import reduce

# parameters 정의
task = sys.argv[1]
dataset = sys.argv[2].lower()

# data directiory 정의 
kinship="kinship"
fb="fb15k-237"
nell="nell-995"
countries='countries'
main_dir = os.getcwd()+'/'
task_dir = ''
if fb == dataset:
    task_dir = main_dir + 'data/FB15k-237/tasks/' + task     
elif nell == dataset:
    task_dir = main_dir + 'data/NELL-995/tasks/'
elif kinship == dataset:
    task_dir = main_dir + 'data/kinship/tasks/' + task
elif countries == dataset:
    task_dir = main_dir + 'data/countries/tasks/' + task

# ontology directory 정의
ontology_filepath_nell = main_dir + 'data/NELL-995/nell_ontology.csv'
ontology_filepath_fb   = main_dir + 'data/FB15k-237/fb15k_ontology.txt'
ontology_filepath_kinship = main_dir
ontology_filepath_countries = main_dir + 'data/countries/country_schema.txt'

# 각 dataset에 대한 parsing
if fb == dataset:
    task_uri = '/' + task.replace('@', '/')
elif nell == dataset or kinship == dataset:
    task_uri = task
elif countries == dataset:
    task_uri = 'locatedIn'

# NELL-995 data parsing 함수
def clean_nell(line):
    return line.replace('\n', '').replace('concept:', '').replace('thing$', '').replace("concept_", '').replace(":", "_")

# NELL-995 data
if nell == dataset:

    # relation path data 불러오기
    relation_paths = []
    with open(main_dir + "data/NELL-995/paths/" + task) as raw_file:
        for line in raw_file.readlines():
            path = line.replace(":", "_").replace('\n', '').split(',')
            out_path = []
            # inverse 표현 방법 변경 / '_' -> '_inv'
            for r in path:
                if r.startswith('_'):
                    out_path.append(r[1:] + '_inv')
                else:
                    out_path.append(r)
            path = ','.join(out_path)

            # 필요없는 relation path filtering
            if 'date' not in path:
                relation_paths.append(path)
    
    # Knowledge graphs 불러오기            
    kb = list()
    with open(main_dir +'data/NELL-995/tasks/graph.txt') as graph_file:
        for line in graph_file:
            # 필요없는 triple filtering
            if 'latitudelongitude' not in line and 'date' not in line:
                e1, r, e2 = clean_nell(line).split('\t')
                kb.append((e1, r, e2))         

    # train data 불러오기            
    train_data = []
    with open(task_dir + 'concept_' + task + '/train.pairs') as raw_file:
        # parsing
        for line in raw_file:
            label = line.strip()[-1]
            ee = line.replace('concept_', '').strip()[:-3]
            e1, e2 = ee.split(',')
            e1 = e1[6:]
            e2 = e2[6:]     
            train_data.append((e1, e2, label))

    # test data 불러오기        
    test_data = []
    with open(task_dir + 'concept_' + task + '/sort_test.pairs') as raw_file: 
        # parsing
        for line in raw_file:
            label = line.strip()[-1]            
            ee = line.replace('concept_', '').strip()[:-3]
            e1, e2 = ee.split(',')
            e1 = e1[6:]
            e2 = e2[6:]
            test_data.append((e1, e2, label)) 

# FB15K-237 data
elif fb == dataset:
    # relation path data 불러오기
    relation_paths = []
    with open(main_dir + 'data/FB15k-237/paths/' + task.replace('@', '-')) as raw_file:
        for line in raw_file.readlines()[1:-2]:
            path = line.replace("-", "/").replace('\n', '').split(',')
            out_path = []
            # inverse 표현 방법 변경 / '_' -> '_inv'
            for r in path:
                if r.startswith('_'):
                    out_path.append(r[1:] + '_inv')
                else:
                    out_path.append(r)
            path = ','.join(out_path)
            # 필요없는 relation path filtering
            if 'date' not in path:
                relation_paths.append(path)
    
    # Knowledge Graphs 불러오기            
    kb = list()
    with open( "data/FB15k-237/tasks/graph.txt") as graph_file:
        for line in graph_file:
            # 필요없는 triple filtering
            if 'latitudelongitude' not in line and 'date' not in line:
                e1, r, e2 = clean_nell(line).split('\t')
                kb.append((e1, r, e2))         
    
    # train data 불러오기
    train_data = []
    with open(task_dir + '/train.pairs') as raw_file:
        # parsing
        for line in raw_file:
            label = line.strip()[-1]
            ee = line.replace('concept_', '').strip()[:-3]
            e1, e2 = ee.split(',')
            e1 = '/m/' + e1[8:]
            e2 = '/m/' + e2[8:]     
            train_data.append((e1, e2, label))
    
    # test data 불러오기
    test_data = []
    with open(task_dir + '/sort_test.pairs') as raw_file: 
        # parsing
        for line in raw_file:
            label = line.strip()[-1]            
            ee = line.replace('concept_', '').strip()[:-3]
            e1, e2 = ee.split(',')
            e1 = '/m/' + e1[8:]
            e2 = '/m/' + e2[8:]
            test_data.append((e1, e2, label)) 

# Kinshop, Countries Data            
elif kinship == dataset or countries == dataset:
    # relation path data 불러오기
    relation_paths = []
    if dataset == countries: # countires
        path_dir = main_dir + 'data/countries/paths/' + task
    elif dataset == kinship: # kinship
        path_dir = main_dir + 'data/kinship/paths/' + task +'.txt'
    with open(path_dir) as raw_file:
        for line in raw_file.readlines(): # countries
            path = line.replace('\n', '').split(',')
            out_path = []
            # inverse 표현 방법 변경 / '_' -> '_inv'
            for r in path:
                if r.startswith('_'):
                    out_path.append(r[1:] + '_inv')
                else:
                    out_path.append(r)
            path = ','.join(out_path)  
            relation_paths.append(path)
    
    # Knowledge Graphs 불러오기             
    kb = list()
    with open(task_dir + "/graph.txt") as graph_file:
        # parsing
        for line in graph_file:
            e1, r, e2 = clean_nell(line).split('\t')
            kb.append((e1, r, e2))         
    
    # train data 불러오기
    train_data = []
    with open(task_dir + '/train.pairs') as raw_file:
        # parsing
        for line in raw_file:
            label = line.strip()[-1]
            ee = line.strip()[:-3]
            e1, e2 = ee.split(',')
            train_data.append((e1, e2, label))
    
    # test data 불러오기
    test_data = []
    with open(task_dir + '/sort_test.pairs') as raw_file: 
        # parsing
        for line in raw_file:
            label = line.strip()[-1]            
            ee = line.strip()[:-3]
            e1, e2 = ee.split(',')

            test_data.append((e1, e2, label)) 

# train_data에서 Correct data만 추출
train_pos = set(filter(lambda x:x[2] == '+', train_data))
train_pos = list(map(lambda x:(x[0], task_uri, x[1]), train_pos))

# subject와 object가 바뀐 inverse 관계 추가
train_pos_inv = list(map(lambda x:(x[2], task_uri + '_inv', x[0]), train_pos))

kb = kb + train_pos
kb = kb + train_pos_inv

print('train_data:', len(train_data))
print('test_data:', len(test_data))
print('kb:', len(set(kb)))
print('paths:', len(relation_paths))

# ontology data 생성
if nell == dataset:
    ontology_file = open(ontology_filepath_nell, 'r').readlines()
elif fb == dataset:
    ontology_file = open(ontology_filepath_fb, 'r').readlines()

# NELL-995 data 해당 relation의 domain range를 반환해주는 함수
# input : relation,  output : (domain, range)
def get_domrange_nell(target):    
    target_domain = ''
    target_range = ''
    for line in ontology_file:
        if 'concept:' + target + '\tdomain\tconcept' in line:
            target_domain = line.split('\t')[-1].replace('\n', '').split(':')[-1]
        if 'concept:' + target + '\trange\tconcept' in line:
            target_range = line.split('\t')[-1].replace('\n', '').split(':')[-1]

    return (target_domain, target_range)

# FB15K-237 data 해당 relation의 domain range를 반환해주는 함수
# input : relation,  output : (domain, range)
def get_domrange_fb(target):    
    target_domain = ''
    target_range = ''
    for line in ontology_file:
        if target in line:
            _, target_domain, target_range = line.replace('\n', '').replace('.', '').split('\t')

    return (target_domain, target_range)

# Kinship 해당 relation의 domain range를 반환해주는 함수
# input : relation,  output : ('person', 'person')
def get_domrange_kinship(target):
    return ("person", "person")

# 모든 relations 정의
relation_list = map(lambda x:x.replace('_inv', '').split(','), relation_paths)
relations = set(reduce(lambda x,y: x + y, relation_list))

# relation을 key값으로 (domain, range)를 value로 갖는 dictionary 생성
# key : relation,  value : (domain, range)
meta_dict = {}
if nell == dataset:
    for r in relations:
        meta_dict[r] = get_domrange_nell(r)

elif fb == dataset:
    for r in relations:
        meta_dict[r] = get_domrange_fb(r)

elif kinship == dataset:
    for r in relations:
        meta_dict[r] = get_domrange_kinship(r)

elif countries == dataset:
    ontology_file = open(ontology_filepath_countries, 'r').readlines()
    for line in ontology_file:
        e, e_type = line.split('\t')
        meta_dict[e] = e_type.replace('\n', '')

# sentence를 생성해주는 함수
# input : train/test data, relation, relation paths, knowledge graphs
# output : sentence
def get_sentences(input_data, task_uri, relation_paths, kb):
    result = list()
    count = 0
    num_relations = len(relation_paths) 
    input_data = list(map(lambda x:x[:2], input_data))

    # query triple의 subject object를 DataFrame으로 생성
    i1_df = pd.DataFrame.from_records(input_data, columns=['s1', 'o2'])
    i2_df = pd.DataFrame.from_records(input_data, columns=['s1', 'o3'])
    i3_df = pd.DataFrame.from_records(input_data, columns=['s1', 'o1'])

    i4_df = pd.DataFrame.from_records(input_data, columns=['s1', 'o4'])
    i5_df = pd.DataFrame.from_records(input_data, columns=['s1', 'o5'])

    ii = 0
    for line in relation_paths:
        r_list = line.split(',')
        ii += 1
        if ii % 100 == 0:
            print(ii)
        # relation paths의 길이가 2일 때    
        if len(r_list) == 2:

            r1, r2 = r_list
            r1 = r1.strip() # relation paths 중 첫 번째 relation
            r2 = r2.strip() # relation paths 중 두 번째 relation
            tset_r1 = list(filter(lambda x:x[1] == r1,kb)) # 첫 번째 relation을 가지는 triples
            tset_r2 = list(filter(lambda x:x[1] == r2,kb)) # 두 번째 relation을 가지는 triples

            # 첫 번째 relation을 가지는 triple과 두 번째 relation을 가지는 triple을 join
            df1 = pd.DataFrame.from_records(tset_r1, columns=['s1', 'p1', 'key']) 
            df2 = pd.DataFrame.from_records(tset_r2, columns=['p2', 'key', 'o2'])
            mdf = pd.merge(df1, df2, on='key').drop_duplicates()
            mdf2 = pd.merge(mdf, i1_df, on=['s1', 'o2'])        
            mdf2 = mdf2[['s1', 'p1', 'key', 'p2', 'o2']]

            cr = mdf2.values

            for e in cr:
                result.append(list(e))

            del [[df1, df2, mdf, mdf2]]
            gc.collect()
        # relation paths의 길이가 3일 때
        elif len(r_list) == 3:

            r1, r2, r3 = r_list
            r1 = r1.strip() # relation paths 중 첫 번째 relation
            r2 = r2.strip() # relation paths 중 두 번째 relation
            r3 = r3.strip() # relation paths 중 세 번째 relation
            tset_r1 = list(filter(lambda x:x[1] == r1,kb)) # 첫 번째 relation을 가지는 triples
            tset_r2 = list(filter(lambda x:x[1] == r2,kb)) # 두 번째 relation을 가지는 triples
            tset_r3 = list(filter(lambda x:x[1] == r3,kb)) # 세 번째 relation을 가지는 triples

            # 첫 번째 relation을 가지는 triple과 두 번째 relation을 가지는 triple을 join
            # join 된 dataframe과 세 번째 relation을 가지는 triple join
            df1 = pd.DataFrame.from_records(tset_r1, columns=['s1', 'p1', 'key1'])
            df2 = pd.DataFrame.from_records(tset_r2, columns=['key1','p2', 'key2'])
            df3 = pd.DataFrame.from_records(tset_r3, columns=['key2', 'p3', 'o3'])

            mdf = pd.merge(df1, df2, on='key1').drop_duplicates()
            mdf2 = pd.merge(mdf, df3, on='key2').drop_duplicates()

            mdf3 = pd.merge(mdf2, i2_df, on=['s1', 'o3'])
            mdf3 = mdf3[['s1', 'p1', 'key1', 'p2', 'key2', 'p3', 'o3']]
            
            cr = mdf3.values

            for e in cr:
                result.append(list(e))

            del [[df1, df2, df3, mdf, mdf2, mdf3]]
            gc.collect() 

        # relation paths의 길이가 4일 때
        elif len(r_list) == 4:

            r1, r2, r3, r4 = r_list

            r1 = r1.strip() # relation paths 중 첫 번째 relation
            r2 = r2.strip() # relation paths 중 두 번째 relation
            r3 = r3.strip() # relation paths 중 세 번째 relation
            r4 = r4.strip() # relation paths 중 네 번째 relation

            tset_r1 = list(filter(lambda x:x[1] == r1,kb)) # 첫 번째 relation을 가지는 triples
            tset_r2 = list(filter(lambda x:x[1] == r2,kb)) # 두 번째 relation을 가지는 triples
            tset_r3 = list(filter(lambda x:x[1] == r3,kb)) # 세 번째 relation을 가지는 tripl
            tset_r4 = list(filter(lambda x:x[1] == r4,kb)) # 네 번째 relation을 가지는 tripl

            # 첫 번째 relation을 가지는 triple과 두 번째 relation을 가지는 triple을 join
            # 차례대로 join 된 dataframe과 세 번째, 네 번째 relation을 가지는 triple과의 join
            df1 = pd.DataFrame.from_records(tset_r1, columns=['s1', 'p1', 'key1'])
            df2 = pd.DataFrame.from_records(tset_r2, columns=['key1','p2', 'key2'])
            df3 = pd.DataFrame.from_records(tset_r3, columns=['key2', 'p3', 'key3'])
            df4 = pd.DataFrame.from_records(tset_r4, columns=['key3', 'p4', 'o4'])

            mdf = pd.merge(df1, df2, on='key1').drop_duplicates()
            mdf2 = pd.merge(mdf, df3, on='key2').drop_duplicates()
            mdf3 = pd.merge(mdf2, df4, on='key3').drop_duplicates()
            mdf4 = pd.merge(mdf3, i4_df, on=['s1', 'o4'])

            mdf4 = mdf4[['s1', 'p1', 'key1', 'p2', 'key2', 'p3', 'key3', 'p4', 'o4']]
            
            cr = mdf4.values

            for e in cr:
                result.append(list(e))

            del [[df1, df2, df3, df4, mdf, mdf2, mdf3, mdf4]]
            gc.collect() 

        # relation paths의 길이가 1일 때
        elif len(r_list) == 1 and task_uri != r_list[0]:

            r1 = r_list[0].strip()
            tset_r1 = list(filter(lambda x:x[1] == r1,kb))

            label1 = ['s1', 'p1', 'o1']
            df1 = pd.DataFrame.from_records(tset_r1, columns=label1)
            mdf = pd.merge(df1, i3_df, on=['s1', 'o1'])
            mdf = mdf[['s1', 'p1', 'o1']]
            
            cr = mdf.values

            for e in cr:
                result.append(list(e))

            del [[df1, mdf]]
            gc.collect() 
            
    return result

# train sentence data, test sentence data 생성 
train_sentences = get_sentences(train_data, task_uri, relation_paths, kb)
test_sentences = get_sentences(test_data, task_uri, relation_paths, kb)

# countries data의 sentence 중 entities를 해당 type로 변환하는 함수
# input : sentence, output : type으로 표현된 sentence
def get_dom_range_country(path):
    
    new_path = [''] * len(path)
    
    if len(path) == 7:

        new_path[0] = meta_dict[path[0]]
        new_path[2] = meta_dict[path[2]]
        new_path[4] = meta_dict[path[4]]           
        new_path[6] = meta_dict[path[6]]           
        new_path[1] = path[1]
        new_path[3] = path[3]
        new_path[5] = path[5]
        
    elif len(path) == 5:

        new_path[0] = meta_dict[path[0]]
        new_path[2] = meta_dict[path[2]]
        new_path[4] = meta_dict[path[4]]
        new_path[1] = path[1]
        new_path[3] = path[3]

    else:

        new_path[0] = meta_dict[path[0]]
        new_path[1] = path[1]
        new_path[2] = meta_dict[path[2]]

    return new_path

# sentence 중 entities를 해당 type로 변환하는 함수
# input : sentence, output : type으로 표현된 sentence
def get_dom_range(path):
    
    new_path = [''] * len(path)

    if nell == dataset:
        new_path[0] = path[0].split('_')[0]
        new_path[-1] = path[-1].split('_')[0]

    if len(path) == 11:
        r1 = path[1]
        r2 = path[3]
        r3 = path[5]
        r4 = path[7]
        r5 = path[9]

        # inverse 관계를 처리하기 위한 코드
        if 'inv' in r1:            
            if fb == dataset or dataset == kinship:
                new_path[0] = meta_dict[r1.replace('_inv', '')][1]
            new_path[2] = meta_dict[r1.replace('_inv', '')][0]
        else:
            if fb == dataset or dataset == kinship:
                new_path[0] = meta_dict[r1][0]
            new_path[2] = meta_dict[r1][1]

        if 'inv' in r2:
            new_path[4] = meta_dict[r2.replace('_inv', '')][0]            
        else:
            new_path[4] = meta_dict[r2][1]

        if 'inv' in r3:
            new_path[6] = meta_dict[r3.replace('_inv', '')][0]            
        else:
            new_path[6] = meta_dict[r3][1]

        if 'inv' in r4:
            new_path[8] = meta_dict[r4.replace('_inv', '')][0]            
        else:
            new_path[8] = meta_dict[r4][1]

        if fb == dataset or dataset == kinship:
            if 'inv' in r5:
                new_path[-1] = meta_dict[r5.replace('_inv', '')][0]
            else:
                new_path[-1] = meta_dict[r5][1]

        new_path[1] = r1
        new_path[3] = r2
        new_path[5] = r3        
        new_path[7] = r4
        new_path[9] = r5
    
    if len(path) == 9:
        r1 = path[1]
        r2 = path[3]
        r3 = path[5]
        r4 = path[7]

        # inverse 관계를 처리하기 위한 코드
        if 'inv' in r1:            
            if fb == dataset or dataset == kinship:
                new_path[0] = meta_dict[r1.replace('_inv', '')][1]
            new_path[2] = meta_dict[r1.replace('_inv', '')][0]
        else:
            if fb == dataset or dataset == kinship:
                new_path[0] = meta_dict[r1][0]
            new_path[2] = meta_dict[r1][1]

        if 'inv' in r2:
            new_path[4] = meta_dict[r2.replace('_inv', '')][0]            
        else:
            new_path[4] = meta_dict[r2][1]

        if 'inv' in r3:
            new_path[6] = meta_dict[r3.replace('_inv', '')][0]            
        else:
            new_path[6] = meta_dict[r3][1]

        if fb == dataset or dataset == kinship:
            if 'inv' in r4:
                new_path[-1] = meta_dict[r4.replace('_inv', '')][0]
            else:
                new_path[-1] = meta_dict[r4][1]

        new_path[1] = r1
        new_path[3] = r2
        new_path[5] = r3        
        new_path[7] = r4

    if len(path) == 7:
        r1 = path[1]
        r2 = path[3]
        r3 = path[5]

        # inverse 관계를 처리하기 위한 코드
        if 'inv' in r1:            
            if fb == dataset or dataset == kinship:
                new_path[0] = meta_dict[r1.replace('_inv', '')][1]
            new_path[2] = meta_dict[r1.replace('_inv', '')][0]
        else:
            if fb == dataset or dataset == kinship:
                new_path[0] = meta_dict[r1][0]
            new_path[2] = meta_dict[r1][1]

        if 'inv' in r2:
            new_path[4] = meta_dict[r2.replace('_inv', '')][0]            
        else:
            new_path[4] = meta_dict[r2][1]

        if fb == dataset or dataset == kinship:
            if 'inv' in r3:
                new_path[-1] = meta_dict[r3.replace('_inv', '')][0]
            else:
                new_path[-1] = meta_dict[r3][1]

        new_path[1] = r1
        new_path[3] = r2
        new_path[5] = r3

    # inverse 관계를 처리하기 위한 코드    
    elif len(path) == 5:
        r1 = path[1]
        r2 = path[3]
        if 'inv' in r1:            
            if fb == dataset or dataset == kinship:
                new_path[0] = meta_dict[r1.replace('_inv', '')][1]
            new_path[2] = meta_dict[r1.replace('_inv', '')][0]
        else:
            if fb == dataset or dataset == kinship:
                new_path[0] = meta_dict[r1][0]
            new_path[2] = meta_dict[r1][1]
        
        if fb == dataset or dataset == kinship:
            if 'inv' in r2:
                new_path[-1] = meta_dict[r2.replace('_inv', '')][0]
            else:
                new_path[-1] = meta_dict[r2][1]

        new_path[1] = r1
        new_path[3] = r2

    else:
        r1 = path[1]
        if fb == dataset or dataset == kinship:
            # inverse 관계를 처리하기 위한 코드 
            if 'inv' in r1:
                new_path[0] = meta_dict[r1.replace('_inv', '')][1]
                new_path[-1] = meta_dict[r1.replace('_inv', '')][0]
            else:
                new_path[0] = meta_dict[r1][0]
                new_path[-1] = meta_dict[r1][1]
        new_path[1] = r1
    return new_path


# 생성된 path들을 기반으로 story를 생성해주는 함수
# input : sentence, output : train/test data
def make_story(input_sentences, input_data):
    skip_count = 0    
    output_data = []
    output_list = []
    for i, sample in enumerate(input_data):
        e1, e2, l = sample

        cxt_path_list = list(filter(lambda path: path[0] == e1 and path[-1] == e2, input_sentences))
        if dataset == countries:
            cxt_sent = list(set(map(lambda path:" ".join(get_dom_range_country(path)), cxt_path_list))) 
        else:
            cxt_sent = list(set(map(lambda path:" ".join(get_dom_range(path)), cxt_path_list))) 
                
        if len(cxt_sent) != 0:
            context = "\n".join(cxt_sent)
            output_data.append([context, e1, e2, l])
        else:
            skip_count += 1
            
        if i % 1000 == 0:
            print(i, "/", len(input_data))
            
    print("input_data:", len(input_data))
    print('skip_count:', skip_count)
    
    return output_data

# train story data, test story data 생성
train_result = make_story(train_sentences, train_data)
test_result = make_story(test_sentences, test_data)

# 생성된 story를 편리하게 불러오기 위해 parsing하는 함수
# input : story, output : parsing story
def convert_to_sentence(input_data):
    output_data = []
    for context, e1, e2, l in input_data:
        output_data.append(context + '\n' + e1 + '\t' + task_uri + '\t' + e2 + '\t' + l)
    return output_data

train_out = convert_to_sentence(train_result)
test_out = convert_to_sentence(test_result)
print('num of training samples:', len(train_out))
print('num of test samples:', len(test_out))


# Output directory 생성 
dataDirName = main_dir + "data/processed_data/" + task
 
# Pre-processing 된 data 저장
try:
    os.mkdir(dataDirName)
    print("Directory " , dataDirName, " Created") 
except FileExistsError:
    print("Directory " , dataDirName, " already exists")

with open(dataDirName + "/" + 'train.txt','w') as f:
    f.write( '\n'.join( train_out ) )

with open(dataDirName + "/" + 'test.txt','w') as f:
    f.write( '\n'.join( test_out ) )