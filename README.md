# Path-based reasoning approach for knowledge graph completion using CNN-BiLSTM with attention mechanism
Knowledge graphs are valuable resources for building intelligent systems such as question answering or recommendation systems. However, most knowledge graphs are impaired by missing relationships between entities. We propose a new approach for knowledge graph completion that combines bidirectional long short-term memory (BiLSTM) and convolutional neural network modules with an attention mechanism. Given a candidate relation and two entities, we encode paths that connect the entities into a low-dimensional space using a convolutional operation followed by BiLSTM. Then, an attention layer is applied to capture the semantic correlation between a candidate relation and each path between two entities and attentively extract reasoning evidence from the representation of multiple paths to predict whether the entities should be connected by the candidate relation.


# 1. Statistics of datasets
||#entities|#relations|#train|#dev|#test|#tasks|
|:-----------:|------------:|------------:|------------:|------------:|------------:|------------:|
|NELL-995|75,492|200|154,213|5,000|5,000|12|
|FB15K-237|14,541|237|272,115|17,535|20,466|10|
|Countries|272|2|1,158|68|72|2|
|Kinship|104|26|6,926|769|1,069|26|

- `#entities` : entities의 갯수   
- `#relations` : relations의 갯수 
- `#train` : Train data의 갯수  
- `#dev` : Development data의 갯수   
- `#test` : Test data의 갯수
- `#tasks` : Link prediction tasks  

# 2. Requirement
해당 코드는 Python 3.6.5에서 실행됨.

### Installing packages
```bash
pip install -r requirements.txt
```

# 3. File description
- `Story_Generator.py` : Data를 model의 input 형태로 processing 하기 위한 python file
- `make_story.sh` : Story_Generator.py를 편리하게 실행시키기 위한 shell script
- `main.py` : Model Training 및 Testing 하기  python file
- `run_main.sh` : main.py를 편리하게 실행시키기 위한 shell script
- `data_utils.py` : 코드 실행 시 필요한 함수들을 정의한 python file


# 4. Run code
**_Note._** data.zip을 압축 해제 후 진행

각 Dataset에 대하여 Pre-Processing을 수행 후 Training과 Testing 수행

### (1) Data Pre-Processing :
```shell
bash make_story.sh --dataset <DATASETS>
```
     
### Options of ``make_story.sh``:
```
useage: [--dataset] - Dataset 이름 (NELL-995, FB15K-237, Countries, Kinship)
```   

-----------------------------------    

### (2) Training and Testing :
```shell
bash run_main.sh 
--dataset <DATASETS>
--LSTM_hidden_size 100
--CNN_num_filter 50
--CNN_pooling_size 2
--batch_size 128
--num_epochs 50
--learning_rate 0.001
--patience 30
--log_name NELL_log
```

### Options of ``run_main.sh``:
```
useage: [--dataset] - Dataset 이름 (NELL-995, FB15K-237, Countries, Kinship)
        [--LSTM_hidden_size] - LSTM hidden unit의 size.
        [--CNN_num_filter] - CNN Filter의 갯수.
        [--CNN_pooling_size] - Pooling size.
        [--batch_size] - Training 과정에서의 Batch size.
        [--num_epochs] - Training 횟수.
        [--learning_rate] - 학습률.
        [--patience] - Early stopping할 Epochs 수.
        [--log_name] - Log file의 이름.
```   
----------------------------------

### Result:
||MAP|MRR|Hits@1|Hits@3|
|:-----------:|------------:|------------:|------------:|------------:|
|NELL-995|0.894|0.898|0.838|0.951|
|FB15K-237|0.652|0.660|0.544|0.708|
|Countries|0.947|0.947|0.916|0.986|
|Kinship|0.946|0.952|0.918|0.984|

# 5. Data description

- `data/<DATASETS>/ontology.txt` : 각 dataset에 대한 Ontology file   
      
- `data/<DATASETS>/paths/`
  - `<RELATIONS>.txt` : Path Ranking Algorithm을 통해 추출된 Relation Paths    
    
- `data/<DATASETS>/tasks/graph.txt` : Knowledge Graphs
- `data/<DATASETS>/tasks/<RELATIONS>/`
  - `train.pair` : Correct, Corrupt Train Data 
  - `train_pos` : Correct Train Data
  - `test.pairs` : Correct, Corrupt Test Data  
  - `sort_test.pairs` : 정렬된 Test Data  
  
# 6. Citation
```
    @article{jagvaral2020path,
      title={Path-based reasoning approach for knowledge graph completion using CNN-BiLSTM with attention mechanism},
      author={Jagvaral, Batselem and Lee, Wan-Kon and Roh, Jae-Seung and Kim, Min-Sung and Park, Young-Tack},
      journal={Expert Systems with Applications},
      volume={142},
      pages={112960},
      year={2020},
      publisher={Elsevier}
    }
```
