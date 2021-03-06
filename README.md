# Path-based reasoning approach for knowledge graph completion using CNN-BiLSTM with attention mechanism
This is an implementation of Knowledge graph completion using CNN-BiLSTM with attention Model.    
If you have any questions or comments, please fell free to contact us by email [alsgh9963@naver.com].

# Data description

- `data/<DATASETS>/ontology.txt` : 각 dataset에 대한 Ontology file   
      
- `data/<DATASETS>/paths/`
  - `<RELATIONS>.txt` : Path Ranking Algorithm을 통해 추출된 Relation Paths    
    
- `data/<DATASETS>/tasks/graph.txt` : Knowledge Graphs
- `data/<DATASETS>/tasks/<RELATIONS>/`
  - `train.pair` : Correct, Corrupt Train Data 
  - `train_pos` : Correct Train Data
  - `test.pairs` : Correct, Corrupt Test Data  
  - `sort_test.pairs` : 정렬된 Test Data  


# Running
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


  
# Citation
```
본 연구는 2020년도 정부(과학기술정보통신부)의 재원으로 정보통신기술진흥센터의 지원을 받아 수행된 연구임 
(No.2019000067,대용량 지식그래프 자동완성을 위한 시맨틱 분석 추론기술 개발)
```
