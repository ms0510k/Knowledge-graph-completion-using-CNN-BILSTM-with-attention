#!/bin/bash

dataset=${dataset:-NELL-995}
LSTM_hidden_size=${LSTM_hidden_size:-100}
CNN_num_filter=${CNN_num_filter:-50}
CNN_pooling_size=${CNN_pooling_size:-2}
batch_size=${batch_size:-128}
num_epochs=${num_epochs:-50}
learning_rate=${learning_rate:-0.001}
patience=${patience:-30}
log_name=${log_name:-NELL_log}

args=("$@")
for i in 0 2 4 6 8 10 12 14 16
do
	param="${args[$i]/--/}"
	declare $param="${args[$i+1]}"
	shift
done

if [ ${dataset} == "NELL-995"  ]
then
	declare -a arr=(
	worksfor 
	organizationhiredperson 
	organizationheadquarteredincity
	athleteplayssport
	teamplayssport
	personborninlocation
	athletehomestadium
	organizationheadquarteredincity 
	athleteplaysforteam
	agentbelongstoorganization
	teamplaysinleague
	personleadsorganization)

elif [ ${dataset} == "FB15K-237"  ]
then
	declare -a arr=(film@director@film
	film@film@language
	film@film@written_by
	location@capital_of_administrative_division@capital_of.@location@administrative_division_capital_relationship@administrative_division
	music@artist@origin
	organization@organization_founder@organizations_founded
	people@person@nationality
	people@person@place_of_birth
	sports@sports_team@sport
	tv@tv_program@languages)

elif [ ${dataset} == "Kinship"  ]
then
	declare -a arr=(term18 term8 term3 term1 term19 term16 term15 term7 term17 term12 term11 term20 term25 term13 term4 term2 term0 term5 term9 term21 term22 term14 term24 term6 term1)


elif [ ${dataset} == "Countries"  ]
then
	declare -a arr=(countries_S1 countries_S2 countries_S3)
fi


file=$log_name

if [ -f $file ] ; then
    rm $file
    touch $file
fi

for task_name in "${arr[@]}"
do
    python main.py $task_name $dataset $LSTM_hidden_size $CNN_num_filter $CNN_pooling_size $batch_size $num_epochs $learning_rate $patience $file
done







