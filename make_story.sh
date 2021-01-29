#!/bin/bash

dataset=${dataset:-f}

args=("$@")
param="${1/--/}"
declare $param="${2}"
echo $dataset

if [ ${dataset} == "NELL-995"  ]
then
	declare -a arr=(
	athleteplaysinleague 
	worksfor 
	organizationhiredperson
	athleteplayssport 
	teamplayssport 
	personborninlocation 
	athletehomestadium 
	organizationheadquarteredincity 
	athleteplaysforteam 
	teamplaysinleague 
	agentbelongstoorganization
	personleadsorganization)
fi

if [ ${dataset} == "FB15K-237"  ]
then
	declare -a arr=(
		film@director@film
		film@film@language
		film@film@written_by
		location@capital_of_administrative_division@capital_of.@location@administrative_division_capital_relationship@administrative_division
		music@artist@origin
		organization@organization_founder@organizations_founded
		sports@sports_team@sport
		tv@tv_program@languages
		people@person@nationality
		people@person@place_of_birth)
fi

if [ ${dataset} == "Kinship"  ]
then
	declare -a arr=(term18 term8 term3 term1 term19 term16 term15 term7 term17 term12 term11 term20 term25 term13 term4 term2 term0 term5 term9 term21 term22 term14 term24 term6 term1)
fi

if [ ${dataset} == "Countries"  ]
then
	declare -a arr=(countries_S1 countries_S2 countries_S3)
fi

for task_name in "${arr[@]}"
do
	echo $task_name
    python Story_Generator.py $task_name $dataset

done
