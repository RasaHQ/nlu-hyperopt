#! /bin/bash 
su

echo $INPUT_MAX_EVALS

config=$(python $@) 

echo "::set-output name=best_config::config"