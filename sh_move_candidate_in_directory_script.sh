#!/bin/bash
while IFS='/' read -r first rest;
do
    d=$( echo '*'$first'*' )
    echo $d
    echo $(../mels_candidate_input/"$first")
    # echo mkdir $(../mels_candidate_input/"$first") 
    # cp $d ../mels_candidate_input/$first
done < /home/lebbat/Documents/audio_mushra_election/44_fichiers_uniq_a_synth.txt