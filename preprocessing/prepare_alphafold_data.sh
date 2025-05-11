#!/bin/bash

BASE_DATA_DIR=./../../data/deepFriData
DATA_DIR=./../../data/deepFriData/alphafold
NPZ_DIR=./../../data/deepFriData/annot_pdb_chains_npz
TFR_DIR=./../../data/deepFriData/alphafold/TFRecords
SEQ_SIM=95

printf "\n\n\tLoad alphafold pdbs and create contact maps"
 printf "\n\tValidation data\n"
 python convertPDB2UniProt.py $BASE_DATA_DIR/nrPDB-GO_valid.txt \
 # no uniprot map %
 # no pdb file %

 printf "\n\Training data\n"
 python convertPDB2UniProt.py $BASE_DATA_DIR/nrPDB-GO_train.txt \
 # no uniprot map %
 # no pdb file %

 printf "\n\tTesting data\n"
 python convertPDB2UniProt.py $BASE_DATA_DIR/nrPDB-GO_test.txt \
 # no uniprot map 0.12%
 # no pdb file 0.53%
 

printf "\n\n\t Create npz files from alphafold contact maps"
 printf "\n\tValidation data\n"
 python build_npz_from_alphafold.py \
    --map_dir $DATA_DIR/alphafold_contact_maps \
    --pdb_dir $DATA_DIR/alphafold_pdb \
    --out_dir $NPZ_DIR \
    --pdb_file data/nrPDB-GO_valid.txt

 printf "\n\Training data\n"
 python build_npz_from_alphafold.py \
    --map_dir $DATA_DIR/alphafold_contact_maps \
    --pdb_dir $DATA_DIR/alphafold_pdb \
    --out_dir $NPZ_DIR \
    --pdb_file data/nrPDB-GO_train.txt

 printf "\n\tTesting data\n"
 python build_npz_from_alphafold.py \
    --map_dir $DATA_DIR/alphafold_contact_maps \
    --pdb_dir $DATA_DIR/alphafold_pdb \
    --out_dir $NPZ_DIR \
    --pdb_file data/nrPDB-GO_test.txt

printf "\n\n\t Create TFRecord files from npz files"
 printf "\n\tValidation data\n"
 python PDB2TFRecord.py \
     -annot $BASE_DATA_DIR/nrPDB-GO_annot.tsv \
     -prot_list $BASE_DATA_DIR/nrPDB-GO_valid.txt \
     -npz_dir $NPZ_DIR/ \
     -num_shards 3 \
     -num_threads 3 \
     -tfr_prefix $TFR_DIR/PDB_GO_valid \

 printf "\n\Training data\n"
    python PDB2TFRecord.py \
        -annot $BASE_DATA_DIR/nrPDB-GO_annot.tsv \
        -prot_list $BASE_DATA_DIR/nrPDB-GO_train.txt \
        -npz_dir $NPZ_DIR/ \
        -num_shards 30 \
        -num_threads 30 \
        -tfr_prefix $TFR_DIR/PDB_GO_train \

 printf "\n\tTesting data\n" 
    python PDB2TFRecord.py \
        -annot $BASE_DATA_DIR/nrPDB-GO_annot.tsv \
        -prot_list $BASE_DATA_DIR/nrPDB-GO_test.txt \
        -npz_dir $NPZ_DIR/ \
        -num_shards 3 \
        -num_threads 3 \
        -tfr_prefix $TFR_DIR/PDB_GO_test \