# Setting up environment

## Folder structure
- root = /home/xlogin
- data_folder = root/data/deepFriData
- repo = root/DeepFRIFork
 - git clone the repo to root so it should be root/DeepFRIFork

## Preparation

- pip install : tensorflow, biopython, scikit-learn, networkx, numpy, obonet
- copy the content of repo/preprocessing/data to data_folder
- run the remove_version.sh script in data_folder

## Data import

- run repo/preprocessing/data_collection.sh
- - currently fails because the calculated residues do not match seqres values (eg error: Somehow the final residue list 80 doesn't match the size of the SEQRES seq 212 )
- - can look at logs in root/DeepFRIFork/preprocessing/dist_map_creation.log (previous content is not deleted when script is run again, so if you run it several times, the logs will all be there)