from my_train import run_pipeline

DATA_DIR = './../data/deepFriData/'
NPZ_DIR = DATA_DIR + 'alphafold/'
ALPHA_DIR = './../data/deepFriData/alphafold/'

def main():
    #run_pipeline(data_folder=ALPHA_DIR, epochs=1, output_dir='test_alpha_eval_should_be_consistent', test_list=DATA_DIR+"nrPDB-GO_test.txt")

    run_pipeline(data_folder=ALPHA_DIR, epochs=200, output_dir='outputs_alpha_200e')
    run_pipeline(data_folder=DATA_DIR, epochs=200, output_dir='outputs_deepfir_200e')

if __name__ == '__main__':
    main()
