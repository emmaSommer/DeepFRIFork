from my_train import run_pipeline

DATA_DIR = './../data/deepFriData/'
NPZ_DIR = DATA_DIR + 'alphafold/'

def main():
    run_pipeline(data_folder=NPZ_DIR, epochs=200, output_dir='outputs_alpha_200e')
    run_pipeline(data_folder=DATA_DIR, epochs=200, output_dir='outputs_deepfir_200e')

if __name__ == '__main__':
    main()
