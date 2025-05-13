import os
import csv
import json
import pickle
import glob
import argparse
import numpy as np
from deepfrier.DeepFRI import DeepFRI
from deepfrier.utils import seq2onehot, load_GO_annot, load_EC_annot
import tensorflow as tf

DATA_DIR = './../data/deepFriData/'
NPZ_DIR = DATA_DIR + 'alphafold/'

def parse_args(npz_folder):
    parser = argparse.ArgumentParser(
        description='Train and evaluate a DeepFRI model with flexible output directory.'
    )
    # model hyperparameters
    parser.add_argument('--gc_dims', type=int, nargs='+', default=[128, 128, 256],
                        help="Dimensions of GraphConv layers.")
    parser.add_argument('--fc_dims', type=int, nargs='+', default=[],
                        help="Dimensions of fully connected layers after graphconv.")
    parser.add_argument('--dropout', type=float, default=0.3, help="Dropout rate.")
    parser.add_argument('--l2_reg', type=float, default=1e-4, help="L2 regularization.")
    parser.add_argument('--lr', type=float, default=2e-4, help="Learning rate.")
    parser.add_argument('--gc_layer', type=str,
                        choices=['GraphConv','MultiGraphConv','SAGEConv','ChebConv','GAT','NoGraphConv'],
                        default='GraphConv', help="Graph conv layer type.")
    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs.")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size.")
    parser.add_argument('--pad_len', type=int, default=None, help="Padding length for sequences.")
    parser.add_argument('--ontology', type=str, default='mf', choices=['mf','bp','cc','ec'],
                        help="Ontology to use.")
    parser.add_argument('--lm_model_name', type=str, default=None,
                        help="Path to pretrained language model.")
    parser.add_argument('--cmap_type', type=str, default='ca', choices=['ca','cb'],
                        help="Contact map type.")
    parser.add_argument('--cmap_thresh', type=float, default=10.0,
                        help="Threshold for contact maps.")
    parser.add_argument('--model_name', type=str, default='GCN_PDB_MF',
                        help="Model name prefix.")
    # data
    parser.add_argument('--train_tfrecord', type=str, default=npz_folder+"TFRecords/PDB_GO_train",
                        help="Glob pattern for train TFRecords.")
    parser.add_argument('--valid_tfrecord', type=str,  default=npz_folder+"TFRecords/PDB_GO_valid",
                        help="Glob pattern for validation TFRecords.")
    parser.add_argument('--test_tfrecord', type=str,  default=npz_folder+"TFRecords/PDB_GO_test",
                        help="Glob pattern for validation TFRecords.")
    parser.add_argument('--annot_fn', type=str, default=DATA_DIR+"nrPDB-GO_annot.tsv",
                        help="Annotation TSV file.")
    parser.add_argument('--test_list', type=str, default="preprocessing/data/nrPDB-GO_test.txt",
                        help="CSV file with test protein IDs.")
    # output
    parser.add_argument('--output_dir', type=str, default='deepFRI_training',
                        help="Directory to store models, plots, and results.")
    parser.add_argument('--data_folder', type=str, default=DATA_DIR,
                        help="Directory to store models, plots, and results.")

    return parser.parse_args()


def run_pipeline(
    gc_dims=[128,128,256], fc_dims=[], dropout=0.3, l2_reg=1e-4, lr=2e-4,
    gc_layer='GraphConv', epochs=10, batch_size=64, pad_len=None,
    ontology='mf', lm_model_name=None, cmap_type='ca', cmap_thresh=10.0,
    model_name='GCN_PDB_MF', train_tfrecord="TFRecords/PDB_GO_train", valid_tfrecord="TFRecords/PDB_GO_valid", test_tfrecord="TFRecords/PDB_GO_test",
    annot_fn=DATA_DIR+"nrPDB-GO_annot.tsv", test_list=DATA_DIR+"nrPDB-GO_test.txt", #test_list="preprocessing/data/nrPDB-GO_test.txt",
    output_dir='deepFRI_training', data_folder=None
):
    """
    Programmatic entry: trains and evaluates DeepFRI using specified parameters.
    Any argument can be overridden; defaults match CLI defaults.
    Returns: (trained_model, test_accuracy)
    """
    
    if data_folder:
        train_tfrecord = data_folder + train_tfrecord
        valid_tfrecord = data_folder + valid_tfrecord
        test_tfrecord = data_folder + test_tfrecord
    
    
    # Build a simple args object
    class Args: pass
    args = Args()
    for k, v in locals().items():
    # only copy the actual parameters of run_pipeline, not the local 'args'
        if k == 'Args' or k == 'args':
            continue
        setattr(args, k, v)

    # ensure patterns for TFRecords
    args.train_tfrecord = train_tfrecord + '*'
    args.valid_tfrecord = valid_tfrecord + '*'
    args.valid_tfrecord = test_tfrecord + '*'
    os.makedirs(output_dir, exist_ok=True)

    prot2annot, goterms, gonames, output_dim, pos_weights = prepare_annotations(args)
    model = train_model(args, goterms, gonames, output_dim)
    test_acc = evaluate_model(model, prot2annot, output_dim, test_list, args)
    return model, test_acc


def prepare_annotations(args):
    # load GO or EC annotations
    if args.ontology == 'ec':
        prot2annot, goterms, gonames, counts = load_EC_annot(args.annot_fn)
    else:
        prot2annot, goterms, gonames, counts = load_GO_annot(args.annot_fn)
    goterms = goterms[args.ontology]
    gonames = gonames[args.ontology]
    output_dim = len(goterms)
    # compute class weights
    class_sizes = counts[args.ontology]
    mean_size = np.mean(class_sizes)
    pw = np.clip(mean_size / class_sizes, 1.0, 10.0)
    pos_weights = {i: {0: pw[i], 1: pw[i]} for i in range(output_dim)}
    return prot2annot, goterms, gonames, output_dim, pos_weights


def train_model(args, goterms, gonames, output_dim):
    model = DeepFRI(
        output_dim=output_dim,
        n_channels=26,
        gc_dims=args.gc_dims,
        fc_dims=args.fc_dims,
        lr=args.lr,
        drop=args.dropout,
        l2_reg=args.l2_reg,
        gc_layer=args.gc_layer,
        lm_model_name=args.lm_model_name,
        model_name_prefix=os.path.join(args.output_dir, args.model_name)
    )
    model.train(
        args.train_tfrecord, args.valid_tfrecord,
        epochs=args.epochs,
        batch_size=args.batch_size,
        pad_len=args.pad_len,
        cmap_type=args.cmap_type,
        cmap_thresh=args.cmap_thresh,
        ont=args.ontology,
        class_weight=None
    )
    # save artifacts
    model.save_model()
    model.plot_losses()
    # save params
    params = vars(args).copy()
    params.update({'goterms': goterms, 'gonames': gonames})
    with open(os.path.join(args.output_dir, args.model_name + '_model_params.json'), 'w') as fw:
        json.dump(params, fw, indent=2)
    return model


def evaluate_model(model, prot2annot, output_dim, test_list, args):
    proteins, Y_pred, Y_true = [], [], []
    print(test_list)
    total = 0
    found = 0
    no_annot = 0
    no_npz = 0
    no_annot_no_npz = 0
    print(test_list)
    with open(test_list, 'r') as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for row in reader:
            total += 1
            prot = row[0]
            npz = os.path.join(args.data_folder + '/annot_pdb_chains_npz/' + prot + '.npz')
            if not os.path.exists(npz) or  prot not in prot2annot:
                if not os.path.exists(npz) and prot not in prot2annot:
                    no_annot_no_npz += 1
                elif not os.path.exists(npz):
                    no_npz += 1
                elif prot not in prot2annot:
                    no_annot += 1
                continue
            found += 1
            data = np.load(npz)
            seq = str(data['seqres'])
            A = (data['C_alpha'] < args.cmap_thresh).astype(int)
            S = seq2onehot(seq)
            A = A[np.newaxis, ...]
            S = S[np.newaxis, ...]
            proteins.append(prot)
            pred = model.predict([A, S]).reshape(output_dim)
            true = prot2annot[prot][args.ontology].reshape(output_dim)
            Y_pred.append(pred)
            Y_true.append(true)
    Y_pred = np.vstack(Y_pred)
    Y_true = np.vstack(Y_true)
    # compute multi-label accuracy per dataset
    pred_bin = (Y_pred >= 0.5).astype(int)
    acc = (pred_bin == Y_true).mean()
    # save results and accuracy
    results = {
        'proteins': proteins,
        'Y_pred': Y_pred,
        'Y_true': Y_true,
        'accuracy': acc,
        'ontology': args.ontology
    }
    with open(os.path.join(args.output_dir, args.model_name + '_results.pckl'), 'wb') as pf:
        pickle.dump(results, pf)
    # store accuracy summary
    summary = {'test_acc': acc, 'total': total, 'found': found, 'test_list': test_list,
               'no_annot': no_annot, 'no_npz': no_npz, 'no_annot_no_npz': no_annot_no_npz}
    with open(os.path.join(args.output_dir, 'accuracy_summary.json'), 'w') as af:
        json.dump(summary, af, indent=2)
    print(f"Test accuracy: {acc:.4f}")
    return acc


def main():
    args = parse_args(DATA_DIR)
    model, test_acc = run_pipeline(
        gc_dims=args.gc_dims,
        fc_dims=args.fc_dims,
        dropout=args.dropout,
        l2_reg=args.l2_reg,
        lr=args.lr,
        gc_layer=args.gc_layer,
        epochs=args.epochs,
        batch_size=args.batch_size,
        pad_len=args.pad_len,
        ontology=args.ontology,
        lm_model_name=args.lm_model_name,
        cmap_type=args.cmap_type,
        cmap_thresh=args.cmap_thresh,
        model_name=args.model_name,
        train_tfrecord=args.train_tfrecord,
        valid_tfrecord=args.valid_tfrecord,
        test_tfrecord=args.test_tfrecord,
        annot_fn=args.annot_fn,
        test_list=args.test_list,
        output_dir=args.output_dir
    )

if __name__ == '__main__':
    main()
