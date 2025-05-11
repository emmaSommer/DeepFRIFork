"""
convert_contacts_to_npz.py
--------------------------

Read the binary Cα-contact maps you already generated
(<protein>.npy) + the matching AlphaFold PDB (<protein>.pdb) and
save DeepFRI-style <protein>.npz files that contain

    C_alpha : NxN   (float32)  – here: 0/1 contacts
    C_beta  : NxN   (float32)  – duplicated from C_alpha
    seqres  : str               – primary sequence in FASTA letters

Usage
-----

python convert_contacts_to_npz.py \
       --map_dir  alphafold_contact_maps \
       --pdb_dir  alphafold_pdb \
       --out_dir  deepfri_npz
"""

import argparse, os, numpy as np, gzip, warnings
from pathlib import Path
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1

from convertPDB2UniProt import OUT_DATA_DIR, get_uniprot_from_pdb_chain, parse_chain_from_line

def cb_distance_matrix(chain):
    coords = []
    for res in chain:
        if res.get_id()[0] != ' ':
            continue
        if 'CB' in res:
            coords.append(res['CB'].coord)
        else:                         # glycine hack: virtual CB
            n, ca, c = res['N'].coord, res['CA'].coord, res['C'].coord
            cb = np.mean([n, ca, c], axis=0) 
            coords.append(cb)
    xyz = np.stack(coords)
    dists = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)
    return (dists < 10).astype(np.float32)   # or keep raw Å distances

def load_pdb_ids(filename):
    pdb_ids = []
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    for line in lines:
        pdb_id, chain = parse_chain_from_line(line)
        if chain is None:
            print(f"[!] Chain info missing for {pdb_id}")
            continue
        pdb_ids.append((pdb_id, chain))
    return pdb_ids


def main(npz_dir, pdb_dir, contact_map_dir, pdb_id_file):
    os.makedirs(npz_dir, exist_ok=True)
    pdb_parser = PDBParser(QUIET=True)
    total = 0
    no_uniprot = 0
    no_pdb = 0
    no_contact_map = 0
    failed = 0
    duplicates = 0
    for pdb_id, chain in load_pdb_ids(pdb_id_file):
        total += 1
    #for npy_path in Path(contact_map_dir).glob('*.npy'):
        pdb_name = f'{pdb_id}-{chain}'
        uniprot_id = get_uniprot_from_pdb_chain(pdb_id, chain)
        if uniprot_id is None:
            print(f"[!] No UniProt ID found for {pdb_id} {chain}")
            no_uniprot += 1
            continue
        
        npz_file = Path(npz_dir, f'{pdb_name}.npz')
        if npz_file.exists():
            print(f"[!] Already exists: {npz_file}")
            duplicates += 1
            continue
        
        uniprot = uniprot_id                   # “P69905” from “P69905.npy”
        pdb_path = Path(pdb_dir, f'{uniprot}.pdb')
        if not pdb_path.exists():
            warnings.warn(f'Skipping {uniprot}: no corresponding PDB found')
            no_pdb += 1
            continue

        # 1 – load your binary contact map (uint8) and cast to float32  
        npy_path = Path(contact_map_dir, f'{uniprot}.npy')
        if not npy_path.exists():
            print(f"[!] No contact map found for {uniprot}")
            no_contact_map += 1
            continue
        ca_contacts = np.load(npy_path).astype(np.float32)

        # 2 – duplicate for Cβ (fallback); if you later compute real Cβ maps,
        #     replace this line with that matrix

        structure = pdb_parser.get_structure('af', pdb_path)
        chain = next(structure[0].get_chains())      # AlphaFold DB ⇒ single chain A
        seqres = ''.join(seq1(residue.get_resname()) for residue in chain.get_residues()
                    if residue.get_id()[0] == ' ')  # skip hetero / waters
        
        cb_contacts = cb_distance_matrix(chain)

        # 4 – persist in DeepFRI format
        np.savez_compressed(npz_file,
                            C_alpha = ca_contacts,
                            C_beta  = cb_contacts,
                            seqres  = seqres)
        print(f'[✓] {pdb_name}.npz written')
    
    failed = no_uniprot + no_pdb + no_contact_map
    print(f"[!] {total} PDB IDs processed")
    print(f"Proteins without uniprot_mapping: {no_uniprot / total*100:.2}%")
    print(f"Proteins without pdb file: {no_pdb / total*100:.2}%")
    print(f"Proteins without contact map: {no_contact_map / total*100:.2}%")
    print(f"Proteins with duplicates: {duplicates / total*100:.2}%")
    print(f"Failed: {failed / total*100:.2}%")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--map_dir', default=OUT_DATA_DIR+"alphafold_contact_maps", help='Directory with *.npy Cα contact maps')
    parser.add_argument('--pdb_dir', default=OUT_DATA_DIR+"alphafold_pdb", help='Directory with AlphaFold PDB files')
    parser.add_argument('--out_dir', default=OUT_DATA_DIR+"annot_pdb_chains_npz", help='Destination for DeepFRI *.npz')
    parser.add_argument('--pdb_file', required=True, help='File with PDB IDs and chains')
    args = parser.parse_args()
    main(args.out_dir, args.pdb_dir, args.map_dir, args.pdb_file)