import requests
import os
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import requests
from Bio.PDB import PDBParser
from io import StringIO

OUT_DATA_DIR = './../../data/deepFriData/alphafold/'

def get_uniprot_from_pdb_chain(pdb_id, chain_id):
    url = f'https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb_id.lower()}'
    r = requests.get(url)
    if r.status_code != 200:
        return None
    data = r.json()
    mappings = data.get(pdb_id.lower(), {}).get('UniProt', {})
    for uni_id, details in mappings.items():
        for segment in details['mappings']:
            if segment['chain_id'] == chain_id:
                return uni_id
    return None

def download_alphafold_npz(uniprot_id, output_dir="alphafold_npz"):
    url = f"https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/{uniprot_id}.npz"
    os.makedirs(OUT_DATA_DIR + output_dir, exist_ok=True)
    out_path = os.path.join(OUT_DATA_DIR, output_dir, f"{uniprot_id}.npz")

    if os.path.exists(out_path):
        print(f"[✓] {uniprot_id}.npz already downloaded.")
        return

    r = requests.get(url, stream=True)
    if r.status_code == 200:
        with open(out_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"[↓] Downloaded: {uniprot_id}.npz")
    else:
        print(f"[!] Contact map for {uniprot_id} not found.")


def download_alphafold_pdb(uniprot_id, output_dir="alphafold_pdb"):
    # url = f"https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/{uniprot_id}.npz"
    url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"
    os.makedirs(OUT_DATA_DIR + output_dir, exist_ok=True)
    out_path = os.path.join(OUT_DATA_DIR, output_dir, f"{uniprot_id}.pdb")

    if os.path.exists(out_path):
        print(f"[✓] {uniprot_id}.pdb already downloaded.")
        return True

    r = requests.get(url, stream=True)
    if r.status_code != 200:
        print(f"[!] Could not download {uniprot_id}.pdb. Status code = {r.status_code}")
        return False
    else:
        with open(out_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"[↓] Downloaded: {uniprot_id}.pdb")
        return True

def compute_ca_distance_map(pdb_path, threshold=8.0):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("AF", pdb_path)
    model = structure[0]
    chain = next(model.get_chains())

    ca_atoms = [res['CA'] for res in chain if 'CA' in res]
    coords = np.array([atom.coord for atom in ca_atoms])
    dist_matrix = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    contact_map = (dist_matrix < threshold).astype(int)

    print(f"[✓] Contact map calculated from structure, shape: {contact_map.shape}")
    return contact_map

def parse_chain_from_line(line):
    if '-' in line:
        return line[:4], line[5:]
    elif '_' in line:
        return line[:4], line[5:]
    else:
        return line.strip(), None
    
def visualize_map(contact_map, title):
    plt.imshow(contact_map, cmap='Greys', origin='lower')
    plt.title(title)
    plt.xlabel("Residue")
    plt.ylabel("Residue")
    plt.tight_layout()
    plt.show()

def load_pdb_to_uniprot_mapping(file_path):
    mappings = {}
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    for line in lines:
        pdb_id, chain = parse_chain_from_line(line)
        if chain is None:
            print(f"[!] Chain info missing for {pdb_id}")
            continue
        uniprot_id = get_uniprot_from_pdb_chain(pdb_id, chain)
        mappings[pdb_id+chain] = uniprot_id
    

def main(input_file):
    mapping_errors = 0
    download_errors = 0
    with open(input_file, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    for line in lines:
        pdb_id, chain = parse_chain_from_line(line)
        if chain is None:
            print(f"[!] Chain info missing for {pdb_id}")
            continue

        uniprot_id = get_uniprot_from_pdb_chain(pdb_id, chain)
        if uniprot_id:
            print(f"{line} -> UniProt: {uniprot_id}")
            #download_alphafold_npz(uniprot_id)
            if not download_alphafold_pdb(uniprot_id, "alphafold_pdb"):
                download_errors += 1
                continue

            contact_map = compute_ca_distance_map(
                pdb_path = os.path.join(OUT_DATA_DIR + 'alphafold_pdb', uniprot_id + '.pdb'),
            )

            os.makedirs(OUT_DATA_DIR + 'alphafold_contact_maps', exist_ok=True)
            np.save(
                os.path.join(OUT_DATA_DIR + "alphafold_contact_maps", uniprot_id),
                contact_map
            )
            print(f"[💾] Contact map saved to {os.path.join(OUT_DATA_DIR + 'alphafold_contact_maps', uniprot_id + '.npz')}")
        
#            visualize_map(contact_map, f"{uniprot_id}")
#            plt.show()
        else:
            print(f"{line} -> UniProt mapping not found")
            mapping_errors += 1
    print(f"Proteins without mapping: {mapping_errors / len(lines)*100:.2}%")
    print(f"Proteins without pdb file: {download_errors / len(lines)*100:.2}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='pdb2alphafoldcontacts',
                    description='What the program does',
                    epilog='Text at the bottom of help')
    parser.add_argument('filename')
    args = parser.parse_args()

    main(args.filename)
