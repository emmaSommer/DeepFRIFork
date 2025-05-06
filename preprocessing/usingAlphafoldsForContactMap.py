
import os
import numpy as np
import matplotlib.pyplot as plt
import requests
from Bio.PDB import PDBParser
from io import StringIO

def download_npz(uniprot_id, out_dir="alphafold_npz"):
    url = f"https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/{uniprot_id}.npz"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{uniprot_id}.npz")
    r = requests.head(url)
    if r.status_code == 200:
        print(f"[✓] .npz available: {url}")
        if not os.path.exists(out_path):
            r = requests.get(url)
            with open(out_path, "wb") as f:
                f.write(r.content)
        return out_path
    return None

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def extract_contact_map_from_npz(npz_path, threshold=8.0, save=True):
    data = np.load(npz_path)
    dist_logits = data['distogram']['logits']
    bin_edges = data['distogram']['bin_edges']

    probs = softmax(dist_logits, axis=-1)
    expected_dist = np.sum(probs * bin_edges[:-1], axis=-1)
    contact_map = (expected_dist < threshold).astype(int)

    if save:
        out_path = npz_path.replace(".npz", "_contact.npy")
        np.save(out_path, contact_map)
        print(f"[💾] Contact map saved: {out_path}")

    return contact_map

def download_alphafold_pdb(uniprot_id):
    url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"
    r = requests.get(url)
    if r.status_code != 200:
        print(f"[!] Could not fetch AlphaFold PDB for {uniprot_id}")
        return None
    return r.text

def compute_ca_distance_map(pdb_str, threshold=8.0):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("AF", StringIO(pdb_str))
    model = structure[0]
    chain = next(model.get_chains())

    ca_atoms = [res['CA'] for res in chain if 'CA' in res]
    coords = np.array([atom.coord for atom in ca_atoms])
    dist_matrix = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    contact_map = (dist_matrix < threshold).astype(int)

    print(f"[✓] Contact map calculated from structure, shape: {contact_map.shape}")
    return contact_map

def visualize_map(contact_map, title):
    plt.imshow(contact_map, cmap='Greys', origin='lower')
    plt.title(title)
    plt.xlabel("Residue")
    plt.ylabel("Residue")
    plt.tight_layout()
    plt.show()

def main(uniprot_id, threshold=8.0):
    npz_path = download_npz(uniprot_id)
    if npz_path:
        contact_map = extract_contact_map_from_npz(npz_path, threshold)
        visualize_map(contact_map, f"{uniprot_id} Contact Map (from .npz)")
    else:
        print(f"[!] .npz not available, falling back to PDB for {uniprot_id}")
        pdb_str = download_alphafold_pdb(uniprot_id)
        if pdb_str:
            contact_map = compute_ca_distance_map(pdb_str, threshold)
            np.save(f"{uniprot_id}_ca_contact.npy", contact_map)
            visualize_map(contact_map, f"{uniprot_id} Contact Map (from structure)")
        else:
            print(f"[X] Failed to retrieve prediction for {uniprot_id}")


if __name__ == "__main__":
    # Example UniProt ID
    uniprot_id = "A0QSG7"
    main(uniprot_id)
