import requests
import os

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
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{uniprot_id}.npz")

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

def parse_chain_from_line(line):
    if '-' in line:
        return line[:4], line[5:]
    elif '_' in line:
        return line[:4], line[5:]
    else:
        return line.strip(), None

def main(input_file):
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
            download_alphafold_npz
