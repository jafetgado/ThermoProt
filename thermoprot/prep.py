"""
Preparatory functions to calculate machine learning features for ThermoProt
"""


# Import modules
#==================
from __future__ import division
import os
import numpy as np
import pandas as pd
import pickle



# Load classifiers and data for calculating features
#====================================================
def load_pickle(pcl_file):
    return pickle.load(open(pcl_file, 'rb'))   


this_dir, this_filename = os.path.split(__file__)  # Get absolute path of data
pcl_names = ['PM.pcl', 'MT.pcl', 'TH.pcl', 'MTH.pcl']
svm_paths = [os.path.join(this_dir, 'models_data', x) for x in pcl_names]
mean_std_path = os.path.join(this_dir, 'models_data', 'mean_std.pcl')
mean_std = load_pickle(mean_std_path)
svms = [load_pickle(x) for x in svm_paths]
clf_names = ['PM', 'MT', 'TH', 'MTH']
classifiers = dict(zip(clf_names, svms))
clf_classes = pd.DataFrame([['Psychro', 'Meso', 'Thermo', 'Meso'], 
                ['Meso', 'Thermo', 'Hyper', 'Thermo/Hyper']], 
                columns=clf_names)
aa_list = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R','S',
           'T', 'V', 'W', 'Y']   # 20 Amino acid letters
maxASA = [129, 167, 193, 223, 240, 104, 224, 197, 236, 201, 224, 195, 159, 225, 274, 155, 
          172, 174, 285, 263]  # ASA of amino acids
mw_aa = [89, 121, 133, 146, 165, 75, 155, 131, 146, 131, 149, 132, 115, 147, 174, 105, 
         119, 117, 204, 281]    # Molecular weight of AAs in Da
hc_aa = [29.22, 50.07, 37.09, 41.84, 48.52, 23.71, 59.64, 45.0, 57.1, 
         48.03, 69.32, 38.3, 36.13, 44.02, 26.37, 32.4, 35.2, 40.35,
         56.92, 51.73] # Heat capacities of amino acids in cal/(mol*K)




# Functions
#=================
def get_ggap_count(seq, g, dplist): 
    """Return the frequency of g-gap dipeptides in dplist."""
    seq_ggap = [seq[i:i+g+2] for i in range(len(seq))]
    seq_ggap = [x for x in seq_ggap if len(x) == g+2]
    count_ggap = []
    for pair in dplist:
        count_ggap.append(np.sum([1 if x[0] == pair[0] and x[-1] == pair[-1] 
                                    else 0 for x in seq_ggap]))
    return count_ggap


def calculate_features(seq):
    """Return a list of 50 calculated features for a protein sequence (seq)."""
    
    n = len(seq)
    features = []
    
    # Amino acid composition, AAC (20 features)
    aa_count = [seq.count(x) for x in aa_list]
    aac = [x/n for x in aa_count]
    features.extend(aac)
    
    # 0-gap dipeptide composition, dpc_0gap (4 features)
    dpset = ['AA', 'RE', 'RR', 'EQ', 'QA', 'KQ']
    dp_0gap_count = get_ggap_count(seq, 0, dplist=dpset)
    n_0gap = np.sum(dp_0gap_count)
    dpc_0gap = [x/n_0gap if n_0gap != 0 else 0 for x in dp_0gap_count]
    features.extend(dpc_0gap)
    
    # 1-gap dipeptide composition, dpc_1gap (1 feature)
    dpset = ['RR']
    dp_1gap_count = get_ggap_count(seq, 1, dplist=dpset)
    n_1gap = np.sum(dp_1gap_count)
    dpc_1gap = [x/n_1gap if n_1gap != 0 else 0 for x in dp_1gap_count]
    features.extend(dpc_1gap)
    
    # 2-gap dipeptide composition, dpc_2gap (5 features)
    dpset = ['AR', 'LQ', 'RR']
    dp_2gap_count = get_ggap_count(seq, 2, dplist=dpset)
    n_2gap = np.sum(dp_2gap_count)
    dpc_2gap = [x/n_2gap if n_2gap != 0 else 0 for x in dp_2gap_count]
    features.extend(dpc_2gap)
    
    # Residue type and physiochemical features, RT (20 features)
    # Acidic residue composition (D+E)/N
    rt1 = np.sum([aac[aa_list.index(x)] for x in ['D','E']])
    # Basic residue composition (K+R+H)/N
    rt2 = np.sum([aac[aa_list.index(x)] for x in ['K', 'R', 'H']])
    # Non-polar residue composition  (AGILMFPWV)
    rt3 = np.sum([aac[aa_list.index(x)] for x in 
                       ['A', 'G', 'I', 'L', 'M', 'F', 'P', 'W', 'V']])
    # Cyclic residue composition (FYWPH)
    rt4 = np.sum([aac[aa_list.index(x)] for x in ['F', 'Y', 'W', 'P', 'H']])
    # Aliphatic residue composition (AGILV)
    rt5 = np.sum([aac[aa_list.index(x)] for x in ['A', 'G', 'I', 'L', 'V']])
    # Aromatic residue composition (HFWY)
    rt6 = np.sum([aac[aa_list.index(x)] for x in ['H', 'F', 'W', 'Y']])
    # Charged residue composition (DEKRH)
    rt7 =rt1 + rt2
    # Basic to acidic ratio
    rt8 = rt2/rt1 if rt1 != 0 else 0
    # Nonpolar to polar ratio
    rt9 = rt3/(1-rt3) if rt3 != 1 else 0
    # Cyclic to acyclic ratio
    rt10 = rt4/(1-rt4) if rt4 != 1 else 0
    # Charged to non-charged ratio
    rt11 = rt7/(1-rt7) if rt7 != 1 else 0
    # EFMR composition
    rt12 = np.sum([aac[aa_list.index(x)] for x in ['E', 'F', 'M', 'R']])
    # (E + K) to (Q + H) ratio
    num = np.sum([aac[aa_list.index(x)] for x in ['E', 'K']])
    den = np.sum([aac[aa_list.index(x)] for x in ['Q', 'H']])
    rt13 = num/den if den!=0 else 0
    # CvP (charged vs polar) composition
    rt14 = np.sum([aac[aa_list.index(x)] for x in ['D', 'E', 'K', 'R']]) \
                - np.sum([aac[aa_list.index(x)] for x in ['N', 'Q', 'S', 'T']])
    # IVYWREL composition 
    rt15 = np.sum([aac[aa_list.index(x)] for x in ['I', 'V', 'Y', 'W', 'R', 'E', 'L']])
    # Tiny residues composition
    rt16 = np.sum([aac[aa_list.index(x)] for x in ['A', 'G', 'P', 'S']]) 
    # Small residues composition
    rt17 = np.sum([aac[aa_list.index(x)] for x in ['T', 'D']])
    # ASA (average maximum solvent accessible surface area)
    rt18 = np.sum([aac[i]*maxASA[i] for i in range(20)])
    # Molecular weight MW (in kDa)
    rt19 = np.sum([mw_aa[i] * aa_count[i] for i in range(20)])/1000
    # Heat capacity, HC
    rt20 = np.sum([aac[i] * hc_aa[i] for i in range(20)])
    
    features.extend([rt1, rt2, rt3, rt4, rt5, rt6, rt7, rt8, rt9, rt10, rt11, 
                     rt12, rt13, rt14, rt15, rt16, rt17, rt18, rt19, rt20])
    return features


def standardize_features(features_list):
    """Return a list of standardized features for a list of 50 features."""
    return [(features_list[i] - mean_std['mean'][i])/mean_std['std'][i] 
            for i in range(len(features_list))]
 
    
def get_features(seq_list):
    """Return a list of standardized features for a list of protein sequences."""
    raw_features = [calculate_features(seq) for seq in seq_list]
    standard_features = [standardize_features(feat) for feat in raw_features]
    return standard_features


def read_fasta(fasta):
    """
    Read the protein sequences in a fasta file
    
    Parameters
    -----------
    fasta: str
    	Filename of fasta file containing protein sequences
    
    Returns
    ----------
    (list_of_headers, list_of_sequences)
    	A tuple of corresponding lists of  protein descriptions 
    	and protein sequences in fasta_file
    	
    """
    with open(fasta, 'r') as fast:
        headers, sequences = [], []
        for line in fast:
            if line.startswith('>'):
                head = line.replace('>','').strip()
                headers.append(head)
                sequences.append('')
            else :
                seq = line.strip()
                if len(seq) > 0:
                    sequences[-1] += seq
    return (headers, sequences)
        
#========================================================================================#