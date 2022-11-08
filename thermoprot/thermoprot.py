"""
ThermoProt: Predict the thermostability of protein sequences with machine learning
"""




import pandas as pd
import numpy as np
import sklearn
import argparse
import pickle
import os






def create_parser():
    '''Parse command-line arguments'''
    
    parser = argparse.ArgumentParser(description='Predict protein thermostability')
    parser.add_argument('--infile', type=str, 
                        help='Path to file containing sequences in fasta format')
    parser.add_argument('--outfile', type=str, 
                        help='Path to which results will be written in csv format')
    parser.add_argument('--modelname', type=str, choices={'PM', 'MT','TH', 'MTH'}, 
                        default='MTH', 
                        help='Binary classification model to use. PM: psychrophic vs. ' \
                             'mesophilic, MT: mesophilic vs. thermophilic, '\
                             'TH: thermophilic vs. hyperthermophilic, '\
                             'MTH: Mesophilic vs. thermophlic/hyperthermophlic.')
    args = parser.parse_args()
    assert os.path.exists(args.infile)
    
    return args
    





def load_data(args):
    '''Return classification model and feature standardization data'''

    this_dir, this_filename = os.path.split(__file__)
    model_path = os.path.join(this_dir, 'models', args.modelname + '.pcl')
    model = pickle.load(open(model_path, 'rb'))
    meanstd_path = os.path.join(this_dir, 'models', 'features_mean_std.csv')
    meanstd = pd.read_csv(meanstd_path, index_col=0)
    
    return model, meanstd




def read_fasta(args):
    '''Read sequences in input file'''
    
    with open(args.infile, 'r') as f:
        headers, sequences = [], []
        for line in f:
            if line.startswith('>'):
                head = line.replace('>','').strip()
                headers.append(head)
                sequences.append('')
            else :
                seq = line.strip()
                if len(seq) > 0:
                    sequences[-1] += seq

    return (headers, sequences)





class FeaturesCalculator():
    
    def __init__(self):
    
        self.aa_list = list('ACDEFGHIKLMNPQRSTVWY')
        # Max. solvent accessible area
        self.maxASA = dict(zip(self.aa_list, 
                               [129, 167, 193, 223, 240, 104, 224, 197, 236, 201, 224,
                                195, 159, 225, 274, 155, 172, 174, 285, 263] ))
        # Molecular weight of AAs in Daltons
        self.mw_aa = dict(zip(self.aa_list, 
                              [89, 121, 133, 146, 165, 75, 155, 131, 146, 131, 149, 132, 
                               115, 147, 174, 105, 119, 117, 204, 281]))
        # Heat capacities of amino acids in cal/(mol*K)
        self.hc_aa = dict(zip(self.aa_list, 
                              [29.22, 50.07, 37.09, 41.84, 48.52, 23.71, 59.64, 45.0, 
                               57.1, 48.03, 69.32, 38.3, 36.13, 44.02, 26.37, 32.4, 35.2,
                               40.35, 56.92, 51.73]))
    
    
    def get_ggap_count(self, seq, g, dplist): 
        '''Return the frequency of g-gap dipeptides in sequence'''
        
        seq_ggap = [seq[i:i+g+2] for i in range(len(seq))] # g-gaps
        seq_ggap = [x for x in seq_ggap if len(x) == g+2]
        count_ggap = []
        for pair in dplist:
            count_ggap.append(np.sum([1 if x[0] == pair[0] and x[-1] == pair[-1] 
                                        else 0 for x in seq_ggap]))
        
        return count_ggap
    
    
    def get_features(self, seq):
        '''Calculate features'''
        
        n = len(seq)
        features = []
        
        # Amino acid composition, AAC (20 features)
        aac = [seq.count(x)/n for x in self.aa_list]
        features.extend(aac)
        aac = dict(zip(self.aa_list, aac))
        
        # 0-gap dipeptide composition, dpc_0gap (4 features)
        dpset = ['AA', 'RE', 'RR', 'EQ', 'QA', 'KQ']
        dp_0gap_count = self.get_ggap_count(seq, 0, dplist=dpset)
        n_0gap = np.sum(dp_0gap_count)
        dpc_0gap = [x/n_0gap if n_0gap != 0 else 0 for x in dp_0gap_count]
        features.extend(dpc_0gap)
        
        # 1-gap dipeptide composition, dpc_1gap (1 feature)
        dpset = ['RR']
        dp_1gap_count = self.get_ggap_count(seq, 1, dplist=dpset)
        n_1gap = np.sum(dp_1gap_count)
        dpc_1gap = [x/n_1gap if n_1gap != 0 else 0 for x in dp_1gap_count]
        features.extend(dpc_1gap)
        
        # 2-gap dipeptide composition, dpc_2gap (5 features)
        dpset = ['AR', 'LQ', 'RR']
        dp_2gap_count = self.get_ggap_count(seq, 2, dplist=dpset)
        n_2gap = np.sum(dp_2gap_count)
        dpc_2gap = [x/n_2gap if n_2gap != 0 else 0 for x in dp_2gap_count]
        features.extend(dpc_2gap)
        
        # Residue type and physiochemical features, RT (20 features)
        # Acidic residue composition (D+E)/N
        rt1 = np.sum([aac[x] for x in 'DE'])

        # Basic residue composition (K+R+H)/N
        rt2 = np.sum([aac[x] for x in 'KRH'])
        
        # Non-polar residue composition  (AGILMFPWV)
        rt3 = np.sum([aac[x] for x in 'AGILMFPWV'])
        
        # Cyclic residue composition (FYWPH)
        rt4 = np.sum([aac[x] for x in 'FYWPH'])
        
        # Aliphatic residue composition (AGILV)
        rt5 = np.sum([aac[x] for x in 'AGILV'])
        
        # Aromatic residue composition (HFWY)
        rt6 = np.sum([aac[x] for x in 'HFWY'])
        
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
        rt12 = np.sum([aac[x] for x in 'EFMR'])
        
        # (E + K) to (Q + H) ratio
        num = np.sum([aac[x] for x in 'EK'])
        den = np.sum([aac[x] for x in 'QH'])
        rt13 = num/den if den!=0 else 0
        
        # CvP (charged vs polar) composition
        rt14 = np.sum([aac[x] for x in 'DEKR']) - np.sum([aac[x] for x in 'NQST'])

        # IVYWREL composition 
        rt15 = np.sum([aac[x] for x in 'IVYWREL'])
        
        # Tiny residues composition
        rt16 = np.sum([aac[x] for x in 'AGPS']) 
        
        # Small residues composition
        rt17 = np.sum([aac[x] for x in 'TD'])
        
        # ASA (average maximum solvent accessible surface area)
        rt18 = np.sum([aac[x] * self.maxASA[x] for x in self.aa_list])
        
        # Molecular weight MW (in kDa)
        rt19 = np.sum([self.mw_aa[x] * aac[x] * n / 1000 for x in self.aa_list])
    
        # Heat capacity, HC
        rt20 = np.sum([aac[x] * self.hc_aa[x] for x in self.aa_list])
        
        features.extend([rt1, rt2, rt3, rt4, rt5, rt6, rt7, rt8, rt9, rt10, rt11, 
                         rt12, rt13, rt14, rt15, rt16, rt17, rt18, rt19, rt20])

        return features            
            
        
        
        


def standardize_features(features, meanstd):
    '''Standardize features to a mean of 0 and std. dev of 1, according to the training
    set distribution'''
    
    return (features - meanstd['mean'].values) / (meanstd['std'].values + 1e-8)


    
    
    
    
def main():
    '''Run prediction'''
    
    args = create_parser()
    model, meanstd = load_data(args)
    headers, sequences = read_fasta(args)
    calculator = FeaturesCalculator()
    features = np.array([calculator.get_features(seq) for seq in sequences])
    features = standardize_features(features, meanstd)
    yproba = model.predict_proba(features)[:,1]
    ypred = (yproba > 0.5)
    ytext = np.array([None] * len(ypred))
    text = {'P': 'Psychrophilic', 'M': 'Mesophilic', 'T': 'Thermophilic', 
            'TH':'Thermophilic/Hyperthermophilic'}
    ytext[~ypred] = text[args.modelname[0]] # False class
    ytext[ypred] = text[args.modelname[1:]] # True class
    ypred = ypred.astype(int)
    dfpred = pd.DataFrame([headers, list(yproba), list(ypred), list(ytext)], 
                          index=['Header', 'Probability', 'Class', 'Prediction'])
    dfpred = dfpred.transpose()
    dfpred.to_csv(args.outfile)






if __name__ == '__main__':
    
    main()
    
    
