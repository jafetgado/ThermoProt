"""
ThermoProt:
Predict the thermostability of protein sequences with machine learning
"""





# Import modules
#==================
import pandas as pd
import sklearn
import thermoprot.prep as prep



    

# Functions to predict thermostability of protein sequences
#==========================================================
def seqPred(seq, clf='MTH', proba=False):
    """
    Predict the thermostability of a protein sequence.
    
    Parameters
    -----------
    seq : str
    	The protein sequence as a string.
    clf : str
    	The classifier type ('PM', 'MT', 'TH', or 'MTH'). If not specified,
    	the default classifier ('MTH'), is used.
    proba : bool
    	If False, the predicted class is returned as a boolean. 
    	If True, instead of the predicted class, the predicted probability 
    	(that the sequence is a positive instance) is returned.
                
    Returns
    ----------
    (pred_class, pred_desc)
    	A tuple of the predicted class and a description of the predicted class.
        If proba=True, (pred_proba, pred_desc) is returned.
    
    Examples
    ------------
    >>> prot = '''MRRELIERLESRLDRREIEKARRDSHARRRPRPCGITVHPGHGCPRACSY
    ...         CYIPEMGFRFERARPYRLSGEGMVLALLYNRGFEPGREGTFIAVGSVTDPFL
    ...         PELADKTLEYLRTFSRWLGNPTQFSTKSAIDGEVAESLARLELPLNGLVTIL
    ...         TPDREKASRLEPRAPRPEERLETITELSKAGLTVDLFFRPILPGIVGLEEAE
    ...         ELFRMARDAGARGVVVGGFRVNEGILSRLKRSGFDVSEIVNRANRPIPKGRK
    ...         QVYVRTGDIKERLLRIAREVGLTPFGAACCACASAAQVPCPNRCWEGPFCTE
    ...         CGNPACPV'''
    >>> pred = tp.seqPred(seq=prot, clf='TH', proba=False)
    >>> print(pred)
    (0, 'Hyper')
    
    prot is DNA photolyase of the hyperthermophilic archaebacteria (Methanopyrus
    kandleri) and is predicted to be thermo/hyperthermophilic
    
    >>> pred = tp.seqPred(seq=prot, clf='TH', proba=False)
    >>> print(pred)
    (0.99999, 'Thermo/Hyper')
    
    prot is predicted to be thermo/hyperthermophilic with a probability of 0.99999
    
    
    """
    
    if clf not in prep.clf_names:
        raise NameError("You must specify the classifier type (clf) as"
        				" 'PM', 'MT', 'TH', or 'MTH'")
    seq = ''.join(char for char in seq if char.isalpha())
    features = pd.DataFrame(prep.get_features([seq]))
    if proba:
        pred_proba = prep.classifiers[clf].predict_proba(features)[0][1]
        pred_proba = round(pred_proba, 5)
        pred = 0 if pred_proba < 0.5 else 1
        return (pred_proba, prep.clf_classes[clf][pred])
    else:
        pred = prep.classifiers[clf].predict(features)[0]
    return (pred, prep.clf_classes[clf][pred])





def fastaPred(fasta, clf='MTH'):
    """
    Predict the thermostability of  proteins in a fasta file.
    
    Parameters
    -----------
    fasta : str
    	Name/path of fasta file containing protein sequences.
    clf : str
    	The classifier type, 'PM', 'MT', 'TH', or 'MTH'. If not specified, 
    	the default classifier, 'MTH', is used.
        
    Returns
    -----------
    	A dataframe containing the predicted class for each protein in the fasta file and  
    	the probability that the sequences are positive instances.
    
    Examples
    -------
    >>> df = tp.fasta_pred(fasta="sequences.fas", clf="MTH")
    >>> df.to_csv('predictions.csv')  # Save predictions as csv file
    
    
    """
    if clf not in prep.clf_names:
        raise NameError("You must specify the classifier type (clf) as 'PM', 'MT', 'TH', or MTH'")
    (headers, sequences) = prep.read_fasta(fasta)
    features = pd.DataFrame(prep.get_features(sequences))
    therm_proba = prep.classifiers[clf].predict_proba(features)[:,1]
    therm_class = [0 if x<0.5 else 1 for x in therm_proba]
    therm_names = [prep.clf_classes[clf][x] for x in therm_class]
    df = pd.DataFrame([headers, therm_class, list(therm_proba), therm_names]).transpose()
    df.columns = ['protein', 'label', 'probability', '{} prediction'.format(clf)]
    return df
    

    
#========================================================================================#