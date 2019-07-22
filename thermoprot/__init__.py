"""
ThermoProt:
Predict the thermostability of protein sequences with machine learning

Functions
----------

seqPred:
	predict the thermostability of a single protein sequence
	
fastaPred:
	predict the thermostability of protein sequences in a fasta file
"""



from .thermoprot import seqPred
from .thermoprot import fastaPred
