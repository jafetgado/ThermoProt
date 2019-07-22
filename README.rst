**ThermoProt**
===============

ThermoProt is a python package to predict the thermostability of proteins as psychrophilic,
mesophilic, thermophilic, or hyperthermophilic using machine learning.



Installation
-------------
Install with pip

.. code:: shell-session

   pip install ThermoProt

Or from source code

.. code:: shell-session
   git clone https://github.com/jafetgado/ThermoProt.git
   cd ThermoProt
   python setup.py install



Prerequisites
-------------

1. Python 3
2. sklearn
3. numpy
4. pandas

Usage
-----
There are 2 main functions in thermoprot:

1. seqPred: predicts the thermostability of a single protein sequence.
2. fastaPred: predicts the thermostability of protein sequences in a fasta file and returns the predictions as a Pandas dataframe.

Examples
--------
.. code:: python

   import thermoprot as tp

   # Predict thermostability of a sequence
   seq = "MVRVPRERSGTRSALGEASTYPVGAMTSQHDDQMTFYEAVGGEETFTRLA"
   pred = tp.seqPred(seq, clf='MTH', probability=False)  # clf can be PM, MT, TH or MTH

   # Predict thermostability of sequences in fasta file and write results to spreadsheet
   df = tp.fastaPred(fasta='sequences.fas', clf='MTH')
   df.to_csv('predictions.csv')   # Write to spreadsheet


Citation
----------
In published works, please cite this paper:

Gado, J.E., Beckham, G.T., Payne, C.M (2019). Predicting protein thermostability
with machine learning.
