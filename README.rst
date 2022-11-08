**ThermoProt**
===============

ThermoProt is a python package to predict the thermostability of proteins.
ThermoProt uses binary classifiers to predict the thermostability of proteins as
	* psychrophilic vs. mesophilic (PM)
	* mesophilic vs. thermophilic (MT)
	* thermophilic vs. hyperthermophilic (TH)
	* mesophilic vs. thermophlic/hyperthermophlic (MTH)



Usage 
-------------
.. code:: shell-session

   git clone https://github.com/jafetgado/ThermoProt.git
   cd ThermoProt
   conda env create -f ./env.yml -p ./env
   conda activate ./env
   python thermoprot/thermoprot.py --infile training_sequences/hyperthermophilic.fasta --outfile ./predictions.csv --modelname MTH




Citation
----------
If you find ThermoProt useful, please cite the following:

Erickson E, Gado JE, et al, 2022. "Sourcing thermotolerant poly(ethylene terephthalate) hydrolase scaffolds from natural diversity.
