# UMD-SS21-ECE-5831-ECU-Fingerprinting
A Python based project that uses a neural network to categorize messages from known ECUs on a CAN vehicle network.

## Project: In-Vehicle Security using Pattern Recognition Techniques

Created | Submitted
--------|----------
2021.08.05 | 2021.08.20 

## Contact
Kunaal Verma - vermakun@umich.edu

# Current Task List
- [ ] Add relevant files to GitHub repository
- [ ] Add LaTeX report files
- [ ] Cleanup filenames, references, and general usage information
- [ ] Modify Record class to produce output for legacy Methods 1 & 2 (Control + Spectral Analysis)
- [ ] Modify Record class to handle file manangement and metadata recording
- [ ] Rename Record class to provide more meaning to users
- [ ] Propogate peak detection logic and remove redundant bug-related quick fixes
- [ ] Create method in Record to handle object instantiation for Control features (similar to Spectral)
- [ ] Add instantiation tags for various data subsets (Control, Spectral, Dominant, Recessive, etc.)

# Usage

Depending on your IDE environment, use the appropriate files when running the code:
* ece_5831_project_methodX.ipynb (for Jupyter Notebooks)
* ece_5831_project_methodX.py (Python 2.7/3.X)
	
Note: X signifies a number, either 1 or 2, pertaining to different Pattern Recognition methods
1. Control Parameter Features
2. Control Parameter + Spectral Analysis Features

## "I. Initialization"

Make sure all relevant Python packages are installed before running the code

## "II. File-preconditioning"

Look for the line declaring "datapath "
Make sure that "datapath" is directed to the provided Data folder path correctly.
If this is configured correctly, this section of the code should run without issue.
If not, make sure that this path is in your Python PATH variable.

## "III. Feature Extraction"

Look for the line starting with " !cp "
This line should be commented out (it is used for Google Colab)
If not, make sure to comment it out
The rest of this section should run fine if MethodX.py is in the same directory as ECE_5831_Project_MethodX.ipynb/.py
		
## "IV. Training and Test Datasets"

There are a number of commented out lines
These represent different feature selections for the neural network training
The only uncommented line is the the best performing featureset
Feel free to change featuresets by commenting/uncommenting lines in this section
