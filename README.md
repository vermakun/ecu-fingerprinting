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
- [ ] Fix SNR Signal and Raw Signal alignment for more accurate calculation

# Usage

*New*:

Two primary files are provided: 
* ecu_fingerprint_sandbox.x - a feature sandbox that processes a single file to demonstrate how feature extraction calculations work
* ecu_fingerprint_classification.x - processes all files within the Dataset and performs machine learning training and classification to identify  ECU records with a test set, with performance metrics calculated to compare selected featuresets and ML hyperparameters

Also, depending on your IDE environment use the appropriate file types when running the code:
* ecu_fingerprint_x.ipynb (for Jupyter Notebooks)
* ecu_fingerprint_x.py    (Python 2.7/3.X)

## Initialization

* Make sure all relevant Python packages are installed in your environment before running the code (not needed for Google Colab users):
	* numpy
	* pandas
	* seaborn
	* matplotlib
	* scipy
	* sklearn
* For Google Colab users, the dataset and library files will need to be copied from your Google Drive. For this to work, you will need to make sure the corresponding lines of code indicated by ### Google Colab Only ### are uncommented. Follow instructions to enable Google Drive availability in your Colab instance.

## File-preconditioning

* Look for the line declaring "datapath"
	* For non-Google Colab users, the path string should be: '.\Dataset'
	* For Google-Colab users, adjust the path string to the location of the dataset within your Google Drive

## Feature Extraction

* Look for the line starting with "!cp" this line should be uncommented for Google Colab users and pointing to the correct library python file
* For non-Google Colab users, this line should be commented out since the library file should already be in your current directory
	* If not, make sure to comment it out
		
## Training and Test Datasets

* There are a number of commented out lines
* These represent different feature selections for the neural network training
* The only uncommented line is the the best performing featureset
* Feel free to change featuresets by commenting/uncommenting lines in this section
