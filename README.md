# DeepFlav: A Comprehensive Resource and AI-Powered Platform for Accurate Prediction of Flavonoid Biosynthetic Genes
This repository contains the code and data for the paper titled “DeepFlav: A Comprehensive Resource and AI-Powered Platform for Accurate Prediction of Flavonoid Biosynthetic Genes”, which is currently under review in *Horticulture Research* (HR).

This project aims to utilize the large protein language model (ESM-2, 650M parameters) to extract protein sequence features, and combine with a Multi-Layer Perceptron (MLP) to construct a classifier for high-precision identification of specific enzyme families (such as Methyltransferases MT, Glycosyltransferases GTs, Acyltransferases ATs).

The project covers the complete pipeline from raw sequence acquisition, negative sample construction, data preprocessing, model training to model evaluation. Through the `argparse` module, high parameterization is achieved, and different enzyme families can be seamlessly switched by modifying the `--enzyme` parameter.

## 📦 Environment Dependencies

This project distinguishes between data acquisition environment (CPU) and model training environment (GPU), and it is recommended to configure them step by step.
1. Dependencies 
Please refer directly to the requirements.txt file
2. PyTorch Environment (Only required for training)
If you want to run network_training.py (must use GPU), please obtain the installation command according to your graphics card CUDA version from the PyTorch official website.
For example (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

## ⚙️ Project Configuration and Directory Structure

Before running the code, please ensure the project directory structure is as follows.
Special attention:
1. **ESM-2 model weight file must be downloaded in advance** (`esm2_t33_650M_UR50D.pt`) and placed in the `esm_models/` folder.
2. **ESM-2 model regression parameter file must be downloaded in advance** (`esm2_t33_650M_UR50D-contact-regression.pt`) and placed in the `esm_models/` folder.
text
.
├── esm_models/
│ └── esm2_t33_650M_UR50D.pt # ESM-2 pre-trained weights that need to be manually downloaded
├── output/ # Automatically created during runtime, storing models and logs
│ └── {enzyme}/ # e.g., MT/, GTs/, ATs/
├── fetch_gene_ids.py # [Script 1] Obtain positive sample gene IDs (need to be prepared manually or according to KEGG API)
├── fetch_aaseq.py # [Script 2] Obtain protein sequences based on gene IDs
├── fetch_noflav_fasta_to_csv.py # [Script 3] Obtain non-flavonoid negative samples (excluding flavonoid interference)
├── fetch_random_negative.py # [Script 4] Obtain random enzyme negative samples (excluding AT/MT/GT)
├── filter_sequences.py # [Script 5] Filter overly long sequences to prevent memory overflow
├── network_training.py # [Script 6] ESM-2 feature extraction and MLP training
├── split_test_set.py # [Script 7] Split test set into sequence and label files
├── predict_test_set.py # [Script 8] Load model to predict on test set
├── evaluate_ROC_AUC.py # [Script 9] Calculate AUC and draw ROC curve
└── calculate_performance_metrics.py # [Script 10] Calculate Accuracy/Precision/Recall/F1

## 🚀 Quick Start

The following process takes **Methyltransferase (MT)** as an example. If you need to train other enzyme families, just replace `MT` in the commands with `GTs` or `ATs`.

### 1. Data Acquisition and Construction

1.1 Obtain positive sample gene IDs
python fetch_gene_ids.py --enzyme MT

1.2 Obtain positive sample protein sequences based on gene IDs
python fetch_aaseq.py --enzyme MT

1.3 Obtain non-flavonoid similar enzyme negative samples (background negative samples)
python fetch_noflav_fasta_to_csv.py --enzyme MT

1.4 Obtain other random enzyme negative samples (general negative samples)
python fetch_random_negative.py --enzyme MT

### 2. Data Preprocessing
Merge the obtained positive and negative samples into a training set (containing `sequence` and `label` columns) and name it `MT_training_new_set.csv`. Perform the same operation on the test set and name it `MT_test_set.csv`.

2.1 Filter overly long sequences in the test set (length > 1024)
python filter_sequences.py --enzyme MT --input MT_test_sequences_for_model.csv

2.2 Split the test set to generate pure sequence table and pure label table (for independent prediction and evaluation)
python split_test_set.py --enzyme MT --input MT_test_set.csv

### 3. Model Training
Use ESM-2 to extract features and train MLP classifier. The model and standardization parameters will be automatically saved to the `./output/MT/` directory.
*(Note: This step requires GPU support)*

python network_training.py --enzyme MT --input MT_training_new_set.csv

### 4. Model Prediction
Load the trained model weights and predict on the filtered test sequences, outputting probability values.

python predict_test_set.py --enzyme MT --input MT_test_sequences_for_model_filtered.csv


### 5. Evaluation and Visualization

5.1 Merge prediction results with true labels, calculate AUC and draw ROC curve (output PNG/PDF/SVG)
python evaluate_ROC_AUC.py --enzyme MT

5.2 Calculate Accuracy, Precision, Recall and F1-score
python calculate_performance_metrics.py --enzyme MT


## 📊 Output Results Explanation

After completing the above process, the following key result files will be generated in the project root directory:

- **Model weights and parameters** (in the `./output/{enzyme}/` directory):
  - `{enzyme}_esm2_650M_nn_classifier.pth`: Trained MLP classifier weights
  - `{enzyme}_esm2_650M_feature_mean.npy`: Feature mean (for standardization)
  - `{enzyme}_esm2_650M_feature_std.npy`: Feature standard deviation (for standardization)
- **Evaluation results** (in the root directory):
  - `{enzyme}_ROC_curve.png/pdf/svg`: ROC curve in three formats
  - `{enzyme}_test_predictions_with_labels.csv`: Table containing prediction probabilities and true labels
  - `{enzyme}_performance_metrics.csv`: Final table of four core classification metrics

## 📌 Notes

1. **ESM model download**: The project defaults to loading the model from the local path `./esm_models/esm2_t33_650M_UR50D.pt`. Please ensure you have downloaded this weight file in advance, otherwise training and prediction stages will report errors.
2. **ESM model regression parameter download**: The project assumes users have downloaded the `esm2_t33_650M_UR50D-contact-regression.pt` regression parameter file in advance. Please ensure you have downloaded this regression parameter file, otherwise training stage will report errors.
3. **API rate limit**: During KEGG and UniProt data acquisition, `sleep` has been added in the code to control request rate. If you encounter 403/429 errors, please appropriately increase the seconds in `time.sleep()`, or fill in your real email in the `User-Agent` in the code.
4. **Hardware requirements**: `network_training.py` requires GPU (CUDA) to accelerate ESM-2 inference; the prediction script `predict_test_set.py` supports running on CPU, but is slower.
