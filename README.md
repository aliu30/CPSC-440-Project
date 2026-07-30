# Low-Resource Question Classification on TREC

This repository contains a CPSC 440 course project on **low-resource question
classification**. The task is to assign an English question from the TREC
dataset to one of six coarse classes: abbreviation (`ABBR`), description
(`DESC`), entity (`ENTY`), human (`HUM`), location (`LOC`), or numeric answer
(`NUM`).

This project was developed jointly for UBC CPSC 440 by [Andrew Young Liu](https://github.com/aliu30) and [@Diamond01010111](https://github.com/Diamond01010111).

The project compares conventional text classifiers, a recurrent neural model,
and a pretrained transformer under several limited-data settings. It also
examines whether each model's confidence is consistent with its observed
accuracy. The notebook is preserved as the primary record of the implemented
course work; this README describes its workflow without adding unreported
results or conclusions.

## Research questions

The implemented experiments address the following questions:

1. How does the amount of labelled training data affect question-classification
   performance across model families?
2. In limited-data settings, how do bag-of-words and TF-IDF baselines compare
   with a GloVe-initialized BiLSTM and fine-tuned DistilBERT?
3. How does predictive calibration vary by model and training-set size, as
   measured by Expected Calibration Error (ECE) and reliability diagrams?

## Implemented models

| Model family | Text representation | Classifier and notebook configuration |
| --- | --- | --- |
| Multinomial Naive Bayes | Lowercased bag-of-words counts from `CountVectorizer` | Selects smoothing `alpha` from 0.1, 0.5, 1.0, and 2.0 using validation macro-F1 |
| Logistic Regression | Lowercased TF-IDF features from `TfidfVectorizer` | Selects `C` from 0.1, 1.0, and 10.0 using validation macro-F1; L-BFGS is run for at most 1,000 iterations |
| BiLSTM with GloVe | A vocabulary built from each training split, initialized with 100-dimensional GloVe 6B vectors; sequences are padded or truncated to 30 tokens | Bidirectional LSTM with hidden size 128 and dropout 0.3; embeddings are fine-tuned; Adam (learning rate 0.001), batch size 32, and 10 epochs |
| DistilBERT | `distilbert-base-uncased` tokenizer with padding/truncation to 64 tokens | Six-class sequence classifier fine-tuned for 4 epochs with learning rate 2e-5, weight decay 0.01, training batch size 16, and evaluation batch size 32; the best checkpoint is selected by validation macro-F1 |

## Experimental design

### Data and splits

- The notebook downloads `CogComp/trec` through Hugging Face Datasets and uses
  the dataset's `coarse_label` target.
- It constructs nominal 100-, 500-, and 1,000-example settings with random seed
  42. The nominal 100 setting samples 16 examples per class (96 total), and the
  nominal 500 setting samples 83 examples per class (498 total).
- The nominal 1,000 setting requests 166 examples per class, but the TREC
  training split contains only 86 `ABBR` examples. Because
  `make_balanced_subset` samples `min(len(x), n_per_class)` from each class, this
  setting is imbalanced and contains 916 examples.
- The unmodified full TREC training split is the fourth setting.
- Each setting is divided into 80% training and 20% validation partitions with
  a stratified split and random seed 42. For the nominal 1,000 setting, this
  produces 732 training examples and 184 validation examples.
- The official TREC test split is reused for final evaluation at every data
  setting. Hyperparameters are selected on validation macro-F1, not on the test
  split.

The notebook trains every model family at each of the four data settings. The
classical models search the hyperparameters shown above. The BiLSTM and
DistilBERT sections currently each specify a single candidate configuration,
while retaining the same validation-selection structure.

### Evaluation

The notebook records the following test metrics for every model and data
setting:

- **Accuracy**
- **Macro-averaged F1**, which weights each of the six classes equally
- **Expected Calibration Error (ECE)** using 10 confidence bins

It then plots test macro-F1 and ECE against training-set size. Reliability
diagrams are produced for the nominal 500-example and full-data settings. The
final analysis also creates a DistilBERT confusion matrix for the full-data run
and lists its ten highest-confidence misclassifications. Results are generated
at execution time and are intentionally not reproduced as fixed claims here.

## Repository contents

| Path | Purpose |
| --- | --- |
| `CPSC_440_Final_Project.ipynb` | Original course-project notebook containing data preparation, training, evaluation, and plots |
| `README.md` | Project scope, experimental design, and reproduction instructions |
| `requirements.txt` | Python packages imported or required by the notebook |

Training creates local DistilBERT checkpoint directories named
`distilbert_trec_<setting>_<learning-rate>`. The BiLSTM section downloads and
extracts the Stanford GloVe 6B archive in the working directory if
`glove.6B.100d.txt` is absent. These generated artifacts are not part of the
repository.

## Setup

The notebook downloads TREC, GloVe, and pretrained DistilBERT assets, so its
first complete run requires an internet connection and sufficient disk space.
Python 3.10 or later is recommended.

### Google Colab

1. Open `CPSC_440_Final_Project.ipynb` in Colab.
2. For the neural and transformer sections, select a GPU runtime from
   **Runtime > Change runtime type**.
3. Run the cells in order from the beginning. The notebook installs its
   Datasets constraint and transformer dependencies in dedicated setup cells.

### Local environment

Create an isolated environment from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate       # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Then start a notebook interface already available in your environment (for
example, JupyterLab or the VS Code notebook editor), open
`CPSC_440_Final_Project.ipynb`, and run all cells in order. If needed, install a
local interface separately, for example with `python -m pip install jupyterlab`.
The notebook's GloVe download cell also expects the command-line tools `wget`
and `unzip`; alternatively, download and extract the GloVe 6B archive manually
so that `glove.6B.100d.txt` is in the repository root.

Colab or another **GPU-enabled environment is recommended** for the BiLSTM and
DistilBERT experiments. The notebook selects CUDA automatically when available,
but a complete CPU run—especially transformer fine-tuning across all four data
settings—can be slow. Execute cells sequentially because later plots and error
analysis use models, predictions, and result tables retained in memory by
earlier cells.

## Reproducibility notes

- Subset sampling and stratified train/validation splitting use seed 42. The
  nominal 100 and 500 subsets contain 96 and 498 examples, respectively; the
  nominal 1,000 subset is imbalanced because only 86 `ABBR` training examples
  are available, giving 916 examples before its 732/184 train/validation split.
- The notebook does not set NumPy or PyTorch training seeds, so neural-model
  results may vary between runs and hardware configurations.
- Package versions are left mostly unconstrained to reflect the notebook, except
  for `datasets<4.0.0`, which matches its explicit setup cell.
- No notebook cells or stored outputs were changed as part of this documentation
  update, preserving the submitted workflow and its academic provenance.
