# Fashion Item Classification with Deep Learning & Clustering

This project tackles the challenge of fashion item classification by implementing and comparing two fundamentally different machine learning methodologies. The goal is to categorize apparel images into broad classes such as shoes, clothing, and accessories. The repository showcases both a modern end-to-end Supervised Deep Learning approach using PyTorch, and a clever Semi-Supervised Clustering pipeline designed to handle datasets with a mix of labeled and unlabeled data. This dual approach not only demonstrates proficiency in state-of-the-art deep learning but also showcases practical strategies for data labeling and analysis using classic machine learning with Scikit-learn.

**For a detailed, step-by-step walkthrough of the entire project, including data exploration, model building, and results analysis, please see the main Jupyter Notebook:**
**🚀 [Fashion_Classification_Full_Analysis.ipynb](./Fashion_Classification_Full_Analysis.ipynb)**

---

## 📂 Project Structure

```
.
├── models/                  # Contains all model architecture definitions
├── utils/                   # Contains data loaders for both approaches
├── data_preprocess.py       # Script for initial data preprocessing
├── Supervised Deep Learning Training.py      # Approach 1: End-to-end supervised training
├── Semi-Supervised Clustering and Labeling.py # Approach 2: Clustering + SVM for labeling
├── Fashion_Classification_Full_Analysis.ipynb  # Main detailed notebook
└── README.md                # You are here
```

---

## 🛠️ Approaches Implemented

### 1. Supervised Deep Learning (`Supervised Deep Learning Training.py`)
- An end-to-end deep learning pipeline using **PyTorch**.
- It trains various neural network models (like ResNet, custom CNNs) to classify images directly.
- The data is loaded using `utils/data_loader_for_Supervised.py`.

### 2. Semi-Supervised Clustering (`Semi-Supervised Clustering and Labeling.py`)
- A hybrid approach using **Scikit-learn**.
- It first extracts image features, then uses **DBSCAN clustering** to group similar images.
- Finally, it trains an **SVM classifier** within each cluster using any available labels to classify the rest.
- The data is loaded using `utils/data_loader_for_Semi-Supervised.py`.

---

## 🚀 How to Run

### 1. Setup
First, clone the repository and install the required dependencies. It's recommended to use a virtual environment.

```bash
# Clone the repository
git clone https://github.com/SpicyyMath/Fashion-Item-Classification
cd Fashion-Item-Classification

# Install dependencies
pip install -r requirements.txt
```

*(Note: You will need to download the UT Fashion 100 dataset and place it in a `Data/` folder within the project root.)*

### 2. Running the Supervised Model

You can run the training script with various arguments. For example:

```bash
python "Supervised Deep Learning Training.py" --model_name Newmodel --task A --epochs 50
```

### 3. Running the Semi-Supervised Model

This script can be run directly:

```bash
python "Semi-Supervised Clustering and Labeling.py"
```
