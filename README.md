Business Entity Resolution
An end-to-end Machine Learning-based Business Entity Resolution system that identifies whether business records from different data sources refer to the same real-world business.
The project combines data preprocessing, entity normalization, blocking/candidate generation, similarity feature engineering, machine learning, model evaluation, and an interactive web application.
📌 Project Overview
Business information is often stored across multiple systems, and the same business may appear with different:
- Names
- Addresses
- Abbreviations
- Legal suffixes
- Punctuation
- Word ordering
- Typos
- Address formats
- Missing information
- Trade/DBA names
For example:
Source 1:
ABC Technologies Private Limited
25 Station Road, Delhi

Source 2:
ABC Technologies Pvt Ltd
25 Station Rd, Delhi

Source 3:
A.B.C. Technologies
25 Station Road Delhi

Although the records are different, they may represent the same real-world business.
The objective of this project is to automatically identify these relationships.
🎯 Problem Statement
Given three business datasets:
- Source 1 – reference business entities
- Source 2 – business records
- Source 3 – business records
the system identifies which Source 2 and Source 3 records correspond to each Source 1 entity.
A Source 1 entity can have:
0 matches
1 match
Multiple matches

The system therefore does not force every business to have exactly one match.
🚀 Key Features
Machine Learning Pipeline
- Dataset validation
- Exploratory Data Analysis
- Data preprocessing
- Business name normalization
- Address normalization
- Candidate generation
- Blocking
- Similarity calculation
- Feature engineering
- Machine Learning classification
- Threshold optimization
- Model comparison
- Final prediction
Web Application
The project includes an interactive interface with:
- Dashboard
- Project & Data
- Entity Map
- Matches
- Unmatched
- Model Lab
- Live Matcher
Live Matcher
Users can enter:
Business Name
Business Address
Country

and use the trained model to search for corresponding businesses in Source 2 and Source 3.
Bring Your Own Data
The application also provides an interface for uploading business TSV files and running the entity-resolution pipeline.
🧠 Machine Learning Approach
The overall pipeline is:
Raw Business Data
        ↓
Data Validation
        ↓
Exploratory Data Analysis
        ↓
Preprocessing
        ↓
Entity Normalization
        ↓
Candidate Generation / Blocking
        ↓
Similarity Features
        ↓
Model Training
        ↓
Validation
        ↓
Threshold Optimization
        ↓
Final Matching
        ↓
Submission / Results

🔹 1. Data Preprocessing
The raw business records are cleaned and normalized before comparison.
Typical operations include:
- Lowercasing
- Removing unnecessary punctuation
- Whitespace normalization
- Unicode normalization
- Business-name normalization
- Address normalization
- Missing-value handling
- Tokenization
- Numeric token extraction
This allows differently formatted records to be compared more effectively.
🔹 2. Candidate Generation
Comparing every Source 1 record with every Source 2 and Source 3 record would be computationally expensive.
Therefore, the project uses blocking/candidate generation.
All possible comparisons
          ↓
     Blocking
          ↓
Potential candidates
          ↓
Similarity calculation
          ↓
Machine Learning model

Candidate generation reduces the number of pairs that need to be evaluated while attempting to preserve the correct matches.
🔹 3. Feature Engineering
The system generates comparison features between two business records.
Business Name Features
Examples include:
- Exact normalized-name match
- Jaccard similarity
- String similarity
- Token overlap
- Name length difference
- Numeric token overlap
Address Features
Examples include:
- Exact address match
- Address string similarity
- Token overlap
- Numeric component comparison
- Address length difference
Country Features
- Country match
- Normalized country comparison
These features are combined into a feature vector for the Machine Learning models.
🤖 Machine Learning Models
The project uses supervised Machine Learning models to classify candidate pairs.
Logistic Regression
Used as an interpretable baseline model for predicting whether two records represent the same business.
The model produces a probability:
P(match | features)

Random Forest
A tree-based ensemble model used to learn nonlinear relationships between the entity-matching features.
It can capture interactions between different similarity features.
📊 Model Evaluation
The project evaluates the models using:
- Precision
- Recall
- F0.5 Score
The primary challenge metric is F0.5.
F0.5 =
(1.25 × Precision × Recall)
--------------------------------
(0.25 × Precision + Recall)

F0.5 places greater emphasis on precision than recall.
This is particularly relevant to entity resolution because incorrectly merging two different businesses can be costly.
🎚️ Threshold Optimization
The model does not simply assume:
Threshold = 0.50

Instead, different thresholds are evaluated on the validation data.
The threshold producing the desired validation performance according to the project evaluation procedure is selected for final prediction.
🔬 Validation Strategy
The validation process is designed to avoid Source 1 entity leakage.
The data is split at the Source 1 entity level, so records belonging to the same reference entity are not unnecessarily distributed between training and validation.
The validation process measures:
Precision
Recall
F0.5
Predicted Matches
Unmatched Entities

💾 Large Dataset and RAM Optimization
The dataset can be large, so the project is designed to avoid unnecessarily loading the complete dataset and all candidate pairs into RAM.
The memory-efficient workflow is:
Large TSV Files
       ↓
Chunked Reading
       ↓
Normalization
       ↓
Blocking
       ↓
Candidate Batches
       ↓
Feature Calculation
       ↓
Model Prediction
       ↓
Output Files

Important optimization techniques include:
- Chunk-based file processing
- Selecting only required columns
- Avoiding full Cartesian products
- Batch candidate generation
- Batch feature calculation
- Disk-backed intermediate results
- Controlled parallel processing
- Checkpointing intermediate stages
This helps prevent RAM exhaustion when working with large datasets.
🌐 Web Application
The project provides an interactive web interface named:
Business Entity Resolution
The application is designed to provide a visual interface for understanding and using the entity-resolution system.
Main Sections
Dashboard
Project & Data
Entity Map
Matches
Unmatched
Model Lab
Live Matcher

🔎 Live Matcher
The Live Matcher allows users to enter a business manually.
Example:
Business Name:
Jade Digital Marine Company

Business Address:
25 Station Road, Delhi

Country:
India

After selecting Resolve, the trained matching pipeline searches for corresponding records in Source 2 and Source 3.
📂 Bring Your Own Data
The application provides a data-upload interface where users can provide three TSV files containing business information.
Expected fields include:
entity_id
business_name
business_address
country

The uploaded data can then be processed through the entity-resolution pipeline.
📁 Dataset Structure
dataset/
│
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
│
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv

📄 Input Data Format
The source datasets contain:
Column	Description
entity_id	Unique identifier for the record
business_name	Name of the business
business_address	Business address
country	Country associated with the business


The files are stored in TSV format.
Example:
import pandas as pddf = pd.read_csv(    "train_source1.tsv",    sep="\t")


📤 Output Files
matching_results.tsv
Contains the final predicted matches.
Example:
source1_entity_id    matched_entity_ids
S1-001               S2-105,S3-204
S1-002
S1-003               S2-502

An empty matched_entity_ids value represents an entity for which no match was predicted.
candidate_pairs.tsv
Contains the candidate pairs generated during the blocking/candidate-generation stage.
The final matching stage operates only on these candidate pairs.
📊 Exploratory Data Analysis
EDA is used to understand the datasets before model training.
The analysis includes areas such as:
- Dataset size
- Column information
- Missing values
- Duplicate records
- Business-name patterns
- Address patterns
- Country distribution
- Text length distributions
- Data quality
For very large datasets, EDA can use chunked or disk-backed aggregation instead of loading everything into memory.
🛠️ Project Structure
business-entity-resolution/
│
├── dataset/
│   ├── train/
│   └── test/
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_candidate_generation.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_model_training.ipynb
│   └── 06_validation.ipynb
│
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── data_validation.py
│   ├── preprocessing.py
│   ├── blocking.py
│   ├── similarity.py
│   ├── feature_engineering.py
│   ├── ground_truth.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   ├── prediction.py
│   ├── submission.py
│   └── utils.py
│
├── models/
│
├── outputs/
│   ├── eda/
│   ├── validation/
│   ├── predictions/
│   └── submission/
│
├── app/
│   ├── backend/
│   └── frontend/
│
├── requirements.txt
├── README.md
├── PROJECT_IMPLEMENTATION.md
└── .gitignore

⚙️ Technologies Used
Programming
- Python
- JavaScript
Machine Learning
- pandas
- NumPy
- scikit-learn
- RapidFuzz / string similarity techniques
Data Processing
- TSV processing
- Data normalization
- Blocking
- Candidate generation
- Chunked processing
Frontend
- React
- Tailwind CSS / CSS
Development Tools
- Visual Studio Code
- Jupyter Notebook
- Git
- GitHub
🚀 Installation
Create a Python virtual environment:
python -m venv .venv

Activate it:
.\.venv\Scripts\Activate.ps1

Install dependencies:
pip install -r requirements.txt

▶️ Running the ML Pipeline
The general execution order is:
1. Data Validation
        ↓
2. EDA
        ↓
3. Preprocessing
        ↓
4. Candidate Generation
        ↓
5. Feature Engineering
        ↓
6. Model Training
        ↓
7. Validation
        ↓
8. Threshold Selection
        ↓
9. Test Prediction
        ↓
10. Submission Generation
        ↓
11. Submission Validation

The project notebooks can be used to understand and execute the individual stages.
🧪 Submission Validation
The generated output can be checked using the provided validation utility.
Example:
python utils/validate_submission.py `
    --matching output/matching_results.tsv `
    --candidate output/candidate_pairs.tsv `
    --test-dir dataset/test

The paths should be adjusted according to the actual project directory.
📌 Important Design Principles
1. Do not force a match
A business may have zero, one, or multiple matches.
2. Candidate generation is important
If a true matching pair is not generated as a candidate, the Machine Learning model cannot recover it later.
3. Precision matters
Incorrectly merging two different businesses can be more harmful than leaving a potential match unresolved.
4. Large datasets require memory-efficient processing
The system should avoid unnecessary full-data operations and Cartesian products.
5. Results must be reproducible
Model parameters, preprocessing rules, thresholds, validation results, and important pipeline configurations should be documented.
📚 Project Documentation
Detailed implementation information should be maintained in:
PROJECT_IMPLEMENTATION.md

This document should contain the actual:
- Dataset statistics
- EDA results
- Preprocessing steps
- Blocking strategy
- Candidate counts
- Features
- Model configurations
- Validation results
- Precision
- Recall
- F0.5
- Selected threshold
- Prediction statistics
- RAM optimization
- Submission validation
- Web application implementation
- Limitations
- Future improvements
All numerical results should represent actual results produced by the project, not manually created values.
🔮 Future Enhancements
Possible future improvements include:
- Advanced multilingual name normalization
- Improved transliteration handling
- More sophisticated blocking strategies
- Hard-negative mining
- Advanced entity embeddings
- Transformer-based entity matching
- Better threshold calibration
- Entity clustering
- Faster distributed processing
- Improved entity relationship visualization
- Batch matching API
- More detailed error analysis
👨‍💻 Project
Business Entity Resolution
An end-to-end Machine Learning project for detecting and matching duplicate representations of real-world businesses across multiple data sources.