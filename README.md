# Jonathan Markey — Data Science Projects

Welcome to my project portfolio. Below are selected data science and machine learning projects demonstrating my skills in deep learning, NLP, financial analysis, GIS, and simulation modeling. Each project is hosted in its own GitHub repo and includes end-to-end implementation, evaluation, and documentation.

### [**The Effect of Cosine Similarity on SBS Signature Extraction and Validation via the Linear Combination Model**](https://github.com/Jmarkey11/The-Effect-of-Cosine-Similarity-on-SBS-Signature-Extraction-and-Validation)

- **Summary:** Investigates how cosine similarity affects SBS signature extraction in cancer genomes using Non-Negative Matrix Factorization (NMF) and evaluates the Linear Combination Model (LCM) as an alternative for more reliable identification.  
- **Methods Used:** NMF for pattern extraction, Monte Carlo bootstrap resampling for robustness, cosine similarity clustering, regression analysis, and LCM for improved signature identification.  
- **Technologies Used:** Python (NumPy, Pandas, Scikit-learn, SciPy, Statsmodels, Seaborn).

### [**Military Aircraft Image Classification Using Custom, Pretrained & Ensemble Deep Learning Models**](https://github.com/Jmarkey11/Military-Aircraft-Image-Classification/tree/main)

- Summary: This project tackles fine-grained image classification across 81 military aircraft classes using the Kaggle Military Aircraft Detection Dataset. It compares the performance of a custom CNN, a custom ResNet-like model, and a pretrained ResNet-18 model, and introduces a Meta MLP ensemble to combine model predictions for improved top-k accuracy.
- Methods Used: Deep learning (CNNs, ResNet, ensemble MLP), image preprocessing, label smoothing, stratified data splitting, top-k accuracy evaluation, and transfer learning with ImageNet weights.
- Technologies Used: Python (PyTorch, Torchvision, NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn)

### [**Unprofitable NDIS Charity Detection & Analysis Bot**](https://github.com/Jmarkey11/Unprofitable-NDIS-Charity-Web-Scraping-And-Analysis)
- Summary: This automation bot identifies unprofitable NDIS-registered charities (net surplus ≤ 4% of revenue) by scraping financial data from the ACNC website, analyzing charity records, and generating detailed HTML reports. The pipeline includes geolocation mapping, database integration, and interactive dashboards to support decision-making.
- Methods Used: Web scraping with Selenium, Supabase API integration, multi-threaded data collection, financial ratio analysis, geolocation mapping with OpenCage API, and interactive reporting using Plotly and Folium.
- Technologies Used: Python (Selenium, BeautifulSoup4, Pandas, Plotly, Folium, Supabase, Geopy, Markdown, Glob2)

### [**Apple Stock Price Prediction Using Ensemble RNN Models**](https://github.com/Jmarkey11/Stock-Price-Prediction-Using-Ensemble-RNN-Models)

- **Summary:** Developed an ensemble model combining RNN, GRU, and LSTM networks to predict Apple’s stock price, leveraging multiple sequence lengths and a Single-Layer Perceptron (SLP) for optimal weighting.  
- **Methods Used:** RNN, GRU, and LSTM models trained with MSE loss and SGD optimizer, ensemble learning using SLP, feature engineering (moving averages, Bollinger Bands), and evaluation using MSE, MAE, and RMSE.  
- **Technologies Used:** Python (PyTorch, NumPy, Pandas, Matplotlib, yfinance, Scikit-learn).

### [**Information-Retrieval Based Question and Answer System**](https://github.com/Jmarkey11/Article-Question-and-Answer-System)  

- **Summary:** Developed an IR-based QA system that extracts direct answers from news articles using NLP techniques, focusing on fact-based questions. It combines question processing, passage retrieval, and answer extraction, with comparative evaluation against a BERT model.  
- **Methods Used:** NLP techniques (NER, dependency parsing, coreference resolution), TF-IDF for passage retrieval, cosine similarity, and proximity-based filtering for answer extraction. BERT is used for model comparison.  
- **Technologies Used:** Python (Pandas, NumPy, NLTK, SpaCy, Coreferee, Matplotlib, Seaborn, Sklearn, Transformers, PyTorch).

### [**Spotify Music Genre Popularity Analysis and Prediction**](https://github.com/Jmarkey11/Music-Genre-Analysis-and-Prediction)  

- **Summary:** Analyzed and predicted song genres using Spotify data, exploring how features like danceability, speechiness, and tempo influence classification. Investigated genre popularity trends and differences in speechiness.  
- **Methods Used:** Machine learning models (Random Forest, KNN, LDA) for genre prediction, EDA with visualizations and summary statistics, trend analysis using linear regression, and evaluation using AUC, sensitivity, and specificity.  
- **Technologies Used:** R (tidyverse, ranger, parsnip, pROC, lubridate, dplyr, kknn, tune).

### [**Lawnmower Repair Discrete Event Simulation**](https://github.com/Jmarkey11/Lawnmower-Repair-Discrete-Event-Simulation)  

- **Summary:** Simulated a lawnmower production line in Julia, focusing on the impact of machine breakdowns on efficiency. Analyzed production downtime, interrupted orders, and the effect of improved machine reliability.  
- **Methods Used:** Discrete event simulation modeling order arrivals, service times, and machine failures; exponential and deterministic distributions for timing events; test harness with 30 simulations per reliability factor.  
- **Technologies Used:** Julia (CSV, Printf, Dates, DataStructures, Distributions, StableRNGs).  

---
# 💻 Data ScienceSkills and Proficiencies

## 🧠 Key Skills at a Glance  
`Python` · `R` · `SQL` · `Julia` · `Java`  
`Pandas` · `NumPy` · `Scikit-learn` · `PyTorch` · `Torchvision` · `Transformers` · `SciPy` · `Statsmodels`  
`CNN` · `RNN` · `GRU` · `MLP` · `ResNet` · `BERT` · `Transfer Learning` · `Ensemble Models` · `Meta Learning`  
`Regression Analysis` · `Classification` · `Clustering` · `Dimensionality Reduction` · `PCA` · `Cosine Similarity` · `NMF`  
`Text Preprocessing` · `NER` · `POS Tagging` · `TF-IDF` · `Topic Modeling` · `Coreference Resolution`  
`Selenium` · `BeautifulSoup4` · `Supabase` · `OpenCage API` · `Glob2`  
`Matplotlib` · `Seaborn` · `Plotly` · `Folium` · `Markdown` · `Power BI` · `Tableau` · `Jupyter Notebooks`  
`ArcGIS` · `GeoPandas` · `ArcPy` · `Geopy`  
`EDA` · `Monte Carlo Resampling` · `Top-k Accuracy` · `Confusion Matrix Analysis`  
`HTML Report Automation` · `Interactive Dashboards` · `Simulation Modeling`  
`Analytical Thinking` · `Stakeholder Communication` · `Financial Analysis` · `Team Collaboration`

### 🛠️ Data Engineering & ETL

* Proficient in **SQL** for querying and managing multi-table databases and optimizing queries.
* Developed custom ETL pipelines using **Python**, **Pandas**, and **NumPy**.
* Automated data ingestion from external websites using **Selenium** and **BeautifulSoup**.
* Integrated real-time online databases using **Supabase** for data storage and retrieval.
* Converted large-scale image datasets into `.pt` tensors to optimize **PyTorch** model training.
* Experienced in pipeline development across structured, unstructured, and image-based data.

**Tools & Libraries:**
`SQL · Supabase · Selenium · BeautifulSoup4 · Pandas · NumPy · SQLite · Glob2`


### 🤖 Machine Learning & Predictive Modeling

* Built classification, regression, and clustering models using **Scikit-learn** and **SciPy**.
* Implemented **ensemble models** including stacked learners and a Meta MLP for combining deep learning outputs.
* Applied model optimization techniques such as **hyperparameter tuning** with Grid Search and Random Search.
* Validated models using **cross-validation**, **top-k accuracy**, and stratified sampling for imbalanced datasets.

**Algorithms:**
`Random Forest · SVM · KNN · Logistic Regression · Gradient Boosting · HDBSCAN · Decision Trees · Naïve Bayes · K-Means`


### 🧠 Deep Learning & Computer Vision

* Designed and trained custom CNN and ResNet-like architectures in **PyTorch**.
* Used **transfer learning** with pretrained **ResNet-18** for fine-grained image classification (81 aircraft classes).
* Built a **Meta MLP ensemble model** to combine softmax outputs and improve Top-5 accuracy.
* Applied regularization and optimization techniques: **dropout**, **batch normalization**, **label smoothing**, **early stopping**, **learning rate scheduling**, and **weight decay**.

**Architectures & Tools:**
`CNN · RNN · GRU · MLP · Transformer Models · PyTorch · Torchvision · CUDA`


### 📚 Natural Language Processing (NLP)

* Built end-to-end pipelines using **SpaCy**, **NLTK**, and **Scikit-learn** for text analysis and information retrieval.
* Preprocessed raw text using tokenization, lemmatization, stemming, stopword removal, and **TF-IDF**.
* Applied advanced techniques: **Named Entity Recognition (NER)**, **POS tagging**, **Dependency Parsing**, and **Coreference Resolution**.
* Developed QA systems and compared classical IR techniques with transformer-based models like **BERT**.

**Applications:**
`Text Classification · Semantic Search · Summarization · Sentiment Analysis · Topic Modeling`


### 🌍 GIS & Spatial Analysis

* Experienced with **ArcGIS**, **ArcPy**, and **GeoPandas** for geospatial workflows.
* Created interactive geospatial visualizations using **Folium** and **Geopy**.
* Automated geolocation mapping with **OpenCage API** to enrich scraped charity datasets.
* Performed spatial joins and visualized location-based insights to support decision-making.

**Libraries & Tools:**
`ArcPy · GeoPandas · Folium · OpenCage · Geopy · Pandas · NumPy`


### 📊 Data Visualization & Dashboards

* Built static and interactive visualizations using **Matplotlib**, **Seaborn**, and **Plotly**.
* Designed HTML dashboards and dynamic reports using **Markdown**, **Plotly**, and **Folium**.
* Experienced in developing business-ready dashboards using **Power BI** and **Tableau**.
* Communicated technical insights using **classification reports**, **confusion matrices**, **top-k accuracy plots**, and **geographic maps**.

**Tools:**
`Matplotlib · Seaborn · Plotly · Power BI · Tableau · Markdown · Jupyter Notebooks`


### 🧮 Statistical Analysis & Feature Engineering

* Performed statistical modelling, data cleaning, exploratory data analysis, and feature engineering.
* Applied dimensionality reduction techniques like **PCA** and evaluated **feature importance**.
* Used **regression analysis**, **cosine similarity**, and **cluster evaluation** (e.g. silhouette width) in research projects.

**Techniques:**
`EDA · PCA · Cosine Similarity · Monte Carlo Resampling · NMF · Clustering Evaluation`


### 🧾 Business & Financial Analysis

* Strong background in financial analysis and reporting from prior experience in accounting.
* Built automated financial reporting systems to evaluate **NDIS charity profitability** (Net Surplus ≤ 4%).
* Applied **financial ratio analysis** to assess organizational sustainability.
* Communicated insights through clear stakeholder-ready outputs (dashboards, HTML reports, CSV exports).

**Skills:**
`Financial Modelling · Ratio Analysis · Automated Reporting · Stakeholder Communication`


### 🤝 Soft Skills & Communication

* Strong **analytical thinking**, **communication**, and **team collaboration** skills.
* Comfortable translating technical results for non-technical stakeholders.
* Experienced in producing clear documentation and visual storytelling for varied audiences.
