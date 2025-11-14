# AI-Fundamentals
Comprehensive AI learning program covering linear algebra, probability, statistics, data preprocessing with NumPy/Pandas, supervised/unsupervised machine learning, NLP, and deep learning with TensorFlow/Keras.

## Chapter 1. Exploratory data analysis
1. numpy_basics: This notebook covers the basics of the NumPy library in Python, including how to create, index, slice, reshape, and modify NumPy arrays.
2. numpy_operations: This notebook demonstrates fundamental NumPy array operations, including arithmetic operations, universal functions, and statistical methods.
3. numpy_indexing_boolean: This notebook demonstrates various NumPy array indexing techniques, including basic, slicing, fancy, and boolean indexing.
4. numpy_array_transposition: This notebook demonstrates how to use NumPy array transposition functions, including .T, np.dot, .transpose(), and .swapaxes().
5. pandas_series: This notebook contains examples and exercises on Pandas Series, including creation, attributes, indexing, and basic operations on Series.
6. pandas_dataFrame_basics: This notebook explores the fundamentals of Pandas DataFrames, including creation, indexing, slicing, file I/O, and basic arithmetic operations.
7. pandas_dataframe_manipulation_and_joins: This notebook demonstrates basic DataFrame manipulation in pandas, including adding/removing columns, merging, and concatenating DataFrames.
8. exploratory_data_analysis: This notebook analyzes the Iris dataset using pandas, covering basic statistics and handling missing values.
9. pandas_multiindex_groupby_pivot: This notebook demonstrates how to work with MultiIndex DataFrames, perform groupby operations, and use pivot tables in pandas for data summarization and analysis.
10. basic_matplotlib_visualizations: This notebook contains basic examples of data visualizations using the Matplotlib library, including bar plots, histograms, box plots, and scatter plots.
11. matplotlib_subplots: This file contains a Matplotlib figure with multiple subplots, demonstrating different types of visualizations including a histogram, a scatter plot, a bar plot, and a line plot.
12. matplotlib_pandas_seaborn_visualization: This notebook provides examples of basic data visualizations using the Matplotlib and Seaborn libraries in Python, covering various plot types like bar plots, histograms, box plots, line plots, scatter plots, and heatmaps.
13. pandas_matplotlib_air_quality: This notebook explores air quality data using Pandas and Matplotlib to create various plots including bar plots, histograms, and scatter plots.
14. data_visualization: This notebook provides basic examples of data visualization using Matplotlib, Pandas, and Seaborn.
## Chapter 2. Statistical Analysis & Probability Distributions
1. discrete_probability_distribution: This notebook covers Discrete probability distribution, focusing on Binomial.
2. continuous_probability_density: This notebook covers Continuous probability density, focusing on Uniform.
3. descriptive_statistics: This notebook covers Descriptive statistics, focusing on Sample statistics.
4. quantile: This notebook covers Quantile, focusing on Standard Normal.
5. interval_estimation_of_the_mean: This notebook covers Interval Estimation of the Mean, focusing on 90% Confidence Interval.
6. interval_estimation_of_the_proportion: This notebook covers Interval Estimation of the Proportion.
7. correlation: This notebook covers Correlation, focusing on Pearson.
8. hypothesis_test_of_the_means: This notebook covers Hypothesis Test of the Means, focusing on One sample t-test.
9. summarizing_categorical_variables: This notebook covers Summarizing categorical variables, focusing on Frequency table (One way table).
10. chisquared_tests: This notebook covers Chi-squared Tests, focusing on For one way table.
11. chisquared_test_of_variance: This notebook covers Chi-squared test of variance, focusing on Using the acceptance range (region).
12. f_test_of_variance_ratio: This notebook covers F test of variance ratio, focusing on Using the p-value.
## Chapter 3. Machine Learning - Supervised Learning
1. linear_regression_with_galton_data: This notebook covers Linear regression with 'Galton' data, focusing on Read in data and visualize.
2. linear_regression_and_diagnostics: This notebook covers Linear regression and diagnostics, focusing on Load the 'Boston' dataset from Scikit-Learn.
3. linear_regression_diagnostics_and_modeling_using_statsmodels_library: This notebook covers Linear regression diagnostics and modeling using StatsModels library, focusing on Load the 'Boston' dataset from Scikit-Learn and convert it into a DataFrame.
4. linear_regression_prediction_and_confidence_interval: This notebook covers Linear regression prediction and confidence interval, focusing on Data.
5. dummy_variable_and_interaction: This notebook covers Dummy variable and interaction, focusing on Read in the data.
6. regularized_regressions: This notebook covers Regularized regressions, focusing on Read in data.
7. classification_with_logistic_regression: This notebook covers Classification with logistic regression, focusing on Read in data and explore.
8. classification_with_tree: This notebook covers Classification with Tree, focusing on Read in data.
9. calssification_with_naive_bayes: This notebook covers Calssification with Naive Bayes, focusing on Read in data and explore.
10. classification_with_knn: This notebook covers Classification with KNN, focusing on Read in data.
11. classification_with_svm: This notebook covers Classification with SVM, focusing on Read in data.
12. compare_the_treelike_algorithms: This notebook covers Compare the Tree-like algorithms, focusing on Read in data and explore.
13. voting_ensemble: This notebook covers Voting Ensemble, focusing on Read in data.
14. bagging_ensemble: This notebook covers Bagging ensemble, focusing on Read in data.
## Chapter 4. Clustering
1. kmeans_simulated_data.ipynb: This notebook demonstrates K‑Means clustering on simulated data, covering data generation, model fitting, and visualization of clusters.
2. kmeans_real_data.ipynb: This notebook applies K‑Means clustering to a real dataset (e.g., Iris), illustrating preprocessing steps, algorithm application, and result interpretation.
3. compare_clustering_algorithms.ipynb: This notebook compares multiple clustering algorithms—K‑Means, Agglomerative, Hierarchical, DBSCAN—highlighting their differences through visualizations and performance metrics.
4. dimensional_reduction_pca.ipynb: This notebook explores dimensionality reduction using Principal Component Analysis (PCA), including data loading, PCA fitting, variance explanation, and projected data visualization.
5. numpy_basics.ipynb: This notebook covers the basics of the NumPy library in Python, including how to create, index, slice, reshape, and modify NumPy arrays.
## Chapter 5.Natural language processing and language modeling in text mining
1. String_and_RegularExpressions.ipynb: This notebook covers Python string functions and regular expression basics. It demonstrates common string methods (e.g., split, replace), regex metacharacters such as dot, question mark, curly braces, caret, dollar sign, alternation, grouping, and the re.sub() function for pattern replacement.
2. Advanced_Regex_Examples.ipynb: This notebook provides advanced regular expression examples using Python's re module. It shows how to read data, define helper functions, and apply a custom grep-like function to search patterns within text.
3. Regex_Groups_and_Capturing.ipynb: This notebook focuses on regex grouping. It illustrates capturing groups, group numbering, shorthand expressions, and extracting matched substrings from input strings.
4. WebScraping_RequestBS4.ipynb: This notebook demonstrates web data acquisition using the Requests library and BeautifulSoup4 to scrape content from a webpage. It includes fetching HTML, parsing with BS4, and extracting specific elements.
5. Twitter_Data_Fetching_and_Preprocessing.ipynb: This notebook fetches tweets from Twitter via the API or web scraping. It shows how to retrieve tweet data, handle pagination, and preprocess the text for analysis.
6. NLTK_Tokenization_Stemming_Lemmatization.ipynb: This notebook introduces the NLTK library for natural language processing tasks such as tokenization, stemming, and lemmatization. It demonstrates how to break sentences into tokens and normalize words.
7. NLTK_StopWords_POS_Tagging.ipynb: This notebook continues with NLTK, covering stop word removal and part-of-speech tagging. It shows how to filter common words and assign grammatical tags to tokens.
8. WordCloud_Text_Visualization.ipynb: This notebook visualizes text data as a WordCloud. It includes preprocessing steps, generating basic wordclouds, and creating masked wordclouds using custom shapes.
9. Integer_Encoding_with_Counter_NLTK_Keras.ipynb: This notebook explores integer encoding techniques for words using Python's Counter, NLTK FreqDist, enumeration, and Keras utilities. It demonstrates mapping tokens to integer indices for model input.
10. Sequence_Padding_with_Keras.ipynb: This notebook shows how to apply padding to sequences with Keras' preprocessing tools, ensuring uniform input lengths for neural network models.
11. OneHot_Encoding_TensorFlow_Keras.ipynb: This notebook demonstrates one-hot encoding of text using TensorFlow's Keras Tokenizer and the to_categorical function. It converts words into binary vectors suitable for machine learning models.
12. NGram_Autofill_Prediction.ipynb: This notebook implements n-gram based autofill, training a dictionary of n-grams from sample text and predicting subsequent words or word sequences using the model.
13. TFIDF_Cosine_Similarity.ipynb: This notebook creates TF-IDF representations for documents, computes cosine similarity between vectors, and demonstrates how to compare document relevance.
14. NLP_Classification_LogisticRegression.ipynb: This notebook performs NLP classification analysis. It reads data, preprocesses it, generates a TF-IDF representation, and trains a logistic regression classifier for text categorization.
15. LatentSemanticAnalysis_LSA.ipynb: This notebook applies Latent Semantic Analysis (LSA) to document-term matrices using truncated SVD, extracting top features per topic and labeling documents with dominant topics.
16. LatentDirichletAllocation_LDA.ipynb: This notebook applies Latent Dirichlet Allocation (LDA) to document-term matrices, extracting topic distributions and identifying key terms for each topic.
17. Document_Classification_LSTM.ipynb: This notebook trains an LSTM network for document classification. It includes data loading, preprocessing, model definition, training, and evaluation.
18. Document_Classification_LSTM_Binary.ipynb: This notebook trains a binary LSTM classifier for document categorization, covering data preparation, model construction, training, and testing.
19. Document_Classification_LSTM_CNN_Binary.ipynb: This notebook trains a combined LSTM+CNN binary classifier for document classification, including data handling, model architecture, training, and evaluation.
20. WordEmbedding_Word2Vec_and_Pretrained.ipynb: This notebook demonstrates word embedding using Word2Vec, training a model on text data, extracting embeddings, finding similar words, and utilizing a pre-trained Google model.
