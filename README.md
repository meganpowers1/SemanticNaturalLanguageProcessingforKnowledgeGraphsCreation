# SemanticNaturalLanguageProcessingforKnowledgeGraphsCreation

# SemanticNaturalLanguageProcessingforKnowledgeGraphsCreation

5.1. Prototype Case
This section contains an overview of the model prototype, starting from importing
text and finishing with the final ontology. This overview is presented in steps.
To begin with, the sample ontology corpus is saved into a .txt file and imported
into a Jupyter Notebook. Stopwords are removed from the dataset, and it is tokenized,
stemmed, and lemmatized. This process is shown in Figures 6, 7, 8, 9 and 10.
The next stage in the process is the Named Entity Recognition stage. First, the
spaCy en core web sm model is imported, trained, and tested. The test results of the
trained en core web sm model are compared to a default en core web sm model. This
process is illustrated in Figures 11, 12, 13, 14, 15 and 16.
After this stage, the trained NER model is used to extract Named Entities (NEs).
Basic dependency visualisation is done. These steps are shown in Figures 17 and 18.
The next step involves importing a vectorizer to convert text to numerical vectors
for further processing, and a model is built for further exploratory analysis, including
TFIDF analysis to determine the most important words, visualising which words are
the most similar, and Agglomerative and K-means clustering to identify clusters of
terms. This stage is depicted in Figures 19, 20, 21, 22, 23, 24, 25 and 26.
29
Figure 6. Overview of the Heliophysics .txt file
![alt text](http://url/to/img.png)

Figure 7. Opening and Saving the Dataset
30
Figure 8. Removing Stopwords from the Corpus
Figure 9. Tokenizing the Corpus
31
Figure 10. Stemming and Lemmatizing the Corpus
Figure 11. Importing spaCy
32
Figure 12. Training Data and Custom Labelling
33
Figure 13. Training the en core web sm Model
Figure 14. Testing Data for the Trained en core web sm Model
34
Figure 15. Evaluation Metrics for Trained en core web sm Model
Figure 16. Evaluation Metrics for Untrained en core web sm Model
35
Figure 17. Employing Trained en core web sm Model to Identify Named Entities (NEs)
Figure 18. Employing Trained en core web sm Model to Identify Named Entities (NEs)
37
Figure 19. TFIDF Vectorizing the Heliophysics Text Corpus
38
Figure 20. Building Word2Vec Model for Text Analysis
39
Figure 21. Display Words by Highest TFIDF Value
Figure 22. Creating Word2Vec Model
40
Figure 23. Using Model to Plot Most Similar Words to ’Solar’
41
Figure 24. Import and Use Agglomerative Clustering to Visualise Clusters of Similar Terms

Figure 25. Result of Using Agglomerative Clustering to Display Total Number of Clusters
43
Figure 26. Importing K-Means Clustering to Visualise Clusters of Similar Terms
44
The next stage is to extract the Named Entities (NEs) and relationships by using
Part of Speech (PoS) tagging in order to convert them into RDF triples. Then, ex-
traneous characters and words are removed from the list. This process is displayed in
Figures 27, 28, 29 and 30.
Figure 27. Define Function get entities to Construct Entities
The next stage is to prepare the entities and relations for the Knowledge Graph
and for processing into RDF triples. The sources (subjects), targets (objects), and
edges (relationships) are stored in a dataframe, and a directed graph is created. As
this graph is large, a sample displaying nodes matching a specific relation is displayed.
This process is visualised in Figures 31, 32, 33 and 34 .
The last stage is to convert the sources, targets, and edges into RDF triples and
45
Figure 28. Use get entities to Extract Entity Pairs
insert them into an ontology. Class and namespace properties were defined, and triples
were added to the graph. Additionally, labels corresponding to the triples were added
to the graph. Afterwards, the graph was serialised to an OWL file. This is visualised
in Figures 35, 36, 37 and 38.
Then, the resulting ontology can be visualised in Prot ́eg ́e. This includes Axioms,
Classes, Instances, and Object Properties. An overview of the automatically generated
ontology is displayed in Figures 39, 40, 41 and 42.
5.2. Exploratory Analysis
5.2.1. NER Performance Metrics
The metrics chosen to evaluate the Named Entity Recognition (NER) portion of the
model were Precision, Recall, and the F1-score. Precision, as is described in 3.1, is
a measure of whether a classifier successfully does not label a positive sample as
negative. Recall, as is described in 3.2, is a measure of whether a classifier successfully
finds all positive samples. The F1-score, as is described in 3.3, is the harmonic mean of
Precision and Recall. It is used to provide an average of the performances of Precision
and Recall (81).
It must be noted that the training corpus has been run twice, as spaCy flags an
error when NEs span multiple words. Table 4 shows the results of running the per-
formance metrics for the en core web sm model that has been trained and tested on
heliophysics data. Meanwhile, Table 5 shows the same metrics generated from an un-
trained en core web sm model that has not had the custom labels appended. If the
metrics were calculated from an untrained en core web sm model with the custom
46
Figure 29. Define and Use Function get relation to Extract Relations Between Entities
labels, the scores would be 0, as spaCy would have no frame of reference to extract
entities, and spaCy needs in-context training information (28). This demonstrates the
utility of transfer learning, where less training was necessitated due to the knowledge
already contained in spaCy en core web sm.
Table 4. Performance Metrics for Trained en core web sm Model
Evaluation Metrics
Precision 100%
Accuracy 100%
Recall 66.6%
F1-Score 80%
5.2.2. NER Word Embeddings and Clustering
After running performance metrics on the NER model, there are additional metrics
that can be applied to the text corpus in order to provide further insight. Term Fre-
47
Figure 30. Remove Extraneous Words and Characters from Entity Pairs and Relations
Table 5. Performance Metrics for untrained en core web sm Model
Evaluation Metrics
Precision 0%
Accuracy 100%
Recall 0%
F1-Score 0%
quency - Inverse Document Frequency (TF-IDF) is a model that can be used for text
to numeric conversion, in order to identify the most important words in a corpus (71).
It assigns higher value to certain words over others, so that even important words
that occur infrequently are assigned high weights. A dictionary of words was created
from the vectorized heliophysics text corpus, and was sorted using TF-IDF in order to
produce words with the highest weights, and therefore, importance. Table 6 displays
the top 10 words and their TFIDF scores.
Another tool is word embeddings, which is a language modelling method that is used
to map words to vectors that consist of real numbers. Words that occur in similar
contexts should hypothetically be closer to each other in vector space. Therefore,
related words in the ontology can be extracted by using word embeddings. Word2vec
48
Figure 31. Extract Corresponding Labels
Figure 32. Create Pandas Dataframe
is an algorithm that uses shallow neural networks for word embeddings, and it can be
used to represent words as vectors (93). A Word2Vec model for the heliophysics corpus
was constructed and trained. Dimensionality reduction algorithms transform data with
a high number of dimensions, such as images, into a lower amount of dimensions
(89). This allows for the interpretation of relationships between vectors extracted
from the heliophysics text corpus. One such example is determining the top-K most
similar words to a particular term, and was applied to the word ‘solar’. The results
are displayed in text and graph format in Table 7.
49
Figure 33. Define and Display Graph
A final tool that provides insight into the text is clustering. In Machine Learning
(ML), clustering is an unsupervised technique that groups entities based on similar
features. Clustering can be used to discover hitherto unknown patterns in data and
can use several different measures for calculating distance (7). Hierarchical clustering,
which divides data into clusters without manually specifying a number of clusters, was
initially applied to the TF-IDF features to determine how the algorithm would divide
up the data. The output of the algorithm is displayed in Figure 43, and demonstrates
that the text corpus can be separated into three clusters.
The hierarchical clustering algorithm demonstrated that the text corpus separates
into three clusters. Choosing an appropriate number of clusters is particularly impor-
tant for K-means clustering. The utility of applying K-means clustering to the data is
that it can determine what values are assigned to which clusters. It aims to separate n
data values into K clusters, where K=3 as was established in Figure 43. The clusters
generated by K-means are visualised in Table 8.
50
Figure 34. Display Graph for Edge ”by”
