# SemanticNaturalLanguageProcessingforKnowledgeGraphsCreation

This ReadMe contains an overview of the model prototype, starting from importing text and finishing with the final ontology. This overview is presented in steps. To begin with, the sample ontology corpus is saved into a .txt file and imported into a Jupyter Notebook. Stopwords are removed from the dataset, and it is tokenized, stemmed, and lemmatized. This process is shown in Figures 1, 2, 3, 4 and 5. The next stage in the process is the Named Entity Recognition stage. First, the spaCy en core web sm model is imported, trained, and tested. The test results of the trained en core web sm model are compared to a default en core web sm model. This process is illustrated in Figures 6, 7, 8, 9, 10 and 11. After this stage, the trained NER model is used to extract Named Entities (NEs). Basic dependency visualisation is done. These steps are shown in Figures 17 and 18. The next step involves importing a vectorizer to convert text to numerical vectors for further processing, and a model is built for further exploratory analysis, including TFIDF analysis to determine the most important words, visualising which words are the most similar, and Agglomerative and K-means clustering to identify clusters of terms. This stage is depicted in Figures 12, 13, 14, 15, 16, 17, 18 and 19.


Figure 1. Overview of the Heliophysics .txt file
![alt text](https://github.com/meganpowers1/SemanticNaturalLanguageProcessingforKnowledgeGraphsCreation/blob/main/AutOntFigures/Step0.png?raw=true)


Figure 2. Opening and Saving the Dataset
![alt text](https://github.com/meganpowers1/SemanticNaturalLanguageProcessingforKnowledgeGraphsCreation/blob/main/AutOntFigures/Step1.png?raw=true)


Figure 3. Removing Stopwords from the Corpus
![alt text](https://github.com/meganpowers1/SemanticNaturalLanguageProcessingforKnowledgeGraphsCreation/blob/main/AutOntFigures/Step2.png?raw=true)


Figure 4. Tokenizing the Corpus
![alt text](https://github.com/meganpowers1/SemanticNaturalLanguageProcessingforKnowledgeGraphsCreation/blob/main/AutOntFigures/Step3.png?raw=true)


Figure 5. Stemming and Lemmatizing the Corpus
![alt text](https://github.com/meganpowers1/SemanticNaturalLanguageProcessingforKnowledgeGraphsCreation/blob/main/AutOntFigures/Step4.png?raw=true)


Figure 6. Importing spaCy
![alt text](https://github.com/meganpowers1/SemanticNaturalLanguageProcessingforKnowledgeGraphsCreation/blob/main/AutOntFigures/Step5.png?raw=true)


Figure 7. Training Data and Custom Labelling
![alt text](https://github.com/meganpowers1/SemanticNaturalLanguageProcessingforKnowledgeGraphsCreation/blob/main/AutOntFigures/Step6.png?raw=true)


Figure 8. Training the en core web sm Model
![alt text](https://github.com/meganpowers1/SemanticNaturalLanguageProcessingforKnowledgeGraphsCreation/blob/main/AutOntFigures/Step7.png?raw=true)


Figure 9. Testing Data for the Trained en core web sm Model
![alt text](https://github.com/meganpowers1/SemanticNaturalLanguageProcessingforKnowledgeGraphsCreation/blob/main/AutOntFigures/Step8.png?raw=true)


Figure 10. Evaluation Metrics for Trained en core web sm Model
![alt text](https://github.com/meganpowers1/SemanticNaturalLanguageProcessingforKnowledgeGraphsCreation/blob/main/AutOntFigures/Step9.png?raw=true)


Figure 11. Evaluation Metrics for Untrained en core web sm Model
![alt text](https://github.com/meganpowers1/SemanticNaturalLanguageProcessingforKnowledgeGraphsCreation/blob/main/AutOntFigures/Step10.png?raw=true)


Figure 12. Employing Trained en core web sm Model to Identify Named Entities (NEs)
![alt text](https://github.com/meganpowers1/SemanticNaturalLanguageProcessingforKnowledgeGraphsCreation/blob/main/AutOntFigures/Step11.png?raw=true)


Figure 13. Employing Trained en core web sm Model to Identify Named Entities (NEs)
![alt text](https://github.com/meganpowers1/SemanticNaturalLanguageProcessingforKnowledgeGraphsCreation/blob/main/AutOntFigures/Step12.png?raw=true)


Figure 14. TFIDF Vectorizing the Heliophysics Text Corpus
![alt text](https://github.com/meganpowers1/SemanticNaturalLanguageProcessingforKnowledgeGraphsCreation/blob/main/AutOntFigures/Step13.png?raw=true)


Figure 15. Building Word2Vec Model for Text Analysis
![alt text](https://github.com/meganpowers1/SemanticNaturalLanguageProcessingforKnowledgeGraphsCreation/blob/main/AutOntFigures/Step14.png?raw=true)


Figure 16. Display Words by Highest TFIDF Value
![alt text](https://github.com/meganpowers1/SemanticNaturalLanguageProcessingforKnowledgeGraphsCreation/blob/main/AutOntFigures/Step15.png?raw=true)


Figure 17. Creating Word2Vec Model
![alt text](https://github.com/meganpowers1/SemanticNaturalLanguageProcessingforKnowledgeGraphsCreation/blob/main/AutOntFigures/Step16.png?raw=true)


Figure 18. Using Model to Plot Most Similar Words to ’Solar’
![alt text](https://github.com/meganpowers1/SemanticNaturalLanguageProcessingforKnowledgeGraphsCreation/blob/main/AutOntFigures/Step17.png?raw=true)


Figure 19. Import and Use Agglomerative Clustering to Visualise Clusters of Similar Terms
![alt text](https://github.com/meganpowers1/SemanticNaturalLanguageProcessingforKnowledgeGraphsCreation/blob/main/AutOntFigures/Step18.png?raw=true)


Figure 20. Result of Using Agglomerative Clustering to Display Total Number of Clusters
![alt text](https://github.com/meganpowers1/SemanticNaturalLanguageProcessingforKnowledgeGraphsCreation/blob/main/AutOntFigures/Step19.png?raw=true)


Figure 21. Importing K-Means Clustering to Visualise Clusters of Similar Terms
![alt text](https://github.com/meganpowers1/SemanticNaturalLanguageProcessingforKnowledgeGraphsCreation/blob/main/AutOntFigures/Step20.png?raw=true)


Figure 22. Define Function get entities to Construct Entities


Figure 23. Use get entities to Extract Entity Pairs


The next stage is to extract the Named Entities (NEs) and relationships by using Part of Speech (PoS) tagging in order to convert them into RDF triples. Then, extraneous characters and words are removed from the list. This process is displayed in Figures 20, 21, 22 and 23. The next stage is to prepare the entities and relations for the Knowledge Graph and for processing into RDF triples. The sources (subjects), targets (objects), and edges (relationships) are stored in a dataframe, and a directed graph is created. As this graph is large, a sample displaying nodes matching a specific relation is displayed. This process is visualised in Figures 31, 32, 33 and 34. The last stage is to convert the sources, targets, and edges into RDF triples and insert them into an ontology. Class and namespace properties were defined, and triples were added to the graph. Additionally, labels corresponding to the triples were added to the graph. Afterwards, the graph was serialised to an OWL file. This is visualised in Figures 35, 36, 37 and 38. Then, the resulting ontology can be visualised in Protege. This includes Axioms, Classes, Instances, and Object Properties. An overview of the automatically generated ontology is displayed in Figures 39, 40, 41 and 42.

The last stage is to convert the sources, targets, and edges into RDF triples and insert them into an ontology. Class and namespace properties were defined, and triples were added to the graph. Additionally, labels corresponding to the triples were added to the graph. Afterwards, the graph was serialised to an OWL file. This is visualised in Figures 6.30 to 6.33.

Then, the resulting ontology can be visualised in Prot´eg´e. This includes Axioms, Classes, Instances, and Object Properties. An overview of the automatically generated ontology is displayed in Figures 6.34 to 6.37.

