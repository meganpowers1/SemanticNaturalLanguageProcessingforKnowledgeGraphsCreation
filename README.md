# SemanticNaturalLanguageProcessingforKnowledgeGraphsCreation

This ReadMe contains an overview of the model prototype, starting from importing text and finishing with the final ontology. This overview is presented in steps. 

To begin with, the sample ontology corpus is saved into a .txt file and imported into a Jupyter Notebook. Stopwords are removed from the dataset, and it is tokenized, stemmed, and lemmatized. This process is shown in Figures 1 to 5. 


<i> Figure 1. Overview of the Heliophysics .txt file </i>

![alt text](https://github.com/meganpowers1/SemanticNaturalLanguageProcessingforKnowledgeGraphsCreation/blob/main/AutOntFigures/Step0.png?raw=true)


<i>Figure 2. Opening and Saving the Dataset </i>

![alt text](https://github.com/meganpowers1/SemanticNaturalLanguageProcessingforKnowledgeGraphsCreation/blob/main/AutOntFigures/Step1.png?raw=true)


<i>Figure 3. Removing Stopwords from the Corpus</i>
  
![alt text](https://github.com/meganpowers1/SemanticNaturalLanguageProcessingforKnowledgeGraphsCreation/blob/main/AutOntFigures/Step2.png?raw=true)


<i>Figure 4. Tokenizing the Corpus</i>
  
![alt text](https://github.com/meganpowers1/SemanticNaturalLanguageProcessingforKnowledgeGraphsCreation/blob/main/AutOntFigures/Step3.png?raw=true)


<i>Figure 5. Stemming and Lemmatizing the Corpus</i>

![alt text](https://github.com/meganpowers1/SemanticNaturalLanguageProcessingforKnowledgeGraphsCreation/blob/main/AutOntFigures/Step4.png?raw=true)


The next stage in the process is the Named Entity Recognition stage. First, the spaCy en core web sm model is imported, trained, and tested. The test results of the trained en core web sm model are compared to a default en core web sm model. This process is illustrated in Figures 6 to 11. 


<i>Figure 6. Importing spaCy</i>

![alt text](https://github.com/meganpowers1/SemanticNaturalLanguageProcessingforKnowledgeGraphsCreation/blob/main/AutOntFigures/Step5.png?raw=true)


Figure 7. Training Data and Custom Labelling
![alt text](https://github.com/meganpowers1/SemanticNaturalLanguageProcessingforKnowledgeGraphsCreation/blob/main/AutOntFigures/Step6.png?raw=true)


<i>Figure 8. Training the en core web sm Model</i>

![alt text](https://github.com/meganpowers1/SemanticNaturalLanguageProcessingforKnowledgeGraphsCreation/blob/main/AutOntFigures/Step7.png?raw=true)


<i>Figure 9. Testing Data for the Trained en core web sm Model</i>
![alt text](https://github.com/meganpowers1/SemanticNaturalLanguageProcessingforKnowledgeGraphsCreation/blob/main/AutOntFigures/Step8.png?raw=true)


<i>Figure 10. Evaluation Metrics for Trained en core web sm Model</i>

![alt text](https://github.com/meganpowers1/SemanticNaturalLanguageProcessingforKnowledgeGraphsCreation/blob/main/AutOntFigures/Step9.png?raw=true)


<i>Figure 11. Evaluation Metrics for Untrained en core web sm Model</i>

![alt text](https://github.com/meganpowers1/SemanticNaturalLanguageProcessingforKnowledgeGraphsCreation/blob/main/AutOntFigures/Step10.png?raw=true)


After this stage, the trained NER model is used to extract Named Entities (NEs). Basic dependency visualisation is done. These steps are shown in Figures 12 and 13. 


<i> Figure 12. Employing Trained en core web sm Model to Identify Named Entities (NEs) </i>
![alt text](https://github.com/meganpowers1/SemanticNaturalLanguageProcessingforKnowledgeGraphsCreation/blob/main/AutOntFigures/Step11.png?raw=true)


<i>Figure 13. Employing Trained en core web sm Model to Identify Named Entities (NEs)</i>

![alt text](https://github.com/meganpowers1/SemanticNaturalLanguageProcessingforKnowledgeGraphsCreation/blob/main/AutOntFigures/Step12.png?raw=true)


The next step involves importing a vectorizer to convert text to numerical vectors for further processing, and a model is built for further exploratory analysis, including TFIDF analysis to determine the most important words, visualising which words are the most similar, and Agglomerative and K-means clustering to identify clusters of terms. This stage is depicted in Figures 14 to 21.

<i>Figure 14. TFIDF Vectorizing the Heliophysics Text Corpus</i>
  
![alt text](https://github.com/meganpowers1/SemanticNaturalLanguageProcessingforKnowledgeGraphsCreation/blob/main/AutOntFigures/Step13.png?raw=true)


<i>Figure 15. Building Word2Vec Model for Text Analysis</i>

![alt text](https://github.com/meganpowers1/SemanticNaturalLanguageProcessingforKnowledgeGraphsCreation/blob/main/AutOntFigures/Step14.png?raw=true)


<i>Figure 16. Display Words by Highest TFIDF Value</i>
  
![alt text](https://github.com/meganpowers1/SemanticNaturalLanguageProcessingforKnowledgeGraphsCreation/blob/main/AutOntFigures/Step15.png?raw=true)


<i>Figure 17. Creating Word2Vec Model</i>
![alt text](https://github.com/meganpowers1/SemanticNaturalLanguageProcessingforKnowledgeGraphsCreation/blob/main/AutOntFigures/Step16.png?raw=true)


<i>Figure 18. Using Model to Plot Most Similar Words to ’Solar’</i>
![alt text](https://github.com/meganpowers1/SemanticNaturalLanguageProcessingforKnowledgeGraphsCreation/blob/main/AutOntFigures/Step17.png?raw=true)


<i>Figure 19. Import and Use Agglomerative Clustering to Visualise Clusters of Similar Terms</i>

![alt text](https://github.com/meganpowers1/SemanticNaturalLanguageProcessingforKnowledgeGraphsCreation/blob/main/AutOntFigures/Step18.png?raw=true)


<i>Figure 20. Result of Using Agglomerative Clustering to Display Total Number of Clusters</i>

![alt text](https://github.com/meganpowers1/SemanticNaturalLanguageProcessingforKnowledgeGraphsCreation/blob/main/AutOntFigures/Step19.png?raw=true)


<i>Figure 21. Importing K-Means Clustering to Visualise Clusters of Similar Terms</i>
  
![alt text](https://github.com/meganpowers1/SemanticNaturalLanguageProcessingforKnowledgeGraphsCreation/blob/main/AutOntFigures/Step20.png?raw=true)


The next stage is to extract the Named Entities (NEs) and relationships by using Part of Speech (PoS) tagging in order to convert them into RDF triples. Then, extraneous characters and words are removed from the list. This process is displayed in Figures 22 to 25. 


<i>Figure 22. Define Function get entities to Construct Entities</i>
![alt text](https://github.com/meganpowers1/SemanticNaturalLanguageProcessingforKnowledgeGraphsCreation/blob/main/AutOntFigures/Step21.png?raw=true)


<i>Figure 23. Use get entities to Extract Entity Pairs</i>

![alt text](https://github.com/meganpowers1/SemanticNaturalLanguageProcessingforKnowledgeGraphsCreation/blob/main/AutOntFigures/Step22.png?raw=true)


<i>Figure 24. Define and Use Function get relation to Extract Relations Between Entities</i>
  
![alt text](https://github.com/meganpowers1/SemanticNaturalLanguageProcessingforKnowledgeGraphsCreation/blob/main/AutOntFigures/Step23.png?raw=true)


<i>Figure 25. Remove Extraneous Words and Characters from Entity Pairs and Relations</i>
  
![alt text](https://github.com/meganpowers1/SemanticNaturalLanguageProcessingforKnowledgeGraphsCreation/blob/main/AutOntFigures/Step24.png?raw=true)


The next stage is to prepare the entities and relations for the Knowledge Graph and for processing into RDF triples. The sources (subjects), targets (objects), and edges (relationships) are stored in a dataframe, and a directed graph is created. As this graph is large, a sample displaying nodes matching a specific relation is displayed. This process is visualised in Figures 26 to 29. 


<i>Figure 26. Extract Corresponding Labels</i>
![alt text](https://github.com/meganpowers1/SemanticNaturalLanguageProcessingforKnowledgeGraphsCreation/blob/main/AutOntFigures/Step25.png?raw=true)


<i>Figure 27. Create Pandas Dataframe</i>
![alt text](https://github.com/meganpowers1/SemanticNaturalLanguageProcessingforKnowledgeGraphsCreation/blob/main/AutOntFigures/Step26.png?raw=true)


<i>Figure 28. Define and Display Graph</i>
![alt text](https://github.com/meganpowers1/SemanticNaturalLanguageProcessingforKnowledgeGraphsCreation/blob/main/AutOntFigures/Step27.png?raw=true)


<i>Figure 29. Display Graph for Edge ”by”</i>
![alt text](https://github.com/meganpowers1/SemanticNaturalLanguageProcessingforKnowledgeGraphsCreation/blob/main/AutOntFigures/Step28.png?raw=true)


The last stage is to convert the sources, targets, and edges into RDF triples and insert them into an ontology. Class and namespace properties were defined, and triples were added to the graph. Additionally, labels corresponding to the triples were added to the graph. Afterwards, the graph was serialised to an OWL file. This is visualised in Figures 30 to 33. 


<i>Figure 30. Create Graph and Namespace</i>

![alt text](https://github.com/meganpowers1/SemanticNaturalLanguageProcessingforKnowledgeGraphsCreation/blob/main/AutOntFigures/Step29.png?raw=true)


<i>Figure 31. Define Domain Spaces and Add RDF Triples to Graph</i>

![alt text](https://github.com/meganpowers1/SemanticNaturalLanguageProcessingforKnowledgeGraphsCreation/blob/main/AutOntFigures/Step30.png?raw=true)


<i>Figure 32. Labels are Generated, Matched to Instances, and Added to Graph</i>

![alt text](https://github.com/meganpowers1/SemanticNaturalLanguageProcessingforKnowledgeGraphsCreation/blob/main/AutOntFigures/Step31.png?raw=true)


<i>Figure 33. The Ontology is Populated</i>

![alt text](https://github.com/meganpowers1/SemanticNaturalLanguageProcessingforKnowledgeGraphsCreation/blob/main/AutOntFigures/Step32.png?raw=true)


Then, the resulting ontology can be visualised in Protege. This includes Axioms, Classes, Instances, and Object Properties. An overview of the automatically generated ontology is displayed in Figures 34 to 37.


<i>Figure 34. Overview of Protege Ontology</i>

![alt text](https://github.com/meganpowers1/SemanticNaturalLanguageProcessingforKnowledgeGraphsCreation/blob/main/AutOntFigures/Step28.png?raw=true)


<i>Figure 35. Example class in Protege</i>

![alt text](https://github.com/meganpowers1/SemanticNaturalLanguageProcessingforKnowledgeGraphsCreation/blob/main/AutOntFigures/Step29.png?raw=true)


<i>Figure 36. Example object property in Protege</i>

![alt text](https://github.com/meganpowers1/SemanticNaturalLanguageProcessingforKnowledgeGraphsCreation/blob/main/AutOntFigures/Step30.png?raw=true)


<i>Figure 37. Example instance in Protege</i>

![alt text](https://github.com/meganpowers1/SemanticNaturalLanguageProcessingforKnowledgeGraphsCreation/blob/main/AutOntFigures/Step31.png?raw=true)



