#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""

    CfHA Parsing-Heliophysics-data
    Written by: Megan T. Powers 
    For Ms.C. Advanced Computer Science
    12/09/2022

"""
"""
    The first stage in construcing the model is in pre-processing the text corpus. 
    As a result, Beautiful Soup is imported for pre-processing text.
    
"""
import bs4 as bs
import urllib.request 

# Create a default blank string to hold the text corpus
article_text = ''


# Open the source file in order to read from it, ensure that it is properly shortened
# using truncation so that unnecessary extra words are not appended to the storage files.

with open('/Users/meganpowers/Desktop/HeliophysicsDataTrain.txt') as f:
    article_text = f.read()

file = open('/Users/meganpowers/Desktop/HeliophysicsDataset.txt', 'w')
file.truncate(0)
file.write(article_text)
file.close()

# Let the user know that the text has been parsed. 

print('Text has been successufully parsed!')


# In[2]:


# NLTK, the Natural Language ToolKit, is imported for further text pre-processing.

import nltk

# The first stage in pre-processing is tokenization - splitting the text corpus into
# unique words, or 'tokens'

with open('/Users/meganpowers/Desktop/HeliophysicsDataset.txt') as f:
  contents = f.read()
corpus = nltk.sent_tokenize(contents)

f.close()


# In[3]:


# re is a Regular Expression (RegEx) library, used for pattern matching.
# This serves to eliminate unnecessary punctuation, short words, and other
# noise from the text corpus.

import re

for i in range(len(corpus )):
  corpus[i] = corpus [i].lower()
  corpus[i] = re.sub(r'\W',' ',corpus[i])
  corpus[i] = re.sub(r'\s+',' ',corpus[i])
  shortword = re.compile(r'\W*\b\w{1}\b') # remove short words
  corpus[i] = re.sub(shortword,' ',corpus[i])


# In[4]:


# The NLTK tokenizer is applied to the corpus to split it into distinct words.

corpus = nltk.word_tokenize(contents)


# In[5]:


# Stopwords are words that link between sentences - for example, pronouns,
# posessives, and comparatives. They are not needed for Named Entity Recognition,
# so the default stopwords from NLTK are imported so that they can be identified in
# the text corpus.

from nltk.corpus import stopwords
stopwords.words('english')


# In[6]:


# A list of stopwords is identified so that they can be cleaned from the dataset.

stop_words = set(stopwords.words('english')) 
fileClean = open('/Users/meganpowers/Desktop/HeliophysicsDatasetCleaned.txt', 'a') 
fileClean.truncate(0)

# Here, the tokens are checked to ensure that they are alphanumeric. 
# If they are, they are joined to the cleaned file. 

for sentence in corpus:
    tokens = nltk.word_tokenize(sentence)
    for token in tokens:
        token = ''.join([i for i in token if i.isalnum()]) 
        if not token in stop_words: #remove stopwords
            fileClean.write(" "+token) # save the cleaned text in a new file
fileClean.close()


# In[7]:


# Here, additional pre-processing modules are imported. 

# Stemming refers to the process of checking word stems, or 
# reducing words to their base components - for example,
# logical -> logic

from nltk.stem import PorterStemmer

# Lemmatizing groups together inflected forms of a word so that they
# can be analysed as a single entity.

from nltk.stem import WordNetLemmatizer

# Numpy has support for large matrices as well as mathematical functions

import numpy as np
import nltk


# In[8]:


with open('/Users/meganpowers/Desktop/HeliophysicsDatasetCleaned.txt') as f:
  contents = f.read()
tokens = nltk.word_tokenize(contents)


# Here, the stemming process is applied to the cleaned heliophysics text corpus.
# The text is facilitate to optimise the stemming process by turning the text
# to a numerical form.

porter=PorterStemmer()
stem_words = np.vectorize(porter.stem)
stemed_text = ' '.join(stem_words(tokens))
print(f"nltk stemed text: {stemed_text}")

# Here, the lemmatizing process is applied to the cleaned heliophysics text corpus.
# The text is vectorized to facilitate the lemmatizing process by turning the text
# to a numerical form.

wordnet_lemmatizer = WordNetLemmatizer()
lemmatize_words = np.vectorize(wordnet_lemmatizer.lemmatize)
lemmatized_text = ' '.join(lemmatize_words(tokens))
print(f"nltk lemmatized text: {lemmatized_text}")

f.close()


# Here, the text pre-processing concludes.


# In[9]:


"""
    The next leg of the model is the Named Entity Recognition (NER) stage. 
    Here, Spacy is used in order to facilitate the NER process. 

"""

# The Spacy module is imported alongside displacy, which is a dependency visualiser. 
import spacy
from spacy import displacy

"""
nlp=spacy.blank("en")
sp = spacy.load('en_core_web_sm')
"""


# Here, a NLP model needs to be imported. In this case, the model is built on top of the
# default pre-trained English model.

nlp = spacy.load("en_core_web_sm")

sp = spacy.load('en_core_web_sm')

# This code provides a demonstration of the Part of Speech (PoS) tagging feature
# of the default Spacy model. It is visualised using displacy.

sen = nlp(u"Action items: Ryan : create image of process of domain modeling from CfHA KT for the paper Ryan : resolve marginalia notes in overleaf All : propose the KG-related citations that have been most influential and insightful in your experience and reference them in the paper Ryan : lead a discussion on the paper with the CfHA KT and share to get their feedbac Edlira : put visuals into Overleaf doc Megan , Swapnali : Share draft proposed work for MS Find out due dates for NSF and EPA SBIRs All : Complete all existing actions for the paper by Friday June 24 Edlira : make sure introduction and methodology are consistent")
displacy.render(sen, style='dep', jupyter=True, options={'distance': 85})


# In[13]:


# In this stage, the default Spacy model needs to be trained on 
# data that is relevant to the CfHA/heliophysics domain.

# First, relevant libraries are imported for training the 
# default model. This is done by creating a pipeline 
# that will update the NER aspect of the model 
# by using batches of training data.

import random
from spacy.training import Example
from spacy.util import minibatch, compounding
from pathlib import Path
from spacy.pipeline import EntityRuler


"""
nlp = spacy.blank('en')
sp = spacy.blank('en')


"""


# Here, a pipe is created for NER specifically to add the training data to the model.
# The new NER labels are based off categories found in the CfHA ontology, and 
# are added to the ner model.

ner = nlp.create_pipe("ner")
ner.add_label("ASTROPHYSICS")
ner.add_label("HELIOPHYSICS")
ner.add_label("MISSION")
ner.add_label("PROJECT")
ner.add_label("PAPER")

"""
    Paper titles, data titles, software titles, project titles, etc
"""


# Here, we need to provide a training data corpus where entities are already tagged.
# This data will provide the NER model with example entities to base itself off of.

TRAIN_DATA = [
    ("Edlira: put visuals into Overleaf doc", {"entities": [(0, 6, "PERSON"), (25, 33, "ORG")]}),
    ("Ryan: review the abstract", {"entities": [(0, 4, "PERSON")]}),
    ("Ellie: improve figures she provided", {"entities": [(0, 5, "PERSON")]}),
    ("Edlira: provide descriptions of new figures you provided and make sure they are addressed in the text", {"entities": [(0, 6, "PERSON")]}),
    ("Ellie and Ryan: offer feedback to Swapnali MS project proposal", {"entities": [(0, 5, "PERSON"), (10, 14, "PERSON"), (34, 42, "PERSON")]}),
    ("Megan and Swapnali: set up weekly meetings for MS projects", {"entities": [(0, 5, "PERSON"), (10,18, "PERSON")]}),
    ("Swapnali and Megan in middle of MS work (until September)", {"entities": [(0, 8, "PERSON"), (13,18, "PERSON")]}),
    ("Edlira, Megan and Swapnali: update ontology and provide interesting axioms (novel)", {"entities": [(0, 6, "PERSON"), (8,13, "PERSON"), (18,26, "PERSON")]}),
    ("Megan and Swapnali: set up weekly meetings for MS projects", {"entities": [(0, 5, "PERSON"), (10,18, "PERSON")]}),
    ("Semantic Web Journal paper types: http://www.semantic-web-journal.net/reviewers", {"entities": [(0, 20, "ORG")]}),
    ("Megan and Swapnali to update methodology+figures based on our last discussion (does this also result in a change to the results section of the paper?)", {"entities": [(0, 5, "PERSON"), (10, 18, "PERSON")]}),
    ("Host in Github for our group (CfHA)", {"entities": [(8, 14, "ORG"), (30, 35, "ORG")]}),
    ("CfHA Knowledge Team Running Notes: https://docs.google.com/document/d/1R4DCc5bs9R-uczF9Jy-b7TSbGSfkMNkjMCL_xNCeTMs/edit?usp=sharing", {"entities": [(0, 4, "ORG")]}),
    ("CfHA Miro: https://miro.com/app/board/o9J_klKaKEg=/", {"entities": [(0, 4, "ORG")]}),
    ("I've rewritten the intro and kept the description of CfHA unchanged", {"entities": [(53, 57, "ORG")]}),
    ("Add the visuals from Megan and Swapnali ", {"entities": [(21, 26, "PERSON"), (31,19, "PERSON")]}),
    ("Megan and Swapnali [k]: provide v1.0 visuals for the paper ", {"entities": [(0, 5, "PERSON"), (10,18, "PERSON")]}),
    ("Giant impacts dominate the final stages of terrestrial planet formation and set the configuration and compositions of the final system of planets.", {"entities": [(55, 61, "ASTROPHYSICS")]}),
    ("The impact of gravity waves GW on diurnal tides and the global circulation in the middle/upper atmosphere of Mars is investigated using a general circulation model GCM.", {"entities": [(14, 28, "ASTROPHYSICS"), (34, 48, "ASTROPHYSICS"), (109, 114, "ASTROPHYSICS")]}),
    ("Separate populations were accelerated above the Io torus and at high latitudes near Jupiter", {"entities": [(48, 51, "ASTROPHYSICS"),(84, 92, "ASTROPHYSICS")]}),
    ("BepiColombo, a joint mission to Mercury by the European Space Agency and Japan Aerospace Exploration Agency, will address remaining open questions using two spacecraft, Mio and the Mercury Planetary Orbiter.", {"entities": [(0, 11, "MISSION"), (32, 39, "ASTROPHYSICS"), (47, 68, "ORG"), (74, 107, "ORG"), (169, 172, "MISSION"), (181, 206, "MISSION")]}),
    ("Mio First Comprehensive Exploration of Mercury's Space Environment: Mission Overview", {"entities": [(0, 85, "PAPER")]}),
    ("Martian Thermospheric Warming Associated With the Planet Encircling Dust Event of 2018", {"entities": [(0, 86, "PAPER")]}),
    ("Ray-and-power tracing provided wave amplitudes as well as trajectories and wave normal angles throughout the plasmasphere.", {"entities": [(31, 46, "ASTROPHYSICS"), (109, 121, "ASTROPHYSICS")]}),
    ("Dynamical Evolution of Simulated Particles Ejected From Asteroid Bennu", {"entities": [(0, 70, "PAPER")]}),
    ("We use global and local hybrid kinetic ions and fluid electrons simulations to investigate the conditions under which foreshock bubbles FBs form and how their topology changes with solar wind conditions.", {"entities": [(48, 58, "ASTROPHYSICS")]}),
    ("FBs form as a result of the interaction between solar wind discontinuities and backstreaming ion beams in the foreshock.", {"entities": [(48, 58, "ASTROPHYSICS"), (93, 97, "ASTROPHYSICS")]}),
    ("The visible and near-infrared imaging spectrometer on board the Yutu-2 rover of ChangE-4 mission has conducted 2 sets of spectrophotometric measurements at two sites on its 10th lunar day.", {"entities": [(38, 51, "ASTROPHYSICS"), (64, 71, "PROJECT"), (80, 89, "MISSION")]}),
    ("The Mars Science Laboratory mission investigated Vera Rubin ridge, which bears spectral indications of elevated amounts of hematite and has been hypothesized as having a complex diagenetic history.", {"entities": [(4, 36, "MISSION"), (49, 60, "ASTROPHYSICS")]}),
    ("The InSight mission to Mars landed within Homestead hollow on an Early Amazonian lava plain.", {"entities": [(4, 20, "MISSION"), (23, 28, "ASTROPHYSICS"), (42, 52, "ASTROPHYSICS"), (81, 91, "ASTROPHYSICS")]}),
    ("The many completed studies show an Ice Giant mission with an in situ probe is feasible and would be welcomed by the international science community.", {"entities": [(35, 53, "MISSION")]}),
    ("NASA Parker Solar Probe mission is currently investigating the local plasma environment of the inner heliosphere &lt;0.25 R<SUB>\u2609</SUB> using both in situ and remote sensing instrumentation.", {"entities": [(0, 5, "ORG"), (5, 32, "MISSION"), (69, 76, "HELIOPHYSICS"), (101, 113, "HELIOPHYSICS")]}),
    ("We will relate the results of the Rosetta mission to those of the flybys.", {"entities": [(34, 50, "MISSION")]}),
    ("Cometary Nuclei: From Giotto to Rosetta", {"entities": [(0, 40, "PAPER")]}),
    ("A Maximum Rupture Model for the Southern San Andreas and San Jacinto Faults, California, Derived From Paleoseismic Earthquake Ages: Observations and Limitations", {"entities": [(0, 161, "PAPER")]}),
    ("The CESM2 is the version of the CESM contributed to the sixth phase of the Coupled Model Intercomparison Project CMIP6.", {"entities": [(4, 10, "PROJECT"), (32, 37, "PROJECT"), (75, 113, "PROJECT")]}),
    ("The datasets of two Ocean Model Intercomparison Project simulation experiments from the Climate Ocean Model Project, forced by two different sets of atmospheric surface data, are described in this paper.", {"entities": [(20, 56, "PROJECT"), (89, 116, "PROJECT")]}),
    ("Model simulations in the Community Earth System Model Large Ensemble Project confirmed the physical connection between the warm CEP SST anomaly and the drought in EC.", {"entities": [(25, 77, "PROJECT")]}),
    ("The pickup process on the extended oxygen corona created by the strong EUV flux contributes to the total O+ loss.", {"entities": [(42, 49, "HELIOPHYSICS")]}),
    ("As systems become more complex over time, the impacts of space weather on space flights and humanity in general are likely to increase.", {"entities": [(57, 71, "ASTROPHYSICS"), (74, 88, "MISSION")]}), 
    ("Humans will encounter extremely serious problems of space flight safety at the beginning of new phase of the Moon exploration.", {"entities": [(52, 65, "MISSION"), (109, 114, "ASTROPHYSICS")]}),
    ("Motivated by a successful prediction on the peak of solar cycle 24 81.7, comparable to the observed 81.9, Du in Astrophys.", {"entities": [(52, 64, "HELIOPHYSICS")]}),
    ("The pickup process on the extended oxygen corona created by the strong EUV flux contributes to the total O+ loss.", {"entities": [(42, 49, "HELIOPHYSICS")]}),
    ("Such plasma is composed by the generalized distributed electrons, Boltzmann distributed positrons and relativistic warm ions.", {"entities": [(5, 12, "ASTROPHYSICS"), (31, 65, "ASTROPHYSICS"), (66, 98, "ASTROPHYSICS"), (102, 125, "ASTROPHYSICS")]}),
    ("A new 1.5 m diameter impact crater was discovered on Mars only ~40 km from the InSight lander.", {"entities": [(21, 35, "ASTROPHYSICS"), (53, 58, "ASTROPHYSICS"), (79, 94, "MISSION")]}),
    ("In this study, we investigate systematically the variations of the occurrence of depletions with both internal and external conditions, using the extensive Solar Wind Electron Analyzer measurements made on board the Mars Atmosphere and Volatile Evolution.", {"entities": [(156, 185, "MISSION"), (216, 255, "MISSION")]}),
    ("Comparisons Between Jupiter's X-ray, UV and Radio Emissions and In-Situ Solar Wind Measurements During 2007.", {"entities": [(0, 108, "PAPER")]}),
    ("The final angular momentum, set by the timing of quasi-resonance escape, is a function of the ratio of tidal strength in the Moon and Earth and the absolute rate of tidal dissipation in the Earth.", {"entities": [(125, 130, "ASTROPHYSICS"), (134, 140, "ASTROPHYSICS"), (165, 183, "ASTROPHYSICS"), (190, 196, "ASTROPHYSICS")]}),
    ("We use an ensemble of simulations of the Goddard Institute for Space Studies Earth system model to compute climate response functions CRFs for the addition of meltwater.", {"entities": [(41, 77, "ORG"), (77, 96, "PROJECT"), (107, 134, "ASTROPHYSICS")]}),
    ("We infer that these clouds formed as a result of semidiurnal thermal tides.", {"entities": [(49, 75, "ASTROPHYSICS")]}),
    ("Magma Oscillations in a Conduit-Reservoir System, Application to Very Long Period (VLP) Seismicity at Basaltic Volcanoes: 2. Data Inversion and Interpretation at Kilauea Volcano", {"entities": [(0, 178, "PAPER")]}),
    ("The German Aerospace Center is currently developing the Reusability Flight Experiment.", {"entities": [(4, 28, "ORG"),(56, 86, "PROJECT")]}),
    ("Results demonstrate that the Iexp mainly improved the model outputs with respect to assimilation-free Massachusetts Institute of Technology General Circulation Model run in the first few months,", {"entities": [(29, 34, "PROJECT"),(102, 140, "ORG"),(140, 166, "PROJECT")]}),
    ("Characteristics of Minor Ions and Electrons in Flux Transfer Events Observed by the Magnetospheric Multiscale Mission", {"entities": [(0, 118, "PAPER")]}),
    ("Eddies were identified and tracked within a numerical simulation that used the Massachusetts Institute of Technology general circulation model and an eddy characterization algorithm.", {"entities": [(79, 117, "ORG")]}),
    ("Our results suggest that the electron precipitation through the polar rain can be a main energy source of the polar wind during periods of high levels of solar activity.", {"entities": [(29, 52, "HELIOPHYSICS"),(64, 75, "HELIOPHYSICS"), (110, 121, "HELIOPHYSICS"), (154, 169, "HELIOPHYSICS")]}),
    ("Causes of Higher Climate Sensitivity in CMIP6 Models", {"entities": [(0, 53, "PAPER")]}),
    ("The Flux-Anomaly-Forced Model Intercomparison Project is an endorsed Model Intercomparison Project in phase 6 of the Coupled Model Intercomparison Project.", {"entities": [(4, 54, "PROJECT"),(69, 99, "PROJECT"),(117, 155, "PROJECT")]}),
    ("The National Aeronautics and Space Administration is currently developing the next generation of spacesuits for use in future exploration missions.", {"entities": [(4, 50, "ORG")]}),
    ("The National Aeronautics and Space Administration Earth Observing System Global Circulation Project is evaluated through a cascade of simulations.", {"entities": [(4, 50, "ORG"),(50, 100, "PROJECT")]}),
    ("Testing of the method on observations under various conditions in the solar wind confirms the reliability and accuracy of the method.", {"entities": [(70, 81, "HELIOPHYSICS")]}),
    ("These predictions may serve as a reference for eventual ionospheric measurements of multiple instruments and are leading to a better understanding of the ionospheric response to solar eclipses.", {"entities": [(56, 81, "HELIOPHYSICS"),(154, 175, "HELIOPHYSICS"), (178, 193, "HELIOPHYSICS")]}),
]


# Here is the actual training portion. For every training sentence and annotated entity in
# TRAIN_DATA, these examples will be appended to the NER model. Randomisation is employed
# to ensure that the model does not overfit, and to avoid the Catastrophic Forgetting 
# Problem where already-learned labels are forgotten.

examples = []
for text, annots in TRAIN_DATA:
    examples.append(Example.from_dict(nlp.make_doc(text), annots))
"""nlp.initialize(lambda: examples)"""
for i in range(40):
    random.shuffle(examples)
    for batch in minibatch(examples, size=8):
        nlp.update(batch)
        sp.update(batch)


        


# In[14]:


from spacy.training import Example


# Now, some testing validation data needs to be created. This test data is
# annotated like the training data, and evaluation metrics are employed
# in order to determine the success of the NER model for identifying entities.

new_test_data = []

test_data = [
    ("Mio First Comprehensive Exploration of Mercury's Space Environment: Mission Overview", {"entities": [(0, 85, "PAPER")]}),
    ("Martian Thermospheric Warming Associated With the Planet Encircling Dust Event of 2018", {"entities": [(0, 86, "PAPER")]}),
    ("To do this, three grids were generated that span the entire, two thirds, and one third of the North Pacific and evaluated for a 30-year climatology using Atmospheric Model Intercomparison Project protocols.",{"entities":[(154,196,"PROJECT")]}),
    ("Although the existence of an intrinsic magnetic field on ancient Mars is also a key factor in ion loss, its effect remains unclear.", {"entities": [(39, 54, "ASTROPHYSICS"), (65, 70, "ASTROPHYSICS"), (94, 98, "ASTROPHYSICS")]}), 
    ("These indices are a well proxy for the ionospheric fluctuations and can be used to describe features of plasma bubble irregularities.", {"entities": [(39, 64, "ASTROPHYSICS"), (104, 111, "ASTROPHYSICS")]}), 
    ("Cosmic radiation hazard is cornerstone of space flights safety.", {"entities": [(0, 17, "ASTROPHYSICS")]}),
    ("Solar flares and GCR are of special concern.", {"entities": [(0, 6, "HELIOPHYSICS"), (17, 21, "HELIOPHYSICS")]}),
    ("Altogether, it is referred to as space weather.", {"entities": [(52, 65, "MISSION"), (34, 47, "ASTROPHYSICS")]}),
    ("In this study, we investigate systematically the variations of the occurrence of depletions with both internal and external conditions, using the extensive Solar Wind Electron Analyzer measurements made on board the Mars Atmosphere and Volatile Evolution.", {"entities": [(156, 185, "MISSION"), (216, 255, "MISSION")]}),
    ("However, the brightest X-ray aurora was coincident with a magnetosphere expansion.", {"entities": [(23, 36, "ASTROPHYSICS"), (58, 72, "ASTROPHYSICS")]}),
    ("The diurnal cycle and orographic forcing, however, substantially enhance rainfall in the seas surrounding the islands.", {"entities": [(4, 18, "ASTROPHYSICS"), (22, 41, "ASTROPHYSICS")]}),
    ("Coobservations of four meteor trails trains from the Long Wavelength Array telescope in New Mexico and the Widefield Persistent Train WiPT camera associate the long-lasting tens of seconds, self-generated radio emission known as MRAs with the long-lasting tens of minutes optical emissions known as PTs.", {"entities": [(23, 30, "ASTROPHYSICS"), (53, 85, "PROJECT"), (107, 141, "PROJECT"), (200, 215, "ASTROPHYSICS"), (224, 229, "ASTROPHYSICS"), (267, 285, "ASTROPHYSICS"), (294, 298, "ASTROPHYSICS") ]}),
    ("Altogether, it is referred to as space weather.", {"entities": [(52, 65, "MISSION"), (34, 47, "ASTROPHYSICS")]}),
    ("Moreover, this new methodology is validated by setting up an experimental campaign performed in the Von Karman Institute Longshot wind tunnel.", {"entities": [(100, 121, "ORG")]}),
    ("The electric solar wind sail is a propulsion system that extracts the solar wind momentum for the thrust force of a spacecraft by using an interaction between solar wind protons and the electric potential structure around charged conducting tethers.", {"entities": [(4, 29, "ASTROPHYSICS"), (70, 81, "ASTROPHYSICS"), (159, 178, "ASTROPHYSICS"), (186, 215, "ASTROPHYSICS"), (230, 259, "ASTROPHYSICS")]}),
    ("The predicted ionosphere vertical TEC and the critical frequency are validated by the Massachusetts Institute of Technology and Global Ionosondes Network, respectively.", {"entities": [(0, 17, "ASTROPHYSICS")]}),
    ("These FTEs were observed just upstream of the Earth postnoon magnetopause by the National Aeronautics and Space Administration Magnetospheric Multiscale spacecraft constellation.", {"entities": [(6, 11, "ASTROPHYSICS"), (46, 52, "ASTROPHYSICS"), (52, 74, "ASTROPHYSICS"), (81, 127, "ORG"), (127, 153, "PROJECT")]}),
    ("The predicted ionosphere vertical TEC and the critical frequency are validated by the Massachusetts Institute of Technology and Global Ionosondes Network, respectively.", {"entities": [(0, 17, "ASTROPHYSICS")]}),
    ("Solar radiation showed an observable influence on surface O3 concentrations in the lower troposphere.", {"entities": [(0, 16, "HELIOPHYSICS")]}),
    ("We perform a statistical study with thousands of hours of magnetospheric multiscale observations in the solar wind, comparing the prediction accuracy of the multispacecraft monitor to all of the OMNIWeb single-spacecraft monitors.", {"entities": [(58, 97, "HELIOPHYSICS"),(106, 115, "HELIOPHYSICS"),(157, 181, "PROJECT"),(195, 203, "PROJECT")]}),
    ("We use an ensemble of simulations of the Goddard Institute for Space Studies Earth System Model Project to compute climate response functions for the addition of meltwater.", {"entities": [(41, 77, "ORG"), (77, 104, "PROJECT")]}),
]
     
# Here, new test data is being extracted from the test data in order to perform
# an evaluation on the new test data. This is done by using an Example,
# which is a type of class in Spacy that holds information for a training instance.

for text, annots in test_data:
    new_test_data.append(Example.from_dict(nlp.make_doc(text), annots))

#end formatted test data

# Using the evaluate function, the evaluation is performed on the new test data.
scores_model = nlp.evaluate(new_test_data)

# Here, relevant evaluation metrics are used. In this case, Precision, Recall, the F-Score,
# and a breakdown of scores by entities are formulated and displayed.
precision_model1 = scores_model["token_acc"]
precision_model = scores_model["ents_p"]
recall_model = scores_model["ents_r"]
f_score_model = scores_model["ents_f"]
scores_entities = scores_model["ents_per_type"]


print(precision_model1)
print(precision_model)
print(recall_model)
print(f_score_model)
print(scores_entities)


# In[15]:


"""
    Testing accuracy metrics on an untrained en_core_web_sm model

"""

nlp_untrained = spacy.load("en_core_web_sm")
ner_untrained = nlp_untrained.create_pipe("ner")
ner_untrained.add_label("ASTROPHYSICS")
ner_untrained.add_label("HELIOPHYSICS")
ner_untrained.add_label("MISSION")
ner_untrained.add_label("PROJECT")
ner_untrained.add_label("PAPER")


# Using the evaluate function, the evaluation is performed on the new test data.
scores_model_untrained = nlp_untrained.evaluate(new_test_data)

precision_model1_untrained = scores_model_untrained["token_acc"]
precision_model_untrained = scores_model_untrained["ents_p"]
recall_model_untrained = scores_model_untrained["ents_r"]
f_score_model_untrained = scores_model_untrained["ents_f"]
scores_entities_untrained = scores_model_untrained["ents_per_type"]


print(precision_model1_untrained)
print(precision_model_untrained)
print(recall_model_untrained)
print(f_score_model_untrained)
print(scores_entities_untrained)


# In[16]:


"""

    The next stage of the process is using the trained and evaluated NER model to 
    extract Named Entities (NEs). 
"""

# Import relevant libraries to extract Named Entities (NEs)
import re
import pandas as pd
import bs4
import requests
import spacy
from spacy import displacy
"""nlp = spacy.load('en_core_web_sm')"""

# Of particular note is importing the matcher and span in order to
# pattern match using Part of Speech tokens, and Span to create
# Span objects for storing slices

from spacy.matcher import Matcher 
from spacy.tokens import Span 

# For creating the Knowledge Graph

import networkx as nx

# For plotting data
import matplotlib.pyplot as plt

# For displaying a loading bar
from tqdm import tqdm


#Open the cleaned text file
with open('/Users/meganpowers/Desktop/HeliophysicsDatasetCleaned.txt') as f:
  contents = f.read()
#file to store the NER result
fileNER = open('/Users/meganpowers/Desktop/HeliophysicsDataset_NER.txt', 'a')
fileNER.truncate(0)

# For every entity that the NER identified, display the entity itself, the assigned label,
# and the explanation associated with the label. Write the labels to the 
# HeliophysicsDataset_NER.txt file

for entity in sp(contents).ents:
  print(entity.text + ' - ' + entity.label_ + ' - ' + str(spacy.explain(entity.label_)))
  fileNER.write(entity.text +'\n')
f.close()
fileNER.close()


# In[17]:


import spacy

# Import a dependency visualiser 
from spacy import displacy

# Test the NER automatic labelling capability
sen = sp(u'Action items: Ryan : create image of process of domain modeling from CfHA KT for the paper Ryan : resolve marginalia notes in overleaf All : propose the KG-related citations that have been most influential and insightful in your experience and reference them in the paper Ryan : lead a discussion on the paper with the CfHA KT and share to get their feedbac Edlira : put visuals into Overleaf doc Megan , Swapnali : Share draft proposed work for MS Find out due dates for NSF and EPA SBIRs All : Complete all existing actions for the paper by Friday June 24 Edlira : make sure introduction and methodology are consistent')
displacy.render(sen, style='ent', jupyter=True)



# In[18]:


import spacy

# Import a dependency visualiser 
from spacy import displacy

# Test the NER automatic labelling capability
sen = sp(u'The First Institute of Oceanography Earth System Model FIO-ESM version 2.0 was developed and participated in the Climate Model Intercomparison Project phase 6 CMIP6.')
displacy.render(sen, style='ent', jupyter=True)


# In[19]:


# Import a vectorizer to convert text to numerical values for 
# further processing 

from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from operator import itemgetter

with open('/Users/meganpowers/Desktop/HeliophysicsDatasetCleaned.txt') as f:
    contents = f.read()

# Tokenize the corpus in order to further vectorize it.
# Then, vectorizing converts text into numerical values
# and this results in the model being able to process the words

corpus = nltk.sent_tokenize(contents)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names())
#print(X.shape)

# Apply TFIDF, which is a statistic that reflects the importance of a word.

tf_idf_model=dict(zip(vectorizer.get_feature_names(), X.toarray()[0]))


# Save word TFIDF values to another text file

fileTFIDF = open('/Users/meganpowers/Desktop/HeliophysicsDatasetTFIDF.txt', 'a')
for key,val in tf_idf_model.items():
  #print (str(key) + ':' + str(val))
  fileTFIDF.write(str(key) + ':' + str(val)+'\n')

fileTFIDF.close()

# This sorts the dictionary of words by tf-idf, or which words are deemed most important
# Import a dependency visualiser and save the results in a text file
fileTFIDF_Sorted = open('/Users/meganpowers/Desktop/WikiDatasetTFIDF_Sorted.txt', 'a')
listofwords = sorted(tf_idf_model.items() , reverse=True, key=lambda x: x[1])

for elem in listofwords :
  print(elem[0] , " ::" , elem[1] )
  fileTFIDF_Sorted.write(str(elem[0]) + ':' +str( elem[1])+'\n')
    
fileTFIDF_Sorted.close()


# In[20]:



# Import gensim, which allows for topic modelling
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim import models, matutils
from gensim.models import Word2Vec

# Training and trained model files
input_filename = '/Users/meganpowers/Desktop/HeliophysicsDatasetCleaned.txt'
model_filename = '/Users/meganpowers/Desktop/HeliophysicsDatasetCleaned.model' 

# skip-gram allows for NLP context word predictions
skip_gram = True

# This step constructs a vocabulary from the words
print ('building vocabulary...')
model = models.Word2Vec()
sentences = models.word2vec.LineSentence(input_filename)
model.build_vocab(sentences)


# The model must be trained to find related words for a given word
print ('training model...')

if skip_gram:
    model.train(sentences, total_examples= model.corpus_count, epochs= 3)
else:
    model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)

# Save the trained model and tell the user that the training has finished
print ('- saving model...')
model.save(model_filename)
print ('all done, whew!')


# In[21]:


# Test the most similar words
print(model.wv.most_similar( 'solar' , topn=10))


# In[22]:


# Sorts corpus words by TFIDF value to display the most important terms

with open('/Users/meganpowers/Desktop/HeliophysicsDatasetCleaned.txt') as f:
    contents_tfidf = f.read()
feature_names = vectorizer.get_feature_names()

"""
    get_tfidf_for_words
    param: text
    
    Takes in text corpus and returns a dictionary of words and their TFIDF scores.

"""
def get_tfidf_for_words(text):
    tfidf_matrix= vectorizer.transform([text]).todense()
    feature_index = tfidf_matrix[0,:].nonzero()[1]
    tfidf_scores = zip([feature_names[i] for i in feature_index], [tfidf_matrix[0, x] for x in feature_index])
    return dict(tfidf_scores)

dic_1 = get_tfidf_for_words(contents_tfidf)

# Sort dictionary by TFIDF score and print out

for k in sorted(dic_1, key=dic_1.get, reverse=True):
    print(k, dic_1[k])
    
    
    


# In[23]:


# Perform model visualisation
# numpy and pyplot allow for graphs
# TSNE visualises high-dimension data and finds probability distribution 

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from gensim.models import Word2Vec

model_filename = '/Users/meganpowers/Desktop/HeliophysicsDatasetCleaned.model' # name for the saved trained model
print('loading the model ...')

# Open the model
model = Word2Vec.load(model_filename)

"""
    display_closestwords_tsnescatterplot
    params: model, word
    
    This function finds and plots the most closely
    related words based on TFIDF information
"""

def display_closestwords_tsnescatterplot(model, word):
    arr = np.empty((0,100), dtype='f')
    word_labels = [word]
    
    # Use word vector to find the most similar words to the param word
    close_words = model.wv.most_similar(word)
    
    # Sum the vector for each of the closest words in the array
    arr = np.append(arr, np.array(model.wv.__getitem__([word])), axis=0)
    
    # For each TFIDF score, get the corresponding item
    # and append the label
    for wrd_score in close_words:
        wrd_vector = model.wv.__getitem__([wrd_score[0]])
        word_labels.append(wrd_score[0])
        arr = np.append(arr, np.array(model.wv.__getitem__([word])), axis=0)
        
    # Create TSNE for high-dimension data to create the plot
    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(arr)
    x_coords = Y[:, 0]
    y_coords = Y[:, 1]
    
    # Display the corresponding scatterplot
    plt.scatter(x_coords, y_coords)
    
    # Create annotations for the different labels
    for label, x, y in zip(word_labels, x_coords, y_coords):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.xlim(x_coords.min()+0.00005, x_coords.max()+0.00005)
    plt.ylim(y_coords.min()+0.00005, y_coords.max()+0.00005)
    plt.show()

# Test model using 'solar'
display_closestwords_tsnescatterplot(model, 'solar')


# In[24]:


# Import clustering libraries to visualise closely
# related clusters of terms

import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster import hierarchy
from sklearn.cluster import AgglomerativeClustering
from matplotlib import pyplot as plt
import pandas as pd

with open('/Users/meganpowers/Desktop/HeliophysicsDataset.txt') as f:
  content = f.read()

# Tokenize and vectorize the corpus to get closely related words
corpus = nltk.sent_tokenize(content)
vectorizer = TfidfVectorizer(stop_words='english')

# Turn the corpus into a data frame in order to allow for clustering
X = vectorizer.fit_transform(corpus)
df_text = pd.DataFrame(X.toarray())
print(df_text)

#Create the Agglomerative Clustering model

agg_clustering = AgglomerativeClustering(n_clusters = 3, affinity = 'euclidean', linkage = 'ward')

# Predict labels for the clusters
labels = agg_clustering.fit_predict(X.toarray())

# A Linkage Matrix identifies relations between clusters
Z = linkage(X.toarray()[:,1:20], method = 'ward')

# A dendrogram displays the Euclidean distance between points - a small distance
# groups points into the same cluster, a higher one separates them

dendro = dendrogram(Z)
plt.title('Dendrogram')
plt.ylabel('Euclidean distance')

# Display the plots
plt.show()
plt.figure(figsize=(10, 7))
plt.scatter(X.toarray()[:,0], X.toarray()[:,1], c=agg_clustering.labels_, cmap='rainbow')


# In[25]:



# Import libraries in order to perform K-Means clustering, where
# the results of the previous Agglomerative Clustering model
# reveal the ideal K number of clusters for K-means

import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

with open('/Users/meganpowers/Desktop/HeliophysicsDataset.txt') as f:
  content = f.read()

corpus = nltk.sent_tokenize(content)
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(corpus)

# Set the k number of clusters for K-means
true_k = 3

# Initialize the K-means clustering model, which clusters data into
# non-overlapping clusters to reveal similar terms

model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(X)
print("Top terms per cluster:")

# Initialize centroid points
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()

# Print the top terms in each cluster
for i in range(true_k):
  print("Cluster %d:" % i),
  for ind in order_centroids[i, :10]:
    print(' %s' % terms[ind]),
  print()


# In[26]:


"""
    get_entities
    param: sent
    return: ent1, ent2
    
    get_entities takes in a corpus by sentence and extracts entities by
    using Part of Speech tagging to get the subject and object of the sentence
    alongside their modifiers. It checks whether the subject is a Named Entity
    before returning the subject and object.

"""

############################################################# https://www.machinelearningplus.com/nlp/training-custom-ner-model-in-spacy/

def get_entities(sent):
    # Open the NER dataset in order to check the subject against entities within it
    contents = ""
    with open('/Users/meganpowers/Desktop/HeliophysicsDataset_NER.txt') as f:
        contents1 = f.read()
    
    # Initialize the subject and object to be returned
    ent1 = ""
    ent2 = ""
    
    # Initialize the dependency tag of the previous token as well as the previous token
    prv_tok_dep = ""    
    prv_tok_text = ""   
    
    # Initialize the prefix and modifier 
    prefix = ""
    modifier = ""
    
    # Save the potential subject and object dependency tags that may appear 
    SUBJECTS = ["nsubj", "nsubjpass", "csubj", "csubjpass", "agent", "expl"]
    OBJECTS = ["dobj", "dative", "attr", "oprd"]
   
    # For each individual word in a sentence
    for tok in nlp(sent):

        # Check if a token is a punctuation mark. If so, go to the next token.
        if tok.dep_ != "punct":
            
            # Check whether a token is a compound word so that modifiers can be
            # stored also.
            if tok.dep_ == "compound":
                prefix = tok.text
                
                # If the prior word was also compound, add the present word to the prior one
                if prv_tok_dep == "compound":
                    prefix = prv_tok_text + " "+ tok.text
                    # Check if the token is in the stop list
                    if str(prv_tok_text).lower() in stop_words:
                        prefix = ""
                        
        # Check whether a token is a modifier. 
        if tok.dep_.endswith("mod") == True:
            modifier = tok.text
            
            # If the previous word was also a compound, add the current word to it.
            if prv_tok_dep == "compound":
                modifier = prv_tok_text + " "+ tok.text
                if str(prv_tok_text).lower() in stop_words:
                    modifier = ""
              
        # Check if the token dependency is either in a subject or object
        if tok.dep_ in SUBJECTS or tok.dep_ in OBJECTS:
            # If the token is a subject, check if it is a Named Entity (NE)
            if tok.dep_.find("subj") == True and str(tok) in contents1:
                tok_info = tok.text
                
                # Check if the token is in the stop list
                if str(tok_info).lower() in stop_words:
                    # Set it to an empty string
                    tok_info = ""
                # Build the subject using any modifiers and prefixes as well as the actual token
                ent1 = modifier +" "+ prefix + " "+ tok_info
                prefix = ""
                modifier = ""
                prv_tok_dep = ""
            # If the token is an object
            if tok.dep_.find("obj") == True:
                # Set the token into to the text of the token 
                tok_info = tok.text 
                # Build the object using any modifiers and prefixes as well as the actual token
                ent2 = modifier + " " + prefix + " " + tok_info
        
        # Update the variables to reflect the previous token dependency
        # and previous token text
        prv_tok_dep = tok.dep_
        prv_tok_text = tok.text
        
    # Return the subject and object without leading whitespace.
    return [ent1.strip(), ent2.strip()]


# In[27]:


entity_pairs = []
contents = ""

# Open the file with the heliophysics text
with open('/Users/meganpowers/Desktop/HeliophysicsDataset.txt') as f:
  contents = f.read()

# Tokenize the text by sentence
tokens = nltk.sent_tokenize(contents)

# For each sentence token, get entities 
for i in tokens:
    entity_pairs.append(get_entities(i))

print(entity_pairs[0:1000])


# In[28]:


# Import matcher to match the root of a sentence
from spacy.matcher import Matcher

"""
    get_relation
    param: sent
    return: span.text
    
    get_relation takes in a corpus one sentence at a time and 
    uses a Matcher object to determine the root of a sentence.
    This root, which is often a verb, is the 

"""

def get_relation(sent):
    doc = nlp(sent)
    # Matcher class object 
    matcher = Matcher(nlp.vocab)
    
    # Define the pattern to discover
    # a root
    pattern = [[{'DEP':'ROOT'}], 
               [{'DEP':'prep','OP':"?"}],
               [{'DEP':'ADV','OP':"?"}],
               [{'DEP':'agent','OP':"?"}],  
               [{'POS':'ADJ','OP':"?"}]]
    
    # Add the pattern to the matcher
    matcher.add("matching_1", pattern) 
    
    # Apply the matcher and determine
    # the total no of matches
    matches = matcher(doc)
    k = len(matches) - 1
    
    # Extract the root of the sentence using the Matcher
    span = doc[matches[k][1]:matches[k][2]] 
    
    # Return the text held in the span
    return(span.text)


# In[29]:


# Apply get_relation to each sentence

relations = [get_relation(i) for i in tokens]

entity_pairs1 = entity_pairs


# In[30]:


print(entity_pairs[0][0])


# In[31]:


# Ensure that no special characters
# are left in the entity pairs. Ensure no space as well.
# This is done to create valid RDF triple IRIs

for i in range(len(entity_pairs)):
    entity_pairs[i][0] = entity_pairs[i][0].replace(" ", "")
    entity_pairs[i][0]=re.sub("[^a-zA-Z0-9]+", '', entity_pairs[i][0])
    
    entity_pairs[i][1] = entity_pairs[i][1].replace(" ", "")
    entity_pairs[i][1]=re.sub("[^a-zA-Z0-9]+", '', entity_pairs[i][1])


# In[32]:


# For each relation, if at least one part of the relation is blank,
# remove the relation.
# This is to avoid empty nodes in the KG

for i in range(len(relations)):
    if(entity_pairs[i][0] == "") or (entity_pairs[i][1] == ""):
        relations[i] = ""
    

# Ensure non-empty entities and relations

ent_list = [x for x in entity_pairs if (x[0] and x[1])]

relat_list = [x for x in relations if (x)]
"""

    ent_list[i][0] = ent_list[i][0].replace(" ", "+")
    
"""

# Replace any non-alphanumeric characters with empty space.

for i in range(len(ent_list)):
    ent_list[i][0]=re.sub("[^a-zA-Z0-9]+", '', ent_list[i][0])
    
    ent_list[i][1]=re.sub("[^a-zA-Z0-9]+", '', ent_list[i][1])

for i in range(len(relat_list)):
    relat_list[i]=re.sub("[^a-zA-Z0-9]+", '', relat_list[i])

# Ensure that the lists are of the same length
print(len(ent_list))
print(len(relat_list))


# In[33]:


# Check to see the relations and how many examples of
# each there are.

pd.Series(relat_list).value_counts()[:50]


# In[35]:


tokEnts = {}

# Extract all corresponding labels for each entity 
# in order to annotate the resulting ontology with
# the labels that correspond to Classes

for tok in nlp(contents):
    SUBJECTS = ["nsubj", "nsubjpass", "csubj", "csubjpass", "agent", "expl"]
    OBJECTS = ["dobj", "dative", "attr", "oprd"]
    if tok.dep_ in SUBJECTS or tok.dep_ in OBJECTS:
        if str(tok).lower() not in stop_words:
            curr_tok = str(tok)
            tokEnts[curr_tok] = ''

# Store each entity as a key, and each label as a value in a dictionary

for entity in sp(contents).ents:
    ent1 = str(entity)
    for key, val in tokEnts.items():
        if key.lower() in ent1.lower():
            tokEnts[key] = str(entity.label_)


# In[36]:



# Extract subject

source = [i[0] for i in ent_list]

# Extract object
target = [i[1] for i in ent_list]

# Create a dataframe that has the subject, object, relationship (source, target, edge)
# This will also be used to create RDF triples
kg_df = pd.DataFrame({'source':source, 'target':target, 'edge':relat_list})


# In[37]:


# create a directed-graph from the dataframe

G=nx.from_pandas_edgelist(kg_df, "source", "target", 
                          edge_attr=True, create_using=nx.MultiDiGraph())


plt.figure(figsize=(12,12))

# Display the Knowledge Graph
pos = nx.spring_layout(G)
nx.draw(G, with_labels=True, node_color='skyblue', edge_cmap=plt.cm.Blues, pos = pos)
plt.show()


# In[40]:


# The Knowledge Graph may be quite large. This allows for a 
# specifice edge value to be passed in, and displays 
# all sources and targets with the same edge value. 
G=nx.from_pandas_edgelist(kg_df[kg_df['edge']=="by"], "source", "target", 
                          edge_attr=True, create_using=nx.MultiDiGraph())

plt.figure(figsize=(12,12))
pos = nx.spring_layout(G, k = 0.5) # k regulates the distance between nodes
nx.draw(G, with_labels=True, node_color='skyblue', node_size=1500, edge_cmap=plt.cm.Blues, pos = pos)
plt.show()


# In[42]:


# rdflib allows for Python to RDF conversion.
# This will be used to create the Protégé ontology.

import pandas as pd
from rdflib import Graph, URIRef, Literal
import rdflib
      
# Create a dataframe from the sources, targets, and edges

d = {
    "source": source,
    "target": target,
    "edge": relat_list,
}

# Create a graph to store RDF triples

g = rdflib.Graph()

# This value corresponds to a particular ontology namespace

n = rdflib.Namespace("http://www.semanticweb.org/meganpowers/ontologies/2022/7/CfHATest#")

df1 = pd.DataFrame(d)


# In[43]:


# Create URIs from the entities and relationships by
# appending them to appropriate domain spaces

namedIndividual = URIRef('http://www.semanticweb.org/meganpowers/ontologies/2022/7/CfHATest#ASTROPHYSICS')

rdftype = URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")

rdfresource = URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#subPropertyOf")

objectProperty = URIRef("http://www.w3.org/2002/07/owl#topObjectProperty")

classProperty = URIRef("http://www.w3.org/2002/07/owl#Thing")

# Initialize the Classes in the ontology

g.add((URIRef('http://www.semanticweb.org/meganpowers/ontologies/2022/7/CfHATest#ASTROPHYSICS'), rdflib.RDFS.subClassOf, classProperty))
g.add((URIRef('http://www.semanticweb.org/meganpowers/ontologies/2022/7/CfHATest#HELIOPHYSICS'), rdflib.RDFS.subClassOf, classProperty))
g.add((URIRef('http://www.semanticweb.org/meganpowers/ontologies/2022/7/CfHATest#PAPER'), rdflib.RDFS.subClassOf, classProperty))
g.add((URIRef('http://www.semanticweb.org/meganpowers/ontologies/2022/7/CfHATest#PROJECT'), rdflib.RDFS.subClassOf, classProperty))
g.add((URIRef('http://www.semanticweb.org/meganpowers/ontologies/2022/7/CfHATest#MISSION'), rdflib.RDFS.subClassOf, classProperty))
g.add((URIRef('http://www.semanticweb.org/meganpowers/ontologies/2022/7/CfHATest#PERSON'), rdflib.RDFS.subClassOf, classProperty))
g.add((URIRef('http://www.semanticweb.org/meganpowers/ontologies/2022/7/CfHATest#ORG'), rdflib.RDFS.subClassOf, classProperty))

# For each row in the dataframe
for index, row in df1.iterrows():
    # Add triple to rdf-graph
    # Classes added temporarily as placeholders 
    g.add((URIRef(n+row["source"]), rdftype, namedIndividual))
    g.add((URIRef(n+row["edge"]), rdflib.RDFS.subPropertyOf, objectProperty))
    g.add((URIRef(n+row["source"]), URIRef(n+row["edge"]), URIRef(n+row["target"])))


# In[44]:


# Create variables for the different possible Class URIs
# Save the variables in a dictionary 

astrophysics = URIRef('http://www.semanticweb.org/meganpowers/ontologies/2022/7/CfHATest#ASTROPHYSICS')
heliophysics = URIRef('http://www.semanticweb.org/meganpowers/ontologies/2022/7/CfHATest#HELIOPHYSICS')
mission = URIRef('http://www.semanticweb.org/meganpowers/ontologies/2022/7/CfHATest#MISSION')
paper = URIRef('http://www.semanticweb.org/meganpowers/ontologies/2022/7/CfHATest#PAPER')
project = URIRef('http://www.semanticweb.org/meganpowers/ontologies/2022/7/CfHATest#PROJECT')
org = URIRef('http://www.semanticweb.org/meganpowers/ontologies/2022/7/CfHATest#ORG')
person = URIRef('http://www.semanticweb.org/meganpowers/ontologies/2022/7/CfHATest#PERSON')

classType ={
    "ASTROPHYSICS": astrophysics,
    "HELIOPHYSICS": heliophysics,
    "MISSION": mission,
    "PAPER": paper,
    "PROJECT": project,
    "ORG": org,
    "PERSON": person
    
}

# For every key and value in the dictionary that stores the tokens and labels

for key, value in tokEnts.items():
    
    # For every index and row in the dataframe that stores the RDF triple components
    for index, row in df1.iterrows():
        
        # Check if the key is a source or target value
        # Then, save the corresponding URI to the graph
        if key in row["source"]:
            if value in classType.keys():
                currURI = classType[value]
                g.add((URIRef(n+row["source"]), rdftype, currURI))
                break
        if key in row["target"]:
            if value in classType.keys():
                currURI = classType[value]
                g.add((URIRef(n+row["target"]), rdftype, currURI))
                break


# In[45]:



# Finally, write the graph to the ontology using an appropriate address.
output_address ="/Users/meganpowers/Desktop/CfHATest.owl"   
print(g.serialize())
g.serialize(destination = output_address)


# In[111]:


with open ("CfHATest.owl") as a:
    print(a)


# In[ ]:




