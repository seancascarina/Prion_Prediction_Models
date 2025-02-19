# Prion Prediction Models
**Project Goal: to determine whether "prion" prediction efficacy can be improved by training machine learning models or fine-tuning existing models on the limited set of known protein sequences with prion activity.**

**Background:** "Prions" are unusual proteins that are able to adopt multiple 3D shapes and then pass their shape information on to other copies of the same protein. This 3D shape information can then be passed from one organism to another, making them both heritable and infectious. A common goal in biology is to predict protein function from a protein's sequence features. Prions are interesting biological molecules, but few known examples exist, and it is very challenging to find or predict new prions. Fortunately, the sequence features that determine prion activity are usually simpler compared to most proteins: the main determinant of prion activity is the relative amounts of the 20 protein building blocks called "amino acids". In other words, if we represent these 20 building blocks as letters from the alphabet, the total percentages of each letter in a sequence is more important than the actual order of the letters. Therefore, prion activity can be modeled based on the percent composition of each amino acid in a protein sequence, resulting in 20 features. In contrast, proteins whose activity depends on both the type of building block and its position in the sequence could have up to 20<sup>*N*</sup> features, where *N* represents the length of the protein. Since most proteins are hundreds of amino acids long, that's a lot of features!

An ongoing challenge in the biological data science field is to identify these rare prion proteins from among thousands or tens of thousands of "normal", non-prion proteins. Multiple prion prediction algorithms have been developed, and these algorithms are already somewhat effective at this task. Therefore, I will experiment with training modern machine learning classifiers for this task and compare their performance to the performance of existing algorithms.