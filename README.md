# Intro

This repository was created to gather all the code used to study Quantum Deep Q-Learning, in a hybrid clasical-quantum fashion, for the project of the discipline MS877 - an undergraduate course in Applied Mathematics of the Unversity of Campinas - 2022.

It is split into 2 folders: variational_classifier and quantum_dqn. The former contains the code available in [1], which we used as a step to learn the VQC used in the quantum DQN code, which is itself contained in the latter folder, which code was mainly based on [2].

[1] https://pennylane.ai/qml/demos/tutorial_variational_classifier.html

[2] https://www.tensorflow.org/quantum/tutorials/quantum_reinforcement_learning

# Setup

In order to run the code, first install the dependencies:
```
pip install -r requirements.txt
```

Now go to the folder you wish to run:

```
cd variational_classifier
```

or 

```
cd quantum_dqn
```

And run the python files:

```
python3 classifier.py
```

or

```
python3 quantum_dqn.py
```

*Obs:* Note that the 'data' folders contain the charts generated by my runs, and included in the article written for this project:
https://www.ime.unicamp.br/~mac/db/2022-1S-165023.pdf
