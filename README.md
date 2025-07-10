🧠 Number Image Classifier & Adversarial Attack Generator

This repository contains a PyTorch-based deep learning project that trains a neural network to classify handwritten digit images (e.g., MNIST), and demonstrates adversarial attacks (e.g., FGSM) that exploit model vulnerabilities by subtly modifying inputs to fool the classifier.
🔍 Project Overview

    Image Classifier
    A convolutional neural network (CNN) that learns to classify digit images from the MNIST dataset.

    Adversarial Attack Generator
    Implementation of the Fast Gradient Sign Method (FGSM) to generate adversarial examples that reduce classification accuracy.

📂 Contents

.
├── classifier.py         # Model definition & training
├── attack.py             # Adversarial attack logic
├── test.py               # Evaluation on clean and adversarial data
├── utils.py              # Utility functions for visualization, loading, etc.
├── README.md             # Project documentation
└── requirements.txt      # Required dependencies

🧪 Features

    Simple, modular CNN architecture for number classification

    Training and testing on MNIST dataset

    Adversarial attack implementation (FGSM)

    Accuracy comparison between clean and adversarial inputs

    Visual inspection of perturbed images
