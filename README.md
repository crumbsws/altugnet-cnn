 🧠 CIFAR-10 Image Classifier with Adversarial Testing (PyTorch)

This project is a convolutional neural network (CNN) built using PyTorch that classifies images from the CIFAR-10 dataset. It includes:

    A custom CNN model

    Automatic training and model persistence

    Standard testing on unseen data

    Simple adversarial attack using FGSM (Fast Gradient Sign Method)

    Command-line based interactive testing

📦 Features

    Loads CIFAR-10 with torchvision

    Builds a CNN from scratch

    Trains the model from scratch or loads it from disk

    Evaluates model on test dataset

    Applies adversarial perturbations to test robustness

    Uses colorama for colored terminal output

    Displays visual comparison between clean and adversarial images
