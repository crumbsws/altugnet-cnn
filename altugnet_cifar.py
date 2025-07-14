import torch
import os
from colorama import Fore, Back, Style
from PIL import Image
from torch import nn,save,load
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse

# Loading Data
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.CIFAR10(root="data", download=True, train=True, transform=transform)
test_dataset = datasets.CIFAR10(root="data", download=True, train=False, transform=transform)

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)




cifar10_labels = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

# Define the image classifier model
class ImageClassifier(nn.Module):
    def __init__(self):
        super(ImageClassifier, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 10) # Adjusted input size
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classifier = ImageClassifier().to(device)

#add code here to check 'ıf model_state.pt, exısts then just load the model
def trainModel():
  if not os.path.exists('model_state_cifar.pt'):
    # Create an instance of the image classifier model

    

    # Define the optimizer and loss function
    optimizer = Adam(classifier.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    loss_list = []

    # Train the model
    for epoch in range(20):  # Train for 10 epochs
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()  # Reset gradients
            outputs = classifier(images)  # Forward pass
            loss = loss_fn(outputs, labels)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights

        loss_list.append(loss.item())
        print(Fore.YELLOW + f"Epoch:{epoch} loss is {loss.item()}")
        if epoch > 0:
          print(Fore.YELLOW +  f"Comparing old loss to the new one: {loss_list[epoch - 1] - loss_list[epoch]} ")
        print(Fore.YELLOW + '--------------------------------------')

    # Save the trained model
    torch.save(classifier.state_dict(), 'model_state_cifar.pt')

  else:

    # Load the saved model
    with open('model_state_cifar.pt', 'rb') as f:
      classifier.load_state_dict(load(f))



def testDatasetClassification():
    test_loss_list = []
    test_accuracy_list = []

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = classifier(images)  # Forward pass
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(outputs, labels)  # Compute loss
        test_loss_list.append(loss.item())

        # Compute test accuracy
        _, predicted = torch.max(outputs, 1)
        total = labels.size(0)
        correct = (predicted == labels).sum().item()
        accuracy = correct / total
        test_accuracy_list.append(accuracy)

    avg_test_loss = sum(test_loss_list) / len(test_loss_list)
    print(Fore.RED + f"Resultant accuracy from the plain test dataset: {sum(test_accuracy_list) / len(test_accuracy_list)}")
    print(Fore.YELLOW + f"Average test loss: {avg_test_loss}")

    # Make sure second_test_accuracy_list is defined before this line
  



def testAdversarialClassification():
  second_test_accuracy_list = []
  altered_example_img = None
  non_altered_example_img = None
  non_altered_example_label = None

  for images, labels in test_loader:



      images, labels = images.to(device), labels.to(device)
      images.requires_grad = True

      non_altered_example_img = images
      pre_adv_true_label = labels.item()
       

      outputs_OG = classifier(images)  # Forward pass
      loss_fn = nn.CrossEntropyLoss()
      loss = loss_fn(outputs_OG, labels)
      loss.backward()
      grad_adv = images.grad
      epsilon = 0.01


      #take sıgn of each element
      grad_adv = torch.sign(grad_adv)
      x_adv = images + epsilon * grad_adv
      altered_example_img = x_adv

      outputs_adv = classifier(x_adv)  # Forward pass
      correct_label= labels.item()

      #now lets actually compute test accuracy
      _, adv_predicted = torch.max(outputs_adv, 1)
      _, actual_predicted = torch.max(outputs_OG, 1)
      non_altered_example_label = adv_predicted.item()
      adv_total = labels.size(0)
      post_adv_false_label = adv_predicted.item()
      adv_correct = (adv_predicted == labels).sum().item()
      adv_accuracy = adv_correct / adv_total
      second_test_accuracy_list.append(adv_accuracy)



      images.grad.zero_()
  


  print(Fore.RED + f"Resultant accuracy from the adversarial attack: {sum(second_test_accuracy_list) / len(second_test_accuracy_list)} ({cifar10_labels[pre_adv_true_label]})")


  fig, axs = plt.subplots(1, 2, figsize=(10, 5))

  axs[0].imshow(non_altered_example_img.squeeze().cpu().detach().permute(1, 2, 0).numpy())
  axs[0].set_title(cifar10_labels[actual_predicted])
  axs[0].axis('off')

  axs[1].imshow(altered_example_img.squeeze().cpu().detach().permute(1, 2, 0).numpy())
  axs[1].set_title(cifar10_labels[post_adv_false_label])
  axs[1].axis('off')

  plt.tight_layout()
  plt.show()




def testTargetedAdversarialClassification():
  second_test_accuracy_list = []
  altered_example_img = None
  non_altered_example_img = None
  non_altered_example_label = None

  for images, labels in test_loader:


      images, labels = images.to(device), labels.to(device)
      images.requires_grad = True

      non_altered_example_img = images
      pre_adv_true_label = labels.item()
       

      outputs_OG = classifier(images)  # Forward pass
      first_output_neuron = outputs_OG[0, 0]
      first_output_neuron.backward()
      grad_adv = images.grad
      epsilon = 0.1


      #take sıgn of each element
      grad_adv = torch.sign(grad_adv)
      x_adv = images + epsilon * grad_adv
      altered_example_img = x_adv

      outputs_adv = classifier(x_adv)  # Forward pass
      correct_label= labels.item()

      #now lets actually compute test accuracy
      _, adv_predicted = torch.max(outputs_adv, 1)
      _, actual_predicted = torch.max(outputs_OG, 1)

      adv_total = labels.size(0)
      post_adv_false_label = adv_predicted.item()
      adv_correct = (adv_predicted == labels).sum().item()
      adv_accuracy = adv_correct / adv_total
      second_test_accuracy_list.append(adv_accuracy)



      images.grad.zero_()
  


  print(Fore.RED + f"Resultant accuracy from the adversarial attack: {sum(second_test_accuracy_list) / len(second_test_accuracy_list)} ({cifar10_labels[pre_adv_true_label]})")


  fig, axs = plt.subplots(1, 2, figsize=(10, 5))

  axs[0].imshow(non_altered_example_img.squeeze().cpu().detach().permute(1, 2, 0).numpy())
  axs[0].set_title(cifar10_labels[actual_predicted])
  axs[0].axis('off')

  axs[1].imshow(altered_example_img.squeeze().cpu().detach().permute(1, 2, 0).numpy())
  axs[1].set_title(cifar10_labels[post_adv_false_label])
  axs[1].axis('off')

  plt.tight_layout()
  plt.show()



#
#pass ımage through network
#calculate the loss
#calculate the gradıent of the loss. loss.backward()
#then store the gradıent wrt x usıng grad = x.grad
# x_adv = x + epsılon* sıng(grad)

print(Fore.LIGHTGREEN_EX + """
        _ _                          _   
       | | |                        | |  
   __ _| | |_ _   _  __ _ _ __   ___| |_ 
  / _` | | __| | | |/ _` | '_ \ / _ \ __|
 | (_| | | |_| |_| | (_| | | | |  __/ |_ 
  \__,_|_|\__|\__,_|\__, |_| |_|\___|\__|
                     __/ |               
                    |___/                
""")

while True:
  trainModel()
  print(Fore.LIGHTGREEN_EX + "Training complete")
  print(Fore.LIGHTGREEN_EX + "Type test or aa to continue: ")
  command = input()
  if(command == "test"):
    print(Fore.LIGHTGREEN_EX +  "Testing model")
    testDatasetClassification()
    print(Fore.LIGHTGREEN_EX + "\nModel tested"  +  Fore.RESET) 
  elif(command == "aa"):
    print(Fore.LIGHTGREEN_EX + "Testing adversarial attack")
    testAdversarialClassification()
    print(Fore.LIGHTGREEN_EX + "\nAdversarial attack complete" +  Fore.RESET)
  else:
    print("Invalid command")


# Perform inference on an image
