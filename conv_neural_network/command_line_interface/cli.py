import os
import sys
sys.path.append("../")

def choose_cnn_model():
    print("Choose the number of the model to train:\n")
    print(" 1. Numpy Multiple Convolutional Layers \n 2. Numpy Double Convolutional Layer \n 3. Pytorch Multiple Convolutional Layers \n 4. Pytorch Multiple Convultional Layers And Fully Connected Layers \n")

    while True:
        model_choice = input("Enter the number of the model you would like to train (eg. 1 for Numpy Multiple Convolutional Layers): ")

        try:
            model = float(model_choice)
            if model == 1:
                return "Numpy Multiple Convolutional Layers"
            elif model == 2:
                return "Numpy Double Convolutional Layer"
            elif model == 3:
                return "Pytorch Multiple Convolutional Layers"
            elif model == 4:
                return "Pytorch Multiple Convultional Layers And Fully Connected Layers"
            else:
                print("Invalid input. Please enter a valid number.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")

        print()

def choose_output_nodes():
    # print("\n\nChoose the number of output nodes for the model:")
    # print("Aim for the number of output nodes to be the same as the number of classes in your dataset.")

    while True:
        output_nodes = input("\n\nEnter the number of output nodes for the model: ")

        try:
            output_nodes = int(output_nodes)
            if output_nodes > 0:
                return output_nodes
            else:
                print("Invalid input. Please enter a valid number.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")

        print()

def choose_batch_size():
    print("The batch size is the number of samples that will be passed through the model at once.")

    while True:
        batch_size = input("\n\nEnter the batch size for the model: ")

        try:
            batch_size = int(batch_size)
            if batch_size > 0:
                return batch_size
            else:
                print("Invalid input. Please enter a valid number.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")

        print()
    
def upload_training_data_zip(output_nodes):
    print("Upload a zip file containing the training data.")
    print("The zip file should contain the training images and labels.")
    print("The images should be in a folder with the class name.")a

    while True:
        upload_choice = input("\n\nEnter 'y' if you have uploaded the zip file to the 'data' folder: ")

        if upload_choice == "y":
            return
        else:
            print("Invalid input. Please enter 'y'.")

        print()

if __name__ == "__main__":
    model = choose_cnn_model()
    output_nodes = choose_output_nodes()
    batch_size = choose_batch_size()
    print(f"Model {model} chosen.")
    print(f"Output nodes: {output_nodes}")
    print(f"Batch size: {batch_size}")