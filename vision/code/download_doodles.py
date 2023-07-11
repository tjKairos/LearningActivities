import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import urllib.request
import os
import os.path as path
from glob import glob



data_folder = path.join(path.dirname(path.dirname(__file__)), "data")

def download_objects(objects = [
    "The Eiffel Tower",
    "alarm clock",
    "axe",
    "banana",
    "baseball",
], show = False, directory = data_folder) -> dict[str, np.ndarray]:
    remote_base = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/"
    all_data = {}
    for obj in objects:
        remote_name = f"{remote_base}{obj.replace(' ', '%20')}.npy"
        file_name = path.join(directory, f"{obj}.npy")

        if not os.path.exists(file_name):
            # Download the file
            urllib.request.urlretrieve(remote_name, file_name)

        # Load the data
        data = np.load(file_name, allow_pickle = True)
        data = data.reshape((-1, 28, 28))
        print(data.shape)
        all_data[obj] = data

        if show:
            # Show the first 20 images
            for i in range(20):
                plt.subplot(4, 5, i + 1)
                plt.imshow(data[i], cmap = "gray")
                plt.axis("off")
            plt.show()

    return all_data

def load_data(percent_data: float = 1.0, classes = ["The Eiffel Tower", "alarm clock", "axe", "banana"],
              show: bool = False) -> torch.utils.data.DataLoader:
    class_dict = download_objects(classes, show = False)
    data = []
    labels = []
    for i, (class_name, class_data) in enumerate(class_dict.items()):
        class_data = class_data[:int(percent_data * class_data.shape[0])]
        data.append(class_data)
        numeric_labels = np.full(class_data.shape[0], i)
        one_hot_labels = np.zeros((class_data.shape[0], len(classes)))
        one_hot_labels[np.arange(class_data.shape[0]), numeric_labels] = 1
        labels.append(one_hot_labels[:class_data.shape[0]])

    data = np.concatenate(data)
    labels = np.concatenate(labels)
    print(data.shape)
    print(labels.shape)

    # Randomize the order of the data
    perm = np.random.permutation(data.shape[0])
    data = data[perm]
    labels = labels[perm]

    # Convert to torch tensors
    data = torch.from_numpy(data).float() / 255.0
    labels = torch.from_numpy(labels).float()

    # Split into train, validation, and test
    val_count = int(10000 * percent_data)
    test_count = int(20000 * percent_data)
    train_data = data[:-(val_count + test_count)]
    train_labels = labels[:-(val_count + test_count)]
    val_data = data[-(val_count + test_count):-test_count]
    val_labels = labels[-(val_count + test_count):-test_count]
    test_data = data[-test_count:]
    test_labels = labels[-test_count:]

    # Take the training data and validation data and randomly move the images around
    # This will help the model generalize better
    shift = 10
    shifted_training = torch.zeros_like(train_data)
    for i in range(train_data.shape[0]):
        # Randomly move the image around
        dx = np.random.randint(-shift, shift)
        dy = np.random.randint(-shift, shift)
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        shifted_training[i] = torch.from_numpy(cv2.warpAffine(train_data[i].detach().numpy(), M, (28, 28)))
        # Add random noise
        train_data[i] += torch.randn_like(train_data[i]) * 0.2
        shifted_training[i] += torch.randn_like(train_data[i]) * 0.2
    train_data = torch.concat([train_data, shifted_training], dim = 0)
    train_labels = torch.concat([train_labels, train_labels], dim = 0)
    for i in range(val_data.shape[0]):
        # Randomly move the image around
        dx = np.random.randint(-shift, shift)
        dy = np.random.randint(-shift, shift)
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        val_data[i] = torch.from_numpy(cv2.warpAffine(val_data[i].detach().numpy(), M, (28, 28)))
        # Add random noise
        val_data[i] += torch.randn_like(val_data[i]) * 0.2

    # Create the dataset
    train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
    val_dataset = torch.utils.data.TensorDataset(val_data, val_labels)
    test_dataset = torch.utils.data.TensorDataset(test_data, test_labels)

    # Create the dataloader
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = 512, shuffle = True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size = 1024)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = 1024)

    for batch in train_dataloader:
        print(batch[0].shape)
        print(batch[1].shape)
        if show:
            for i in range(20):
                plt.subplot(4, 5, i + 1)
                plt.imshow(batch[0][i], cmap = "gray")
                plt.axis("off")
            plt.show()
        break

    return train_dataloader, val_dataloader, test_dataloader


def labeled_classes(classes: list[str]) -> list[str]:
    return [class_name for class_name in classes if len(glob(f"{data_folder}/{class_name}/*.npy")) > 0]


def load_labeled(classes: list[str] = ["alarm clock", "axe", "banana"],
                 show: bool = False) -> torch.utils.data.DataLoader:
    data = []
    labels = []
    for i, class_name in enumerate(classes):
        class_data = []
        for f in glob(f"{data_folder}/{class_name}/*.npy"):
            single_data = np.load(f)
            # Swap W and H axes
            single_data = np.swapaxes(single_data, 1, 2)
            class_data.append(single_data)
        if len(class_data) == 0:
            print("No data found for class", class_name)
            continue
        class_data = np.concatenate(class_data)
        class_data = class_data[:, :, :, 0]
        numeric_labels = np.full(class_data.shape[0], i)
        one_hot_labels = np.zeros((class_data.shape[0], len(classes)))
        one_hot_labels[np.arange(class_data.shape[0]), numeric_labels] = 1
        data.append(class_data)
        labels.append(one_hot_labels[:class_data.shape[0]])

    data = np.concatenate(data)
    labels = np.concatenate(labels)

    # Randomize the order of the data
    perm = np.random.permutation(data.shape[0])
    data = data[perm]
    labels = labels[perm]

    # Convert to torch tensors
    data = torch.from_numpy(data).float()
    labels = torch.from_numpy(labels).float()

    val_count = int(len(data) * 0.1)
    test_count = int(len(data) * 0.1)

    # Split into train, validation, and test
    train_data = data[:-(val_count + test_count)]
    train_labels = labels[:-(val_count + test_count)]
    val_data = data[-(val_count + test_count):-test_count]
    val_labels = labels[-(val_count + test_count):-test_count]
    test_data = data[-test_count:]
    test_labels = labels[-test_count:]

    # Take the training data and validation data and randomly move the images around
    # This will help the model generalize better
    shift = 10
    shifted_training = torch.zeros_like(train_data)
    for i in range(train_data.shape[0]):
        # Randomly move the image around
        dx = np.random.randint(-shift, shift)
        dy = np.random.randint(-shift, shift)
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        shifted_training[i] = torch.from_numpy(cv2.warpAffine(train_data[i].detach().numpy(), M, (28, 28)))
        # Add random noise
        train_data[i] += torch.randn_like(train_data[i]) * 0.2
        shifted_training[i] += torch.randn_like(train_data[i]) * 0.2
    train_data = torch.concat([train_data, shifted_training], dim = 0).clamp(0, 1)
    train_labels = torch.concat([train_labels, train_labels], dim = 0)
    for i in range(val_data.shape[0]):
        # Randomly move the image around
        dx = np.random.randint(-shift, shift)
        dy = np.random.randint(-shift, shift)
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        val_data[i] = torch.from_numpy(cv2.warpAffine(val_data[i].detach().numpy(), M, (28, 28)))
        # Add random noise
        val_data[i] += torch.randn_like(val_data[i]) * 0.2
    val_data = val_data.clamp(0, 1)

    # Create the dataset
    train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
    val_dataset = torch.utils.data.TensorDataset(val_data, val_labels)
    test_dataset = torch.utils.data.TensorDataset(test_data, test_labels)

    # Create the dataloader
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = 512, shuffle = True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size = 1024)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = 1024)

    for batch in train_dataloader:
        print(batch[0].shape)
        print(batch[1].shape)
        if show:
            for i in range(8):
                plt.subplot(4, 5, i + 1)
                plt.imshow(batch[0][i], cmap = "gray")
                plt.axis("off")
            plt.show()
        break

    return train_dataloader, val_dataloader, test_dataloader


if __name__ == "__main__":
    # data = download_objects(show = True)
    data = load_labeled(show = True)
    # for image, label in data:
    #     print(image.shape)
    #     print(label)
    #     break
    print(data)