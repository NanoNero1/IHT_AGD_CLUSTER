import torch
import torchvision.transforms
from datasets import load_dataset
import time

# If the dataset is gated/private, make sure you have run huggingface-cli login
timeOne = time.time()
dataset = load_dataset("imagenet-1k")


print("yes this did gather imagenet!")

timeTwo = time.time()

print(timeTwo - timeOne)

# Select the first row in the dataset
sample = dataset['train'][0]

# Split up the sample into two variables
datapt, label = sample['image'], sample['label']

print(label)
print(datapt)


### SOURCE: https://medium.com/@ricodedeijn/image-classification-computer-vision-from-scratch-pt-3-9d5fbcf3c363
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        # Loads the dataset that needs to be transformed
        self.dataset = load_dataset("imagenet-1k", split=f"{dataset}")

    def __getitem__(self, idx):
        # Sample row idx from the loaded dataset
        sample = self.dataset[idx]
        
        # Split up the sample example into an image and label variable
        data, label = sample['image'], sample['label']
        
        transform = transforms.Compose([
            transforms.Resize((256, 256)),  # Resize to size 256x256
            Lambda(lambda x: x.convert("RGB") if x.mode != "RGB" else x),  # Convert all images to RGB format
            transforms.ToTensor(),  # Transform image to Tensor object
        ])
        
        # Returns the transformed images and labels
        return transform(data), torch.tensor(label)

    def __len__(self):
        return len(self.dataset)

# Call the class to populate variable train_set with the train data
train_set = MyDataset('train')
test_set = MyDataset('test')

BATCH_SIZE = 256

train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE)

print("so far so good!")

exit()