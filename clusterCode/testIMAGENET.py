from datasets import load_dataset

# If the dataset is gated/private, make sure you have run huggingface-cli login
dataset = load_dataset("imagenet-1k")


print("yes this did gather imagenet!")

# Select the first row in the dataset
sample = dataset[0]

# Split up the sample into two variables
datapt, label = sample['image'], sample['label']

print(label)
print(datapt)



exit()