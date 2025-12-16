import pandas as pd
from torch import nn as nn
import torch
import os
from PIL import Image
from sklearn import preprocessing
from torchvision import transforms
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


class DinoVisionTransformerClassifier(nn.Module):
    def __init__(self, num_classes):
        super(DinoVisionTransformerClassifier, self).__init__()
        
        self.transformer = torch.hub.load('/work/ec285/ec285/krodrig/hub/facebookresearch_dinov2_main/','dinov2_vitb14',source='local', force_reload='False')

        self.classifier = nn.Sequential(
            nn.Linear(self.transformer.num_features, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.transformer(x)
        x = self.transformer.norm(x)
        x = self.classifier(x)
        return x

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, label_encoder, folder_path = "", transform=None):
        self.data = data
        self.transform = transform
        self.folder_path = folder_path
        self.label_encoder = label_encoder
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        
        img = Image.open(os.path.join(self.folder_path, sample['path'])).convert('RGB')
#         print(idx)
        if self.transform:
            img = self.transform(img)
        
        return {
            'uid':sample['uid'],
            'label_name':sample['class'],
            'img': img,
            'label': torch.as_tensor(self.label_encoder.transform([sample['class']])[0])
        }

def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(train_data_loader):
        # Every data instance is an input + label pair
        inputs = data['img'].to(device)
        labels = data['label'].type(torch.LongTensor).to(device)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        del loss, inputs, labels, outputs
        if i % 100 == 99:
            last_loss = running_loss / 100 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(train_data_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss

#data_df_rel = pd.read_csv("data/rel.csv")[['class', 'uid', 'path']]
#data_df_med = pd.read_csv("data/med.csv")[['class', 'uid', 'path']]
#data_df_fur = pd.read_csv("data/fur.csv")[['class', 'uid', 'path']]

data_df = pd.read_csv("data/trainingdata.csv")[['class', 'uid', 'path']]
# data_df = pd.concat([data_df_rel, 
#     data_df_med,
#     data_df_fur], 
#     ignore_index=True)


le = preprocessing.LabelEncoder()
labels = data_df['class']
le.fit(labels)
print(le)


train_transform = transforms.Compose([
    transforms.Resize((518,518)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.2)
])


val_transform = transforms.Compose([
    transforms.Resize((518,518)),
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.2)
])

#Split dataset into train and validation
train_indices, val_indices = train_test_split(list(range(len(data_df))), random_state=42,
                                              test_size=0.2, stratify=data_df['class'])

train_dataset = CustomDataset(data_df.iloc[train_indices], le, transform = train_transform)
val_dataset = CustomDataset(data_df.iloc[val_indices], le, transform = val_transform)

#Create DataLoader
train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True)
val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=2, shuffle=False)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

#read the number of classes
le = preprocessing.LabelEncoder()
le.classes_ = np.load("listClasses.npy", allow_pickle=True)
# print(len(le.classes_))

model = DinoVisionTransformerClassifier(num_classes = len(le.classes_))
model.to(device)


loss_fn = nn.CrossEntropyLoss()

# optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=0.000001)

start_from_epoch = 0


if start_from_epoch == 0:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/DINOv2_trainer_{}'.format(timestamp))


EPOCHS = 25

best_vloss = 1_000_000.
model_path = ""
for epoch_number in range(start_from_epoch, start_from_epoch+EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch_number, writer)
    print("epoch training completed. Starting evaluating the epoch...")

    running_vloss = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, vdata in enumerate(val_data_loader):
            vinputs = vdata['img'].to(device)
            vlabels = vdata['label'].type(torch.LongTensor).to(device)
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch_number + 1)
    writer.flush()

    # Track best performance, and save the model's state

    print("********Loss: "+str(best_vloss)+" and avg_vloss: "+str(avg_vloss))

    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = 'model_{}_{}'.format(timestamp, epoch_number)
        print("->new best_vloss: "+str(best_vloss))

    torch.save(model.state_dict(), model_path)

    epoch_number += 1

#rename our last model to model
print("rename the model:"+model_path)
os.rename(model_path, 'model')


