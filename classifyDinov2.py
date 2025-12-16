import argparse
import os
from torch import nn as nn
import torch

from PIL import Image
from sklearn import preprocessing

import numpy as np
import pandas as pd
from torchvision import transforms
import shutil

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
        # this will give me an embedding array - 512 values which can be later
        # mapped into a 3 - value through UMAP or other similarity algorithm, e.g. PCA
        
        x = self.transformer.norm(x)
        
        x = self.classifier(x)
        return x

    def getTransformers(self, x):
        x = self.transformer(x)
        # this will give me an embeddings array 
        # mapped into a 3 - value through UMAP or other similarity algorithm, e.g. PCA        
        #should normalise the transforms
        x = self.transformer.norm(x)
        # x = self.classifier(x)
        return x


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, label_encoder, folder_path = "Documents\\transformed", transform=None):
        self.data = data
        self.transform = transform
        self.folder_path = folder_path
        self.label_encoder = label_encoder
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        
        img = Image.open(os.path.join(self.folder_path, sample['uid'])).convert('RGB')
        if self.transform:
            img = self.transform(img)
        
        return {
            'uid': sample['uid'],
            'img': img
        }


def create_output_folders(root, classes):
    output_dir = "categories"
    output_dir_path = os.path.join(root, output_dir)
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)

    for cls in classes:
        class_path = os.path.join(output_dir_path, cls)
        if not os.path.exists(class_path):
            os.makedirs(class_path)


def run_classification(image_folder_path, model_fpath, label_enc_fpath, output_folder, csv):
    # import label encoder for the classes
    le = preprocessing.LabelEncoder()
    le.classes_ = np.load(label_enc_fpath, allow_pickle=True)
    
    # create output folders if not created
    if output_folder:
        create_output_folders(args.output_folder, le.classes_)

    # validation transform for the images.
    # images must be resized to 518x518
    val_transform = transforms.Compose([
        transforms.Resize((518,518)),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.2)
    ])

    # create a dataset / dataloader from the image folder
    data_df = pd.DataFrame({'uid':os.listdir(image_folder_path)})
    dataset = CustomDataset(data_df, le, image_folder_path, val_transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    # init model and import the weights
    model = DinoVisionTransformerClassifier(len(le.classes_ ))
    model.load_state_dict(torch.load(model_fpath, map_location=torch.device('cpu')))
    
    # storing results for csv
    results_df = pd.DataFrame()

    n_batches = len(dataloader)
    
    print("Processing...")

    # Validation mode
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            # compute progress
                
            print(str(round(100*i/n_batches, 2)) + "%", end="\r")
                
            print(data)
            # input the image to the model
            inp = data['img']
            
            outputs = model(inp)
            outputs = outputs.detach().cpu()
            print(outputs.shape)
            print(outputs)

            # get softmax as probabilities for each class
            predicted = torch.softmax(outputs, dim=1)
            #KARINA has changed from here onwards
            #The tensor itself is 2-dimensional
            #the tensor is size [2,40] meaning 
            #2 rows and 40 columns
            print(predicted.shape)
            print(predicted)

            
            #I store this predictions in a data frame which are stored in the CSV
            # temp_t = pd.DataFrame(predicted)
            # results_df = pd.concat([results_df, temp_t])

            # compute top 3 predicted classes
            top3 = torch.topk(predicted, k=3)
            
            top_lab = np.array([np.array(le.inverse_transform(i)) for i in top3.indices.numpy()])
            top_conf = top3.values.numpy()
            print(data['uid'])
            print ("A",top_lab[:, 0]," ",top_conf[:,0])
            print ("B",top_lab[:, 1]," ",top_conf[:,1])
            print ("C",top_lab[:, 2]," ",top_conf[:,2])
            
            # create dataframe for the batch
            temp_df = pd.DataFrame({
                'uid': data['uid'],
                'predicted1': top_lab[:, 0],
                'confidence1': top_conf[:,0],
                'predicted2': top_lab[:, 1],
                'confidence2': top_conf[:,1],
                'predicted3': top_lab[:, 2],
                'confidence3': top_conf[:,2],
            })


            transformers = model.getTransformers(inp)
            transformersDF = pd.DataFrame(transformers)

            print(transformersDF)
            # temp_df.join(temp_t)
            # print(results_df)
            print(temp_df)

            new = temp_df.join(transformersDF)
            print("Merged")
            print(new)


            # store the results
            results_df = pd.concat([results_df, new])

            # if output_folder:
            #     # copy the images to the class folder
            #     output_dir = "categories"

            #     for id, cls in zip(data['uid'], top_lab[:, 0]):
            #         src_img_path = os.path.join(image_folder_path, id)
            #         output_img_path = os.path.join(output_folder, output_dir, cls, id)
            #         shutil.copyfile(src_img_path, output_img_path)

    print("Completed.")
    results_df.reset_index(inplace=True, drop=True)

    if csv:
        # save results to csv
        results_df.to_csv(csv)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='DINOv2 classifier',
                    description='This script uses DINOv2 based fine-tuned classifier on a collection of images and produces a csv with the results or copies the images into different folders based on the predicted class. The script runs on CPU and RAM.',
                    epilog='At least one of the -o or -c must be specified.')
    
    # define script args
    arg_ifp = parser.add_argument('image_folder_path', help="path to the folder that contains all the images.")
    arg_mfp = parser.add_argument('-m', '--model_fpath', default="model", help="path to the model to be used. Defaults to 'model'")
    arg_lefp = parser.add_argument('-l', '--label_enc_fpath', default="classes.npy", help="path to the label encoder file to be used. Defaults to 'classes.npy'")

    arg_ofp = parser.add_argument('-o', '--output_folder',
                        help="path to the folder in which sorted images would be saved. If not specified, the images won't be moved")

    arg_cfp = parser.add_argument('-c', '--csv', 
                        metavar="CSV_FPATH",
                        help="path to the csv in which the results are stored. In this folder, the script will create another folder and subfolders within it for each category. If not specified, the results won't be saved.")

    args = parser.parse_args()

    # validate args
    if not os.path.isdir(args.image_folder_path):
        raise argparse.ArgumentError(arg_ifp, "must be a folder path.")
    
    if not os.path.isfile(args.model_fpath):
        raise argparse.ArgumentError(arg_mfp, "must be a file path.")
    
    if not os.path.isfile(args.label_enc_fpath):
        raise argparse.ArgumentError(arg_lefp, "must be a file path.")
    
    if args.output_folder and not os.path.isdir(args.output_folder):
        raise argparse.ArgumentError(arg_ofp, "must be a file path.")
    
    if not (args.output_folder or args.csv):
        parser.error('no action requested, add --output_folder or --csv.')
    
    
    # run classification
    run_classification(image_folder_path = args.image_folder_path,
                       model_fpath = args.model_fpath,
                       label_enc_fpath = args.label_enc_fpath,
                       output_folder = args.output_folder,
                       csv = args.csv)


