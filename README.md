# Supplementary code for the paper: Expanding Perspectives to Improve Access to Visual Archives through Multimodal Image Enrichment


This repository contains supplementary code for the paper

> Karina Rodriguez Echavarria and Myrsini Samaroudi. 2025. Expanding Perspectives to Improve Access to Visual Archives through Multimodal Image Enrichment. ACM Journal of Computing and Cultural Heritage (November 2025). https://doi.org/10.1145/3771993

## Abstract


The research tackles key challenges for improving the discovery of large-scale visual collections within the Cultural Heritage (CH) domain, particularly in museums and archives. Its contribution is a multimodal content understanding approach designed for image collections that lack relevant metadata, hindering effective discovery. The proposed method utilises AI-assisted image classification and unified vision-language understanding, combining visual features with semantic context to generate rich and meaningful metadata. The proposed approach enables experts to enrich and visualise large-scale datasets of image collections by assigning both expert and non-expert labels, aligning with FAIR principles (Findable, Accessible, Interoperable, Reusable). Thus, the novel workflow broadens access to this visual material for diverse audiences through search and browse interfaces in a web browser. The proposed approach is demonstrated using the previously unclassified Design Archives’ glass plate negatives dataset, which consists of approximately 
10,000 digitised images depicting 20th-century historical designs. Through an AI workflow, the dataset is enriched with expert and non-expert information. Users can search and browse results using 2D and 3D visualisations, as well as text-based search. The research also explores the advantages and current limitations of the proposed visualisation approach in creating more meaningful search and browsing functionalities for large-scale CH collections. The results demonstrate that while 3D visualisations offer more affordances than their 2D counterpart, users require further support to interact with the large-scale datasets meaningfully. Hence, there is a need for discovery interfaces that support interactivity, visual cues, and text-based search to enhance the users’ discovery journey.


## Getting started: Using the model
The paper presents an approach for fine-tuning an existing foundation model, [Dinov2](https://dinov2.metademolab.com/), according to a pre-defined expert taxonomy, or a user-created one. 

We currently use the taxonomy provided by the Design Council, using the following high-level classifications were
used: Textiles; Souvenirs; Tableware; Interior design; Jewellery; Lighting; Ceramics; Other; Graphics; Furniture;
Wallpaper; Building accessories; Telecommunications; Tools; Clocks; Design theory; Smoking Accessories;
Portraits; Travel goods; Ornaments; Clothing; Carpets; Toys; Industrial and manufacturing processes; Domestic
appliances; Musical instruments; Heating; Scientific equipment; Engineering; Transport; Religious objects; Office
equipment; Architecture; Optical; Photography; Drawing and painting equipment and accessories; Sanitary
equipment; and Materials.



## Fine tunning

If wanting to fine-tune with a different taxonomy is necessary to create training data. 


There are two folders to consider:

> *trainingdata* -> has the data to train
> *datatoclassify* -> has the data to classify


The *trainingdata* folder has a folder called images and another called *trainingdata.csv*.  

Place the training data in the *trainingdata/images* folder, and divide the images into folders with an appropiate name. The images should be in JPG format.

List all files with the class name in a csv file in the *trainingdata* folder.

The *trainingdata.csv* contains the image and its type as follows:

>class,uid,path
>ceramics,130:30 H-O.jpg,trainingdata/images/ceramics/130:30 H-O.jpg
>ceramics,130:130 H-O.jpg,trainingdata/images/ceramics/130:130 H-O.jpg

The folder datatoclassify has a list of images in jpg

Copy the appropriate folders, images to these folders and then execute the script.

### Fine-tuning the model
If using a cluster, call the 
>$sbatch testDeployAll.slurm


### Classifying


