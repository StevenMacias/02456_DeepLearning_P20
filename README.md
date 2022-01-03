# 02456_DeepLearning_P20
In this repository you can find the dataset used by the group and the developed code. The structure of the repository is the following one:

- *rcnn_model:* This folder contains the different Jupyter Notebooks used to extract the frames from the dataset, train the model and evaluate the results.
- *Tracking_algorithm_and_cv2_implementation:* This folder contains the object tracking code.
- *project_20_data.zip:* Dataset used in the training of the model. It consists of two videos and the `.xml` files that contain the coordinates of the bounding boxes (bboxes). 

## Dependencies
In order to run the training notebook `rcnn_model/BeerCans_FastRCNN.ipynb` you should create a conda environment with the following commands:
```
conda create --name deep_learning python=3.9
conda activate deep_learning
conda install -c conda-forge jupyterlab
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
conda install -c conda-forge opencv
```
## Home directory structure


The training notebook will create a folder named `02456_Temp`. The structure of the temporal folder in the following one:

- *02456_Temp/Dataset:* in this directory the dataset is placed, uncompressed and procesed to be used by the training notebook. 
- *02456_Temp/Models:* directory where the resulting models are stored. 
- *02456_Temp/Preds:* For each trained model, two folders are created in this directory. Each folder contains a video showing the bboxes predictions of the model over the original dataset.  




Authors: Martin Hoffmann & Steven Macías	
