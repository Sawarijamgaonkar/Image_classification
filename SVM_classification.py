"""use the below commented code if you want to insert any other image set from kaggle and also upload the .jason format file 
from the files to google colab while running the code. By this code you will be able to use any dataset from kaggle and then 
without actually downloading it in your local storage you can easily apply the SVM """

# !mkdir -p ~/.kaggle
# !cp kaggle.json ~/.kaggle/

# !mkdir -p ~/.kaggle
# !kaggle datasets download -d chetankv/dogs-cats-images#
# !cp kaggle.json ~/.kaggle/

# import zipfile
# zip_ref= zipfile.ZipFile('/content/dogs-cats-images.zip' , 'r')
# zip_ref.extractall('/content/')
# zip_ref.close()


import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, models, transforms
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import os
import time
import pickle

## Define file directories
file_dir = "./data-full"
output_dir = "./output/SVM_trained.pth"
out_report_dir = './output/classification_report.txt'
TRAIN = "train"
TEST = "test"


def get_data(file_dir):
    print("[INFO] Loading data...")
    # Initialize data transformations
    data_transform = {
        TRAIN: transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        ),
        TEST: transforms.Compose(
            [
                transforms.Resize(254), 
                transforms.CenterCrop(224), 
                transforms.ToTensor()
            ]
        ),
    }
    # Initialize datasets and apply transformations
    datasets_img = {
        file: datasets.ImageFolder(
            os.path.join(file_dir, file), transform=data_transform[file]
        )
        for file in [TRAIN, TEST]
    }
    # Load data into dataloaders
    dataloaders = {
        file: torch.utils.data.DataLoader(
            datasets_img[file], batch_size=8, shuffle=True, num_workers=4
        )
        for file in [TRAIN, TEST]
    }
    # Get class names and dataset sizes
    class_names = datasets_img[TRAIN].classes
    datasets_size = {file: len(datasets_img[file]) for file in [TRAIN, TEST]}
    for file in [TRAIN, TEST]:
        print(f"[INFO] Loaded {datasets_size[file]} images under {file}")
    print(f"Classes: {class_names}")

    return datasets_img, datasets_size, dataloaders, class_names


def get_vgg16_modified_model(weights=models.VGG16_BN_Weights.DEFAULT):

    print("[INFO] Getting VGG-16 pre-trained model...")
    # Load VGG-16 pretrained model
    vgg16 = models.vgg16_bn(weights)
    # Freeze training for all layers
    for param in vgg16.features.parameters():
        param.requires_grad = False
    # Remove the classifier layers
    features = list(vgg16.classifier.children())[:-7]
    # Replace the model's classifier
    vgg16.classifier = nn.Sequential(*features)
    # print(vgg16)
    return vgg16


def get_classification_report(truth_values, pred_values):

    report = classification_report(truth_values, pred_values, target_names=class_names,  digits=4)
    conf_matrix = confusion_matrix(truth_values, pred_values, normalize='all') 
    print('[Evalutaion Model] Showing detailed report\n')
    print(report)
    print('[Evalutaion Model] Showing confusion matrix')
    print(f'                       Predicted Label              ')
    print(f'                         0            1         ')
    print(f' Truth Label     0   {conf_matrix[0][0]:4f}     {conf_matrix[0][1]:4f}')
    print(f'                 1   {conf_matrix[1][0]:4f}     {conf_matrix[1][1]:4f}')
    
    
def save_classification_report(truth_values, pred_values, out_report_dir):
    print('[INFO] Saving report...')
    c_report = classification_report(truth_values, pred_values, target_names=class_names,  digits=4)
    conf_matrix = confusion_matrix(truth_values, pred_values, normalize='all') 
    matrix_report = ['                       Predicted Label              ', 
                     f'                         0            1         ',
                     f' Truth Label     0   {conf_matrix[0][0]:4f}     {conf_matrix[0][1]:4f}',
                     f'                 1   {conf_matrix[1][0]:4f}     {conf_matrix[1][1]:4f}']
    
    with open(out_report_dir, 'w') as f:
        f.write(c_report)
        f.write('\n')
        for line in matrix_report:
            f.write(line)
            f.write('\n')
            

def get_features(vgg, file=TRAIN):
    print(f"[INFO] Getting '{file}' features...")
    svm_features = []
    svm_labels = []
    data_batches_len = len(dataloaders[file])
    for i, data_batch in enumerate(dataloaders[file]):
        print(f"\r[FEATURE] Loading batch {i + 1}/{data_batches_len} ({len(data_batch[1])*(i+1)} images)", end='', flush=True)
        # In this case, loaded databatch of 8 images including 8 features and 8 labels
        inputs, labels = data_batch
        if use_gpu:
            # Get the data through the feature extractor of VGG16
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
            # Extract data from VGG16 feature extractor as a vector
            features = vgg(inputs)
            # print(features.shape)     # torch.Size([8, 25088])
            # print(labels.shape)       # torch.Size([8])
            features = features.cpu().detach().numpy()
            labels = labels.cpu().detach().numpy()
        else:
            # Get the data through the feature extractor of VGG16
            inputs = Variable(inputs)
            labels = Variable(labels)
            # Extract data from VGG16 feature extractor as a vector
            features = vgg(inputs)
            features = features.detach().numpy()
            labels = labels.detach().numpy()
        
        # Add feature with correct label into an array
        # print(features.shape)     # (8, 25088)
        # print(labels.shape)       # (8,)
        for index in range(len(labels)):
            feature = features[index]  
            label = labels[index]
            # Add it to the features list
            # print(feature.shape)     # (25088,)
            svm_features.append(feature)
            # print(label.shape)       # (1)
            svm_labels.append(label)
            
    print("\n[FEATURE] Features loaded")
    return svm_features, svm_labels


def svm_classifier(train_data, test_data):
    since = time.time()
    FEATURE_INDEX = 0
    LABEL_INDEX = 1
    print('[INFO] Getting model...')
    # There are 1000 images in the train data
    train_features = np.array(train_data[FEATURE_INDEX])
    # print(features.shape)     # (1000, 25088)
    train_labels = np.array(train_data[LABEL_INDEX])
    # print(labels.shape)       # (1000,)
    
    # There are 600 images in the test data
    test_features = np.array(test_data[FEATURE_INDEX])
    # print(features.shape)     # (1000, 25088)
    test_labels = np.array(test_data[LABEL_INDEX])
    # print(labels.shape)       # (1000,)
    
    # Create model
    svm_model = SVC(gamma="auto")
    # Train model
    print('[INFO] Fitting...')
    svm_model.fit(train_features, train_labels)
    print('[INFO] Model completed')
    # Get result
    print('[INFO] Testing...')
    pred_labels = svm_model.predict(test_features)
    print('[INFO] Printing classification report')
    get_classification_report(test_labels, pred_labels)
    elapsed_time = time.time() - since
    print(f"[INFO] Model produced in {(elapsed_time // 60):.0f}m {(elapsed_time % 60):.0f}s")
    save_classification_report(test_labels, pred_labels, out_report_dir)
    return svm_model


if __name__ == "__main__":
    # Use GPU if available. Note that this only to load features using VGG16. Scikit Learn SVM does not support GPU
    use_gpu = torch.cuda.is_available()
    print("[INFO] Using CUDA") if use_gpu else print("[INFO] Using CPU")
    # Get Data
    datasets_img, datasets_size, dataloaders, class_names = get_data(file_dir)
    # Get VGG16 pre-trained model
    vgg16 = get_vgg16_modified_model()
    # Move model to GPU
    if use_gpu:
        torch.cuda.empty_cache()
        vgg16.cuda()
    # Extract features and labels from the VGG16 model
    svm_train_features, svm_train_labels = get_features(vgg16, TRAIN)
    svm_test_features, svm_test_labels = get_features(vgg16, TEST)
    # Run SVM 
    svm_model = svm_classifier(
        [svm_train_features, svm_train_labels],
        [svm_test_features, svm_test_labels],
    )
    # Save model
    print('[INFO] Saving model...')
    pickle.dump(svm_model, open(output_dir, 'wb'))
    print('[INFO] Done')
