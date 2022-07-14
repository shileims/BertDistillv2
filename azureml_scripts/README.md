# Training and Evaluating Models on the Cloud With AzureML
This file describes how to run scripts on the OXO-ML-Sunnyvale AML Workspace

### Motivation
Many local machines do not have the necessary computing resources for training and evaluating models. AzureML has these resources but it can be a hassle to keep the local and cloud environments in sync. The cloud_executor script provides the necessary logic for ensuring that cloud resources are initialized with the proper libraries, manipulated datasets are kept in sync locally and on the cloud, and any models, graphs, or logs are available to be downloaded or shared.

### General Architecture
![General Architecture](img/architecture.png "Architecture")
   1. Specify the script, data files, and machine configuration that are required for setting up a new machine on Azure ML.
   2. The cloud executor performs the following in series
      1. Upload any new data files to the existing OXO-ML-Sunnyvale Workspace Blob Storage instance
      2. A new cloud compute node is created if necessary from the provided .yaml configuration file. Note that if this configuration was used once before, an existing VM is reused with a new working container.
      3. The cloud compute node downloads or mounts any required datasets from the Blob Storage instance to the VM instance
      4. The specified script runs on the cloud computing cluser until completion or when CTRL-C is pressed. Note that if an experiment is cancelled, any graphs or models produced up until cancellation are not preserved.
   3. Any models, graphs, or other files produced are saved in the Experiments tab [\(example\)](https://mlworkspace.azure.ai/portal/subscriptions/67aa06b0-2686-40dc-92b0-1316ea0304d9/resourceGroups/OXO-ML-Sunnyvale/providers/Microsoft.MachineLearningServices/workspaces/OXO-ML-Sunnyvale/experiments/ScreencopyDetection-Training/runs/ScreencopyDetection-Training_1556634872_f80153da).
   4. Files produced in the Experiments tab can be downloaded or shared as necessary for further evaluation or implementations.

### Rules
In order for your script to be runnable both locally and on AML, there are some rules that must be followed. To help follow these rules, the **file_paths.py** file contains all methods necessary for retrieving full paths of all bolded directories below regardless if running a script locally or on the cloud.
   1. Any dependencies must be listed in the **pptdaml.yaml** [conda](https://conda.io/en/latest/) configuration file. 
   2. All scripts and required supporting files must be placed under the **pipeline** directory. The cloud_executor uploads all files that are under this folder, with some exceptions.
   3. Any required datasets must be placed under the **data_blobs** directory. This folder is special because it is not uploaded to AzureML directly. Instead, folders within this directory are first uploaded to Blob Storage and then downloaded on the AML compute cluster.
   4. Datasets under the **data_blobs** directory should be placed into either the **data_blobs/raw** and **data_blobs/working** directories.
      1. **data_blobs/raw** should always contain raw data that needs to be preprocessed prior to training a model
      2. **data_blobs/working** should always contain the processed data used for training a model. If working with a lot of files, you probably should consider zipping the files into one or more larger files as AML currently doesn't support uploading or downloading files in parrallel.
   5. Any produced models, graphs, or other files must be written to the **outputs** directory.

### Example AML usage
Check out the following example scripts that preprocess data and then train a CNN for screenshot detection.

```sh
python pipeline/azureml/cloud_executor.py --script-path image/classification/screenshot/model_train.py --experiment-name ScreencopyDetection-Train-ExtData --config-path image/classification/screenshot/configs/baseline.json --required-datasets-path working/image/classification/screenshot/preprocessednpk/64x64 --should-upload-datasets False
```