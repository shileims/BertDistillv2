# System Dependencies
import argparse
import os
import signal
import sys
import webbrowser

# External Dependencies
from azureml.core.datastore import Datastore
from azureml.core import Experiment
from azureml.core import Workspace
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.runconfig import RunConfiguration
from azureml.train.estimator import Estimator
from azureml.train.dnn import TensorFlow


sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from file_paths import get_data_dir_name, get_data_root_path, get_root_path

COMPUTE_TARGET_TO_VM_SIZE = { "OXO-ML-CA-Comput" : "STANDARD_NC6S_V2", "OXO-ML-CA-NC12": "STANDARD_NC12S_V2" }
DEFAULT_COMPUTE_TARGET = list(COMPUTE_TARGET_TO_VM_SIZE.keys())[0]

def convert_string_to_bool(str_to_convert):
    return str_to_convert.lower() in ('yes', 'true', 't', 'y', '1')

def forward_slashify(to_fwd_slash):
    if to_fwd_slash is None:
        return None
    return to_fwd_slash.replace('\\\\', '\\').replace('\\', '/')

script_path_arg_name = '--script-path'
script_exp_arg_name  = '--experiment-name'
parser = argparse.ArgumentParser(description="Use this script to submit a model training or evaluation session to Azure. See README.md for more detailed information.")
parser.add_argument(script_path_arg_name,       type=str,                    dest='script_path',                                        help='Relative path from the \"pipeline\" directory to the .py script to run. Something like ./azureml_tutorial/solutions/01_hello_world.py')
parser.add_argument(script_exp_arg_name,        type=str,                    dest='experiment_name',                                    help='Name of the Azure experiment. This can be anything you want.')
parser.add_argument('--config-path',            type=str,                    dest='config_path',            default="",                 help='The path to the config.json to be passed as the single input argument to the specified script. Something like ./image/classification/screenshot/configs/baseline.json')
parser.add_argument('--required-datasets-path', type=str,                    dest='required_datasets_path', default="",                 help='The relative path from the \"data_blobs\" directory to the directory containing all required datasets. Something like ./working/mnist')
parser.add_argument('--compute-target',         type=str,                    dest='compute_target',         default='OXO-ML-CA-Comput', help='The name of the compute target. Current options are {0}. Defaults to OXO-ML-CA-Comput'.format(list(COMPUTE_TARGET_TO_VM_SIZE.keys()), DEFAULT_COMPUTE_TARGET))
parser.add_argument('--should-upload-datasets', type=convert_string_to_bool, dest='should_upload_datasets', default=True,               help='Whether to upload the datasets from the required-datasets parameter. This only needs to be run for the first time a training script is executed. Datasets with the same file names as existing uploaded content will not be uploaded. Defaults to true')
parser.add_argument('--should-open-url',        type=convert_string_to_bool, dest='should_open_url',        default=True,               help='Whether to open the AzureML Experiment URL in the default web browser. Defaults to true')
parser.add_argument('--datastore',              type=str,                    dest='datastore',              default='workspaceblobstore', help='Name of the blob storage. You can work on your own azure storage account by using `register_azure_storage.py` to register it. Defaults to workspaceblobstore')
parser.add_argument('--basedir',                type=str,                    dest='basedir',                default='pretrain/pretrain_dnn', help='base directory on blob')
parser.add_argument('--extra-params',           type=str,                    dest='extra_params',                                       help='extra parameters of your script')


args = parser.parse_args()

script_path = forward_slashify(args.script_path)
experiment_name = args.experiment_name
should_open_url = args.should_open_url

def exit_missing_argument(arg):
    print("Missing required {} argument".format(arg))
    parser.print_help()
    exit(1)

if script_path is None:
    exit_missing_argument(script_path_arg_name)
if experiment_name is None:
    experiment_name = os.path.basename(script_path)


compute_target           = args.compute_target
vm_size                  = COMPUTE_TARGET_TO_VM_SIZE[compute_target]
should_upload_datasets   = args.should_upload_datasets
required_datasets_path   = args.required_datasets_path
config_path              = forward_slashify(args.config_path)
script_requires_datasets = not not required_datasets_path
script_requires_config   = not not config_path

ws_environment_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
ws = Workspace.from_config(path=ws_environment_path)
experiment_to_run = Experiment(workspace=ws, name=experiment_name)

conda_dependencies_file_path = os.path.join(get_root_path(), "pptdaml.yaml")
dependencies = CondaDependencies(conda_dependencies_file_path=conda_dependencies_file_path)

conda_packages = dependencies.conda_packages
# Remove tensorflow from pip packages as this conflicts with the Azure installation
pip_packages = [package for package in dependencies.pip_packages if package.lower().find("tensorflow")]

ds = Datastore(ws, args.datastore)

script_params = {'--basedir': ds.path(args.basedir).as_mount()}
if args.extra_params:
    param_list = args.extra_params.strip().split()
    arg_name = None
    arg_val = []
    for i, para in enumerate(param_list):
        if not para.startswith('-'):
            arg_val.append(para)
        if para.startswith('-') or i == len(param_list)-1:
            if arg_name:
                script_params[arg_name] = ' '.join(arg_val)
            arg_name = para
            arg_val = []

print(script_params)

est = TensorFlow(source_directory           =get_root_path(),
                 compute_target             =compute_target,
                 vm_size                    =COMPUTE_TARGET_TO_VM_SIZE[compute_target],
                 vm_priority                =None,
                 entry_script               =script_path,
                 script_params              =script_params,
                 node_count                 =1,
                 process_count_per_node     =1,
                 distributed_backend        =None,
                 use_gpu                    =True,
                 use_docker                 =True,
                 custom_docker_image        =None,
                 image_registry_details     =None,
                 user_managed               =False,
                 conda_packages             =conda_packages,
                 pip_packages               =pip_packages,
                 pip_requirements_file_path =None,
                 environment_definition     =None,
                 # inputs                     =datasets,
                 source_directory_data_store=None,
                 framework_version          ="1.12")

run = experiment_to_run.submit(config=est)
azure_portal_url = run.get_portal_url()

def signal_handler(sig, frame):
    print("\n*****\nAborting Azure job...\n******")
    run.fail()
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

print("Submitted job to Azure, press CTRL+C at any time to cancel the run,\n Note that cancelling the run will prevent any charts or models from being saved")
print("View this experiment at {}".format(azure_portal_url))

if(should_open_url):
    webbrowser.open_new_tab(azure_portal_url)
run.wait_for_completion(show_output=True)
