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
from azureml.train.dnn import TensorFlow, PyTorch
from azureml.core.container_registry import ContainerRegistry

def make_container_registry(address, username, password):
    cr = ContainerRegistry()
    cr.address = address
    cr.username = username
    cr.password = password
    return cr

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
print(f'Working file path: {os.path.abspath(__file__)}')
from file_paths import get_data_dir_name, get_data_root_path, get_root_path

COMPUTE_TARGET_TO_VM_SIZE = { "NC24s-v3" : "STANDARD_NC24S_V3", "OXO-ML-CA-NC12": "STANDARD_NC12S_V2", "compute":"STANDARD_NC12S_V2","compute1":"STANDARD_NC6S_V2",
                              "OXO-ML-CA-Comput": "STANDARD_NC6S_V2",
                              "NC24S-V3s": "STANDARD_NC24S_V3",
                              "australia1GPUcl": "STANDARD_NC6S_V3",
                              "NC24s-v3-VP": "STANDARD_NC24S_V3",
                              }
DEFAULT_COMPUTE_TARGET = list(COMPUTE_TARGET_TO_VM_SIZE.keys())[0]


configs_defaultsinglegpu = {}
configs_defaultsinglegpu['config.json'] = ['OXO-ML-CA-Comput', 'lei2021datastore']

configs_defaultcpus = {}
configs_defaultcpus['config.json'] = ['lei-compute-cpu', 'lei2021datastore']


def convert_string_to_bool(str_to_convert):
    return str_to_convert.lower() in ('yes', 'true', 't', 'y', '1')

def forward_slashify(to_fwd_slash):
    if to_fwd_slash is None:
        return None
    return to_fwd_slash.replace('\\\\', '\\').replace('\\', '/')

def exit_missing_argument(arg):
    print("Missing required {} argument".format(arg))
    parser.print_help()
    exit(1)

def output_datastores(ws):
    print('=== current datastores ===')
    for dname in ws.datastores:
        print(dname)


def signal_handler(sig, frame):
    print("\n*****\nAborting Azure job...\n******")
    run.fail()
    sys.exit(0)

script_path_arg_name = '--script-path'
script_exp_arg_name  = '--experiment-name'
parser = argparse.ArgumentParser(description="Use this script to submit a model training or evaluation session to Azure. See README.md for more detailed information.")
parser.add_argument('--config-path',            type=str,                    dest='config_path',            default="",                 help='The path to the config.json to be passed as the single input argument to the specified script. Something like ./image/classification/screenshot/configs/baseline.json')
parser.add_argument('--required-datasets-path', type=str,                    dest='required_datasets_path', default="",                 help='The relative path from the \"data_blobs\" directory to the directory containing all required datasets. Something like ./working/mnist')
parser.add_argument('--compute-target',         type=str,                    dest='compute_target',         default='lei-mmtraining', help='The name of the compute target. Current options are {0}. Defaults to OXO-ML-CA-Comput'.format(list(COMPUTE_TARGET_TO_VM_SIZE.keys()), DEFAULT_COMPUTE_TARGET))
# parser.add_argument('--compute-target',         type=str,                    dest='compute_target',         default='lei4g3', help='The name of the compute target. Current options are {0}. Defaults to OXO-ML-CA-Comput'.format(list(COMPUTE_TARGET_TO_VM_SIZE.keys()), DEFAULT_COMPUTE_TARGET))
# parser.add_argument('--compute-target',         type=str,                    dest='compute_target',         default='lei-compute', help='The name of the compute target. Current options are {0}. Defaults to OXO-ML-CA-Comput'.format(list(COMPUTE_TARGET_TO_VM_SIZE.keys()), DEFAULT_COMPUTE_TARGET))
# parser.add_argument('--compute-target',         type=str,                    dest='compute_target',         default='OXO-ML-CA-Comput', help='The name of the compute target. Current options are {0}. Defaults to OXO-ML-CA-Comput'.format(list(COMPUTE_TARGET_TO_VM_SIZE.keys()), DEFAULT_COMPUTE_TARGET))

parser.add_argument('--should-upload-datasets', type=convert_string_to_bool, dest='should_upload_datasets', default=True,               help='Whether to upload the datasets from the required-datasets parameter. This only needs to be run for the first time a training script is executed. Datasets with the same file names as existing uploaded content will not be uploaded. Defaults to true')
parser.add_argument('--should-open-url',        type=convert_string_to_bool, dest='should_open_url',        default=True,               help='Whether to open the AzureML Experiment URL in the default web browser. Defaults to true')

# parser.add_argument('--workspace-config',       type=str,                    dest='workspace_config',       default='config.json',      help='config.json | config_hi.json for high-perf workspace')

# parser.add_argument('--datastore',              type=str,                    dest='datastore',              default='lei2021datastore', help='Name of the blob storage. You can work on your own azure storage account by using `register_azure_storage.py` to register it. Defaults to workspaceblobstore')
# parser.add_argument('--datastore',              type=str,                    dest='datastore',              default='leiu2netdatastore', help='Name of the blob storage. You can work on your own azure storage account by using `register_azure_storage.py` to register it. Defaults to workspaceblobstore')
# parser.add_argument('--datastore',              type=str,                    dest='datastore',              default='lei4g1', help='Name of the blob storage. You can work on your own azure storage account by using `register_azure_storage.py` to register it. Defaults to workspaceblobstore')
parser.add_argument('--basedir',                type=str,                    dest='basedir',                default='.', help='base directory on blob')
parser.add_argument('--extra-params',           type=str,                    dest='extra_params',                                       help='extra parameters of your script')

parser.add_argument('--config_type', type=str, default='configs_defaultcpus', choices=['configs_defaultsinglegpu', 'configs_defaultcpus'])
parser.add_argument('--config_index', type=int, default=-1, help='Specify which compute-target is used while more than one compute-target exists!')
parser.add_argument('--config_file', type=str, default='config.json', help='Specify the workspace config file')
parser.add_argument('--experiment-tag-name',    type=str, dest='experiment_tag_name', default='check_data1', help='add experiment tag name')
parser.add_argument(script_path_arg_name,       type=str,                    dest='script_path',            default='azure_experiments/check_data.py',                           help='Relative path from the \"pipeline\" directory to the .py script to run. Something like ./azureml_tutorial/solutions/01_hello_world.py')
# parser.add_argument(script_path_arg_name,       type=str,                    dest='script_path',            default='azure_experiments/run_distrib_distill.py',                           help='Relative path from the \"pipeline\" directory to the .py script to run. Something like ./azureml_tutorial/solutions/01_hello_world.py')
parser.add_argument(script_exp_arg_name,        type=str,                    dest='experiment_name',        default='distill_refactor_dist_1.4M',                            help='Name of the Azure experiment. This can be anything you want.')

# support docker image
parser.add_argument('--custom-docker-image', type=str, default='philly/jobs/custom/pytorch:pytorch1.2.0-py36-nlp-sum-fp16')
parser.add_argument('--use-docker', action='store_false')
parser.add_argument('--show-output', action='store_true')

args = parser.parse_args()

if __name__ == '__main__':

    script_path = forward_slashify(args.script_path)
    experiment_name = '_'.join(args.experiment_name.strip().split('_')[:2])
    should_open_url = args.should_open_url

    if script_path is None:
        exit_missing_argument(script_path_arg_name)
    if experiment_name is None:
        exit_missing_argument(script_exp_arg_name)

    global_configs = globals()
    assert args.config_type in global_configs, f'config_type should be global definition'
    config_type = global_configs[args.config_type]
    assert args.config_file in config_type, f'config file does not exist in the global config setting'
    settings = config_type[args.config_file]
    if args.config_file == 'config_shilei.json':
        settings = settings[args.config_index]
    args.workspace_config = args.config_file
    args.compute_target = settings[0]
    args.datastore = settings[1]

    compute_target           = args.compute_target
    vm_size                  = COMPUTE_TARGET_TO_VM_SIZE[compute_target]
    should_upload_datasets   = args.should_upload_datasets
    required_datasets_path   = args.required_datasets_path
    config_path              = forward_slashify(args.config_path)
    script_requires_datasets = not not required_datasets_path
    script_requires_config   = not not config_path

    ws_environment_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.workspace_config)
    print(f'workspace environment path: {ws_environment_path}')
    ws = Workspace.from_config(path=ws_environment_path)
    experiment_to_run = Experiment(workspace=ws, name=experiment_name)

    conda_packages = None
    pip_packages = None

    # conda_dependencies_file_path = os.path.join(get_root_path(), "openmmlab.yaml")
    # dependencies = CondaDependencies(conda_dependencies_file_path=conda_dependencies_file_path)

    # output_datastores(ws)
    ds = Datastore(ws, args.datastore)


    data_dir_name = get_data_dir_name()
    expected_path_on_compute = forward_slashify(os.path.join(data_dir_name, required_datasets_path))

    if should_upload_datasets and script_requires_datasets:
        src_path = get_data_root_path(required_datasets_path)
        ds.upload(src_path, target_path=expected_path_on_compute, overwrite=True, show_progress=True)

    # conda_packages = dependencies.conda_packages
    # # Remove tensorflow from pip packages as this conflicts with the Azure installation
    # pip_packages = [package for package in dependencies.pip_packages if package.lower().find("tensorflow")]

    datasets = []
    if script_requires_datasets:
        datasets.append(ds.path(expected_path_on_compute).as_download(path_on_compute="./"))

    if args.datastore == 'workspaceblobstore':
        script_params = {config_path: ''} if script_requires_config else []
    else:
        # Set node number for distribution training
        if 'distrib' in args.script_path:
            if args.config_file == 'config_dummtraining.json':
                node_num = 8
            else:
                node_num = 4

            script_params = {'--basedir': ds.path(args.basedir).as_mount(),
                             '--tag': args.experiment_tag_name,
                             '--experiment_name': args.experiment_name,
                             '--node_num': node_num}
        else:
            script_params = {'--basedir': ds.path(args.basedir).as_mount(),
                             '--tag': args.experiment_tag_name,
                             '--experiment_name': args.experiment_name}

    if args.extra_params:
        if isinstance(script_params, list):
            script_params = {}
        if args.extra_params.startswith('"') and args.extra_params.endswith('"'):
            args.extra_params = args.extra_params[1:-1]
        print('params {}'.format(args.extra_params))
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

    if args.use_docker:
        est = PyTorch(source_directory=get_root_path(),
                         compute_target=compute_target,
                         vm_size=COMPUTE_TARGET_TO_VM_SIZE[compute_target],
                         entry_script=script_path,
                         script_params=script_params,
                         node_count=1,
                         process_count_per_node=1,
                         use_gpu=True,
                         use_docker=True,
                         image_registry_details=make_container_registry(
                             address=None,
                             username='',
                             password=''),
                         custom_docker_image='',
                         user_managed=True,
                         inputs=datasets)
    else:
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
                        image_registry_details     =None,
                        custom_docker_image        =None,
                        user_managed               =False,
                        conda_packages             =conda_packages,
                        pip_packages               =pip_packages,
                        pip_requirements_file_path =None,
                        environment_definition     =None,
                        inputs                     =datasets,
                        source_directory_data_store=None,
                        framework_version          ="2.0")

    experiment_tags = {'tag': args.experiment_tag_name} if args.experiment_tag_name else None
    run = experiment_to_run.submit(config=est, tags=experiment_tags)
    azure_portal_url = run.get_portal_url()

    # signal.signal(signal.SIGINT, signal_handler)

    print("Submitted job to Azure, press CTRL+C at any time to cancel the run,\n Note that cancelling the run will prevent any charts or models from being saved")
    print("View this experiment at {}".format(azure_portal_url))

    run.wait_for_completion(show_output=args.show_output)
