
import os
import sys, argparse
import azureml.core
from azureml.core import Workspace
from azureml.core import Datastore
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

parser = argparse.ArgumentParser()
parser.add_argument('--datastore_name')
parser.add_argument('--blob_container_name')
parser.add_argument('--blob_account_name')
parser.add_argument('--blob_account_key')
parser.add_argument('--unregister', action='store_true')
parser.add_argument('--config', default='config_hi.json', help='config.json | config_hi.json')

args = parser.parse_args()

#
# Configurations
#

datastore_name = args.datastore_name
blob_container_name = args.blob_container_name
blob_account_name = args.blob_account_name
blob_account_key = args.blob_account_key
unregister = args.unregister

#
# Prepare the workspace.
#

ws_environment_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.config)
ws = Workspace.from_config(path=ws_environment_path)
print(ws.get_details())

print('=== current datastores ===')
for dname in ws.datastores:
    print(dname)

#
# Register an existing datastore to the workspace.
#

if datastore_name not in ws.datastores:
    Datastore.register_azure_blob_container(
        workspace=ws,
        datastore_name=datastore_name,
        container_name=blob_container_name,
        account_name=blob_account_name,
        account_key=blob_account_key
    )
    print("Datastore '%s' registered." % datastore_name)
else:
    print("Datastore '%s' has already been regsitered." % datastore_name)
    if unregister:
        datastore = Datastore.get(workspace=ws, datastore_name=datastore_name)
        print(datastore)
        datastore.unregister()
        print('Datastore {} unregsitered!'.format(datastore_name))


print('=== current datastores ===')
for dname in ws.datastores:
    print(dname)

# (END)
