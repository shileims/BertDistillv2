# first version of hyperparam sweep script
# only support lr sweep

import argparse, os, sys

def run_cmd(cmd):
    os.system(cmd)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--script-path', required=True)
    parser.add_argument('--experiment-name', required=True)
    parser.add_argument('--should-upload-datasets', choices=['True', 'False'])
    parser.add_argument('--required-datasets-path', required=True)
    parser.add_argument('--compute-target', required=True)
    parser.add_argument('--workspace-config',       type=str,                    dest='workspace_config',       default='config_hi.json',      help='config.json | config_hi.json for high-perf workspace')
    parser.add_argument('--datastore', default='workspaceblobstore')
    parser.add_argument('--experiment-tag-name', required=True)
    parser.add_argument('--lr', required=True, help='seperated by ,')
    parser.add_argument('--use-docker', action='store_true')
    parser.add_argument('--show-output', action='store_true')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()
    lrs = args.lr.strip().split(',')
    dirname = os.path.dirname(os.path.abspath(__file__))
    exe_file = os.path.join(dirname, 'cloud_executor.py')
    print(exe_file)

    import multiprocessing
    pool = multiprocessing.Pool(len(lrs))
    for lr in lrs:
        cmd = '''
            {python} -u {exe_file} \
                    --script-path {script_path} \
                    --experiment-name {experiment_name} \
                    --should-upload-datasets {should_upload_datasets} \
                    --required-datasets-path {required_datasets_path} \
                    --compute-target {compute_target} \
                    --workspace-config {workspace_config} \
                    --experiment-tag-name {experiment_tag_name} \
                    --datastore {datastore} \
                    --extra-params "--lr {lr}"
            '''.format(python=sys.executable, exe_file=exe_file, script_path=args.script_path,
                        experiment_name=args.experiment_name, should_upload_datasets=args.should_upload_datasets,
                        required_datasets_path=args.required_datasets_path, compute_target=args.compute_target,
                        workspace_config=args.workspace_config,
                        datastore=args.datastore,
                        experiment_tag_name=args.experiment_tag_name + '_lr{}'.format(lr),
                        lr=lr,
                    )
        if args.use_docker:
            cmd = cmd + ' --use-docker'
        if args.show_output:
            cmd = cmd + ' --show-output'
        cmd = ' '.join( cmd.split() )
        print(cmd)
        pool.apply_async(run_cmd, args=(cmd, ))

    pool.close()
    pool.join()
