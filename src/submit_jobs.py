"""Test for spearmint."""
from __future__ import print_function, division

import time
import subprocess
import os
import argparse
import yaml
from itertools import product
import datetime


def generate_combinations(dictionary):
    # Get all the values from the dictionary
    values_lists = list(dictionary.values())

    # Generate combinations using itertools.product
    for combination in product(*values_lists):
        yield dict(zip(dictionary.keys(), combination))


def create_sub_file(args):
    """
    Create a condor job submission file.

    Create a submission file, submit it to condor,
    return only when the job has been properly submitted.
    """
    # write the submission file in the local mount of the cluster

    with open(args['sub_name'], 'w') as fid:
        fid.write(f'executable = {args["sh_name"]}\n')
        fid.write(f'request_cpus = {args["cpus"]}\n')
        fid.write(f'request_memory = {args["memory"]}\n')
        fid.write(f'request_disk = {args["disk"]}\n')

        # Gpu related lines
        if args['job_type'] in ['train', 'eval']:
            fid.write(f'request_gpus = {args["gpus"]}\n')
            fid.write(f'requirements = {args["requirements"]}\n')

        fid.write(f'log_root = {args["log_root"]}$(ClusterId)\n')
        # arguments = SCRIPT + params
        # fid.write('arguments = ' + arguments + '\n')
        fid.write('error = $(log_root).err\n')
        fid.write('output = $(log_root).out\n')
        fid.write('log = $(log_root).log\n')
        fid.write(f'MaxTime = {args["maxtime"]}\n')
        fid.write('periodic_remove = (JobStatus =?= 2) && ((CurrentTime - JobCurrentStartDate) >= $(MaxTime))\n')
        fid.write('queue\n')


def create_sh_file(args, config):
    """
    Create a condor job submission file.

    Create a submission file, submit it to condor,
    return only when the job has been properly submitted.
    """
    script_dict = {}
    script_dict['train'] = "style_classifier.train"
    script_dict['eval'] = "style_classifier.eval"
    script_dict['vis'] = "visualization.main"

    param_str = script_dict[args['job_type']]
    for key in config.keys():
        param_str += " --" + key + " " + str(config[key])

    # write the submission file in the local mount of the cluster

    with open(args['sh_name'], 'w') as fid:
        fid.write('module load cudnn/8.4.1-cu11.6\n')
        fid.write('echo "Loaded cudnn/8.7.0-cu11.x"\n')
        fid.write('module load cudnn/8.4.1-cu11.6\n')
        fid.write('source /home/pmayilvahanan/.env/bin/activate\n')
        fid.write('echo "activated source from /home/pmayilvahanan/.env/bin/activate"\n')
        fid.write(f'cd {args["script_root"]}\n')
        fid.write(f'echo change dir to {args["script_root"]}\n')
        fid.write(f'echo param_str {param_str}\n')
        fid.write(f'python3 -m {param_str}\n')

    # change permissions of file
    new_permissions = 0o777  # For example, 0777 (octal notation)
    os.chmod(args['sh_name'], new_permissions)


def condor_submit(sub_file, bid):
    """
    Submit a job to condor.

    Create a submission file, submit it to condor,
    return only when the job has been properly submitted.
    """
    submission_command = "condor_submit_bid " + str(bid) + " " + sub_file
    output = subprocess.check_output(submission_command, shell=True, text=True)
    output_lines = output.split('\n')

    # Iterate through the lines to find the line containing the job ID
    for line in output_lines:
        if 'cluster' in line:
            # Extract the job ID from the last word in the line
            job_id = line.split()[-1]

    print("Submission successful. Command used:", submission_command)
    print(f"job_id = {job_id}")
    return job_id


def await_exec(job_id, log_root, POLLING_INTERVAL=30):
    """Wait for the condor job to finish."""
    log_file = os.path.join(log_root, job_id + 'log')
    while True:
        time.sleep(POLLING_INTERVAL)

        with open(log_file, 'r') as log_fid:
            log_content = log_fid.read()
            if ('Job terminated' in log_content) or ('Job executing' in log_content):
                print("Job executed or terminated, onto next")
                break


def get_args():
    '''
    Job no. is the most important argument through which we get everything else
    '''

    parser = argparse.ArgumentParser()

    # input args
    parser.add_argument("--job_type", type=str, default='train', choices=['train', 'eval', 'vis'])
    parser.add_argument("--config", type=str,
                        default="")
    parser.add_argument("--log_root", type=str, default='/is/cluster/fast/pmayilvahanan/clip_ood_part2/jobs/')

    # sub args
    parser.add_argument("--cpus", type=str, default='4')
    parser.add_argument("--gpus", type=str, default='1')
    parser.add_argument("--memory", type=str, default='300000')
    parser.add_argument("--disk", type=str, default='100G')
    parser.add_argument("--requirements", type=str, default='(CUDADeviceName == "NVIDIA A100-SXM4-80GB")',
                        choices=['(CUDADeviceName == "NVIDIA A100-SXM4-40GB")',
                                 '(CUDADeviceName == "NVIDIA A100-SXM4-80GB")',
                                 '(TARGET.CUDACapability >= 7.0) && (TARGET.CUDAGlobalMemoryMb >=40000)',
                                 '(TARGET.CUDACapability >= 7.0) && (TARGET.CUDAGlobalMemoryMb >=80000)',
                                 ])
    parser.add_argument("--maxtime", type=str, default='259200')
    parser.add_argument("--bid", type=str, default='1000')

    # submit_jobs arg
    parser.add_argument("--sleep_time", type=int, default=5)

    # Create args and add some extra things
    ap = parser.parse_args()
    args = vars(ap)

    # open config and paths
    args['script_root'] = os.path.dirname(os.path.abspath(__file__))
    relative_path_to_config_folder = "config"
    config_file_name = args['job_type']+".yaml"
    config_file_path = os.path.join(args['script_root'], relative_path_to_config_folder, config_file_name)
    with open(config_file_path, "r") as ymlfile:
        args['config'] = yaml.safe_load(ymlfile)

    # make 'run' dir for running scripts and change dir there
    args['run_root'] = os.path.join(args['script_root'], "run")
    os.makedirs(args['run_root'], exist_ok=True)
    os.chdir(args['run_root'])

    return args


if __name__ == '__main__':
    args = get_args()

    # create sub file
    for config_job in generate_combinations(args['config']):
        # create sh and sub file name
        args['sh_name'] = args['job_type'] + '_' + f'{datetime.datetime.now():%Y%m%d%H%M%S}' + '.sh'
        args['sub_name'] = args['job_type'] + '_' + f'{datetime.datetime.now():%Y%m%d%H%M%S}' + '.sub'
        create_sub_file(args)
        create_sh_file(args, config_job)
        print(f"starting job with {config_job}")
        job_id = condor_submit(args['sub_name'], args['bid'])
        # await_exec(job_id, args['log_root'])  # uncommented this so that all jobs can be run
        time.sleep(args['sleep_time'])