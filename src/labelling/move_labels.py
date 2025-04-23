'''
This is a script that runs locally.
'''

import os
import json
import subprocess


def move_json_files(local_folder, remote_folder, start_folder, end_folder):
    # Create the remote folder if it doesn't exist
    ssh_command = f'ssh mpi3 mkdir -p {remote_folder}'
    subprocess.run(ssh_command, shell=True, check=True)

    # Iterate over folders '00' to '09'
    for i in range(start_folder, end_folder+1):
        folder_name = f'{i:02}'  # Zero-padded folder name
        json_file_path = os.path.join(local_folder, folder_name, 'labels.json')
        remote_json_file_path = f'mpi3:{remote_folder}/labels_{folder_name}.json'

        # Check if the json file exists
        if os.path.exists(json_file_path):
            # Save json data to the remote folder with the new name using SCP
            scp_command = f'scp {json_file_path} {remote_json_file_path}'
            subprocess.run(scp_command, shell=True, check=True)

            print(f"Moved {json_file_path} to {remote_json_file_path}")
        else:
            print(f"No JSON file found in folder {folder_name}")

# Example usage
local_folder_path = '/Users/prasannamayil/Documents/datasets/100K_3/'
remote_folder_path = '/is/cluster/fast/pmayilvahanan/clip_ood_part2/labels/labelling_round1/'
start_folder = 0
end_folder = 9
move_json_files(local_folder_path, remote_folder_path, start_folder, end_folder)
