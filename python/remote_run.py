import numpy as np
import subprocess
import torch

from nn_system.networks import *

if __name__ == "__main__":
    data   = np.random.randn(96, 2)
    labels = np.random.randn(96, 1)
    net = MLP(2, 32, layer_norm=False, dropout=True, input_noise=10, output_noise=10)

    # First pickle/npsave the data to std location, overwrite possible old files.
    dir_name = 'remote_GPU'
    np.save(dir_name+'/GPU_data', data)
    np.save(dir_name+'/GPU_labels', labels)

    # Then save the torch model
    torch.save(net.state_dict(), dir_name+'/GPU_model.pt')

    # Then scp those files over
    cmd = "rsync -zaP remote_GPU RLG:/home/rverkuil/integration/drake-pytorch/python"
    subprocess.call(cmd.split(' '))

    # Remotely run the training code with arguments
    # (remotely, progress is printed and new weights are saved to a file)
    python_path = "/home/rverkuil/integration/integration/bin/python"
    script_path = "/home/rverkuil/integration/drake-pytorch/python/remote_train.py 1 1000 5 0.03"
    sub_cmd = python_path+" "+script_path
    #cmd = "ssh RLG \""+sub_cmd+"\""
    subprocess.call(['ssh','RLG',sub_cmd])

    # SCP the new weights back
    cmd = "rsync -zaP RLG:/home/rverkuil/integration/drake-pytorch/python/remote_GPU ."
    subprocess.call(cmd.split(' '))

    # Load the new weights back
    net = torch.load(dir_name+'/GPU_model.pt')

    # Do optional cleanup or files that were used!
    
