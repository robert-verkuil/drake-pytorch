import sys

import numpy as np
import subprocess
import torch

from nn_system.networks import *
#from igor import igor_supervised_learning_cuda


def igor_supervised_learning_cuda(trajectories, net, use_prox=True, iter_repeat=1, EPOCHS=1, lr=1e-2):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    alpha, lam, eta = 10., 10.**2, 10.**-2
    #alpha, lam, eta = 1e-3, 1e2, 1e6
    frozen_parameters = [param.clone().detach() for param in net.parameters()]
    #print("frozen_parameters: ", frozen_parameters)

    criterion1 = nn.MSELoss()
    criterion2 = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    # My data, and drop the last point
    all_inputs = np.vstack([np.expand_dims(traj[1][:-1], axis=0) for traj in trajectories])
    all_labels = np.vstack([np.expand_dims(traj[2][:-1], axis=0) for traj in trajectories])
    all_inputs = all_inputs.reshape(-1, all_inputs.shape[-1])
    all_labels = all_labels.reshape(-1, all_labels.shape[-1])
    print(all_inputs.shape)
    print(all_labels.shape)

    inputs = torch.tensor(all_inputs)
    labels = torch.tensor(all_labels)
    inputs, labels = inputs.to(device), labels.to(device)
    #def my_gen():
    #    for _ in range(iter_repeat):
    #        yield all_inputs, all_labels
    if use_prox:
        print("using prox cost!")

    for epoch in range(EPOCHS):
        running_loss = 0.0
        # for i, data in enumerate(my_gen(), 0):
        for i, _ in enumerate(range(iter_repeat)):
            # Unpack data
            #inputs, labels = data
            # inputs = torch.tensor(inputs)
            # labels = torch.tensor(labels)

            # Forward pass = THE SAUCE!
            outputs = net(inputs)
            loss = alpha/2*criterion1(outputs, labels)
            #loss = 0
            if use_prox:
                for param, ref in zip(net.parameters(), frozen_parameters):
                    loss += eta/2*criterion2(param, ref)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if (i+1) % iter_repeat == 0:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                sys.stdout.flush()  
            running_loss = 0.0
    print('Finished Training')
    #print("frozen_parameters: ", frozen_parameters)
    #print("net.parameters(): ", net.parameters())



if __name__ == "__main__":
    print("args: ", sys.argv)

    # First pickle/npsave the data to std location, overwrite possible old files.
    import os
    #os.remove(dir_name+'/new_GPU_model.pt')
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_name = dir_path+'/remote_GPU'
    all_times = np.load(dir_name+'/GPU_all_times.npy')
    all_x_knots = np.load(dir_name+'/GPU_all_x_knots.npy')
    all_u_knots = np.load(dir_name+'/GPU_all_u_knots.npy')
    trajectories = [(all_times[i], all_x_knots[i], all_u_knots[i]) for i in range(all_times.shape[0])]
    print(all_times.shape, all_x_knots.shape, all_u_knots.shape)
    print(trajectories[0][0].shape, trajectories[0][1].shape, trajectories[0][2].shape)

    # Then load the torch model
    def kNetConstructor():
#        return MLP(2, 32, layer_norm=True, dropout=True)
        #return MLP(2, 32, layer_norm=False, dropout=False)
#     return MLP(2, 2, layer_norm=False)
#     return FCBIG(2, 2)
#     return FCBIG(2, 2)

        # For cartpole
        return MLP(4, 128, layer_norm=False)
    net = kNetConstructor()
    #import pdb; pdb.set_trace()
    net.load_state_dict(torch.load(dir_name+'/GPU_model.pt'))
    net.train()

    # Run the training code with arguments
    use_prox    = bool(sys.argv[1])
    iter_repeat = int(sys.argv[2])
    EPOCHS      = int(sys.argv[3])
    lr          = float(sys.argv[4])
    print(use_prox, iter_repeat, EPOCHS, lr)
    igor_supervised_learning_cuda(trajectories, net, use_prox=use_prox, iter_repeat=iter_repeat, EPOCHS=EPOCHS, lr=lr)

    # (remotely, progress is printed and new weights are saved to a file)

    # Then save the torch model
    net.cpu()
    print("remote net params hash: ", hash(np.hstack([param.data.flatten() for param in net.parameters()]).tostring()) )
    torch.save(net.state_dict(), dir_name+'/new_GPU_model.pt')

