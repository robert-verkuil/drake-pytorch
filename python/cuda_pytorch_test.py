import torch

def igor_supervised_learning_cuda(trajectories, net, use_prox=True, iter_repeat=1, EPOCHS=1, lr=1e-2):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    alpha, lam, eta = 10., 10.**2, 10.**-2
    frozen_parameters = [param.clone() for param in net.parameters()]

    criterion1 = nn.MSELoss()
    criterion2 = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    # My data
    all_inputs = np.vstack([np.expand_dims(traj[1], axis=0) for traj in trajectories])
    all_labels = np.vstack([np.expand_dims(traj[2], axis=0) for traj in trajectories])
    all_inputs = all_inputs.reshape(-1, all_inputs.shape[-1])
    all_labels = all_labels.reshape(-1, all_labels.shape[-1])
    print(all_inputs.shape)
    print(all_labels.shape)

    inputs = torch.tensor(inputs)
    labels = torch.tensor(labels)
    inputs, labels = inputs.to(device), labels.to(device)
    #def my_gen():
    #    for _ in range(iter_repeat):
    #        yield all_inputs, all_labels

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
            running_loss = 0.0
    print('Finished Training')

if __name__ == "__main__":
    print(device)
    trajectories = 
    net = 

    igor_supervised_learning_cuda(trajectories, net, use_prox=False, iter_repeat=1, EPOCHS=1, lr=1e-2):



