import numpy as np
from .muse_architecture import MUSE
from scipy.spatial.distance import pdist
import phenograph
import torch
import torch.optim as optim


""" initial parameter setting """
# parameter setting for neural network
n_hidden = 128  # number of hidden node in neural network
learn_rate = 1e-4  # learning rate in the optimization
batch_size = 64  # number of cells in the training batch
n_epochs_init = 200  # number of training epoch in model initialization
print_epochs = 50  # epoch interval to display the current training loss
cluster_update_epoch = 200  # epoch interval to update modality-specific clusters
""" construct computation graph using PyTorch """
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def randomly_permute_samples(n_sample, inputs, labels=None):
    random_idx = np.random.permutation(n_sample)
    data_train = [inputs[i][random_idx, :] for i in range(len(inputs))]
    if labels is not None:
        label_train = [labels[i][random_idx] for i in range(len(labels))]
        return data_train, label_train
    return data_train


def get_batch_tensors(i, n_sample, data_inputs, label_inputs=None):
    offset = (i * batch_size) % (n_sample)

    # 先将列表转换为单个numpy数组，再转换为tensor
    batch_inputs = [
        data_input[offset : (offset + batch_size), :].reshape(batch_size, -1)
        for data_input in data_inputs
    ]
    # 使用np.stack将列表转换为单个numpy数组
    batch_inputs = np.stack(batch_inputs)
    batch_inputs = torch.from_numpy(batch_inputs).float().to(device)

    if label_inputs is not None:
        batch_labels = [
            label_input[offset : (offset + batch_size)].reshape(-1)
            for label_input in label_inputs
        ]
        # 同样优化标签数据的转换
        batch_labels = np.stack(batch_labels)
        batch_labels = torch.from_numpy(batch_labels).float().to(device)
        return batch_inputs, batch_labels

    return batch_inputs


def get_all_tensor(data_inputs, label_inputs=None):
    data_inputs = np.stack(data_inputs)
    data_inputs_tensor = torch.from_numpy(data_inputs).float().to(device)
    if label_inputs is not None:
        label_inputs = np.stack(label_inputs)
        label_inputs_tensor = torch.from_numpy(label_inputs).float().to(device)
        return data_inputs_tensor, label_inputs_tensor
    return data_inputs_tensor


""" Model fitting and feature prediction of MUSE """


def muse_fit_predict(
    data_inputs,
    label_inputs,
    latent_dim=100,
    n_epochs=500,
    weight_penalty=5,
    triplet_lambda=5,
):

    # read data-specific parameters from inputs
    feature_dims = [data_input.shape[1] for data_input in data_inputs]
    n_sample = data_inputs[0].shape[0]

    model = MUSE(
        feature_dims,
        latent_dim,
        n_hidden,
        weight_penalty,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learn_rate)

    print("++++++++++ MUSE for multi-modality single-cell analysis ++++++++++")
    """ MUSE optimization """
    total_batch = int(n_sample / batch_size)

    model.train()
    for epoch in range(n_epochs_init):
        # randomly permute samples
        data_train_inputs = randomly_permute_samples(n_sample, data_inputs)

        for i in range(total_batch):
            # input data batches
            batch_inputs = get_batch_tensors(i, n_sample, data_train_inputs)

            optimizer.zero_grad()
            _, _, _, loss, _, _, _ = model(batch_inputs)
            loss.backward()
            optimizer.step()

        # calculate and print loss terms for current epoch
        if epoch % print_epochs == 0:
            with torch.no_grad():
                data_inputs_tensor = get_all_tensor(data_train_inputs)

                (
                    _,
                    _,
                    _,
                    loss,
                    reconstruction_error,
                    sparse_penalty,
                    _,
                ) = model(data_inputs_tensor)

                print(
                    f"epoch: {epoch}, \t total loss: {loss.item():03.5f},\t reconstruction loss: {reconstruction_error.item():03.5f},\t sparse penalty: {sparse_penalty.item():03.5f}"
                )

    # estimate the margin for the triplet loss
    with torch.no_grad():
        data_inputs_tensor = get_all_tensor(data_inputs)

        latent, _, _, _, _, _, _ = model(data_inputs_tensor)
        latent = latent.cpu().numpy()
        latent_pd_matrix = pdist(latent, "euclidean")
        latent_pd_sort = np.sort(latent_pd_matrix)
        select_top_n = int(latent_pd_sort.size * 0.2)
        margin_estimate = np.median(latent_pd_sort[-select_top_n:]) - np.median(
            latent_pd_sort[:select_top_n]
        )

    # refine MUSE parameters with reference labels and triplet losses
    for epoch in range(n_epochs_init):
        # randomly permute samples
        data_train_inputs, label_train_inputs = randomly_permute_samples(
            n_sample, data_inputs, label_inputs
        )

        for i in range(total_batch):
            # data batches
            batch_train_inputs, batch_train_labels = get_batch_tensors(
                i,
                n_sample,
                data_train_inputs,
                label_train_inputs,
            )
            optimizer.zero_grad()
            _, _, _, loss, _, _, _ = model(
                batch_train_inputs,
                batch_train_labels,
                triplet_margin=margin_estimate,
                triplet_lambda=triplet_lambda,
            )
            loss.backward()
            optimizer.step()

        # calculate loss on all input data for current epoch
        if epoch % print_epochs == 0:
            with torch.no_grad():
                data_train_inputs_tensor, label_train_inputs_tensor = get_all_tensor(
                    data_train_inputs, label_train_inputs
                )

                (
                    _,
                    _,
                    _,
                    loss,
                    reconstruction_error,
                    weight_penalty,
                    trip_loss,
                ) = model(
                    data_train_inputs_tensor,
                    label_train_inputs_tensor,
                    triplet_margin=margin_estimate,
                    triplet_lambda=triplet_lambda,
                )

                print(
                    f"epoch: {epoch}, \t total loss: {loss.item():03.5f},\t reconstruction loss: {reconstruction_error.item():03.5f},\t sparse penalty: {weight_penalty.item():03.5f},\t triplet loss: {trip_loss.item():03.5f}"
                )

    # update cluster labels based modality-specific latents
    with torch.no_grad():
        data_inputs_tensor, label_inputs_tensor = get_all_tensor(
            data_inputs, label_inputs
        )

        latent, _, encoded_inputs, _, _, _, _ = model(
            data_inputs_tensor,
            label_inputs_tensor,
            triplet_margin=margin_estimate,
            triplet_lambda=triplet_lambda,
        )
        encoded_inputs = encoded_inputs.cpu().numpy()

        # update cluster labels using PhenoGraph
        # print(encoded_inputs.shape)
        # print(latent.shape)
        # quit()
        labels_update = [phenograph.cluster(z)[0] for z in encoded_inputs]
        print("Finish initialization of MUSE")

    """ Training of MUSE """
    for epoch in range(n_epochs):
        # randomly permute samples
        data_train_inputs, label_train_inputs = randomly_permute_samples(
            n_sample, data_inputs, labels_update
        )

        # loop over all batches
        for i in range(total_batch):
            # batch data
            batch_inputs, label_inputs = get_batch_tensors(
                i,
                n_sample,
                data_train_inputs,
                label_train_inputs,
            )

            optimizer.zero_grad()
            _, _, _, loss, _, _, _ = model(
                batch_inputs,
                label_inputs,
                triplet_margin=margin_estimate,
                triplet_lambda=triplet_lambda,
            )
            loss.backward()
            optimizer.step()

        # calculate and print losses on whole training dataset
        if epoch % print_epochs == 0:
            with torch.no_grad():
                data_train_inputs_tensor, label_train_inputs_tensor = get_all_tensor(
                    data_train_inputs, label_train_inputs
                )
                (
                    _,
                    _,
                    _,
                    loss,
                    reconstruction_error,
                    weight_penalty,
                    trip_loss,
                ) = model(
                    data_train_inputs_tensor,
                    label_train_inputs_tensor,
                    triplet_margin=margin_estimate,
                    triplet_lambda=triplet_lambda,
                )

                print(
                    f"epoch: {epoch}, \t total loss: {loss.item():03.5f},\t reconstruction loss: {reconstruction_error.item():03.5f},\t sparse penalty: {weight_penalty.item():03.5f},\t triplet loss: {trip_loss.item():03.5f}"
                )

        # update cluster labels based on new modality-specific latent representations
        if epoch % cluster_update_epoch == 0:
            with torch.no_grad():
                data_inputs_tensor, label_inputs_tensor = get_all_tensor(
                    data_inputs, labels_update
                )

                _, _, encoded_inputs, _, _, _, _ = model(
                    data_inputs_tensor,
                    label_inputs_tensor,
                    triplet_margin=margin_estimate,
                    triplet_lambda=triplet_lambda,
                )
                encoded_inputs = encoded_inputs.cpu().numpy()

                # use PhenoGraph to obtain cluster label
                labels_update = [phenograph.cluster(z)[0] for z in encoded_inputs]

    """ MUSE output """
    with torch.no_grad():
        data_inputs_tensor, label_inputs_tensor = get_all_tensor(
            data_inputs, labels_update
        )

        latent, reconstruct_inputs, encoded_inputs, _, _, _, _ = model(
            data_inputs_tensor,
            label_inputs_tensor,
            triplet_margin=margin_estimate,
            triplet_lambda=triplet_lambda,
        )
        latent = latent.cpu().numpy()
        reconstruct_inputs = reconstruct_inputs.cpu().numpy()
        encoded_inputs = encoded_inputs.cpu().numpy()

    print("++++++++++ MUSE completed ++++++++++")

    return latent, reconstruct_inputs, encoded_inputs
