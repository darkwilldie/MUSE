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


def randomly_permute_samples(n_sample, data_x, data_y, label_x=None, label_y=None):
    random_idx = np.random.permutation(n_sample)
    data_train_x = data_x[random_idx, :]
    data_train_y = data_y[random_idx, :]
    if label_x is not None and label_y is not None:
        label_train_x = label_x[random_idx]
        label_train_y = label_y[random_idx]
        return data_train_x, data_train_y, label_train_x, label_train_y
    return data_train_x, data_train_y


def get_batch_tensors(i, n_sample, data_x, data_y, label_x=None, label_y=None):

    offset = (i * batch_size) % (n_sample)
    batch_x_input = torch.tensor(
        data_x[offset : (offset + batch_size), :], dtype=torch.float32
    ).to(device)
    batch_y_input = torch.tensor(
        data_y[offset : (offset + batch_size), :], dtype=torch.float32
    ).to(device)
    if label_x is not None and label_y is not None:
        label_x_input = torch.tensor(
            label_x[offset : (offset + batch_size)], dtype=torch.float32
        ).to(device)
        label_y_input = torch.tensor(
            label_y[offset : (offset + batch_size)], dtype=torch.float32
        ).to(device)
    else:
        label_x_input = torch.zeros(batch_x_input.size(0)).to(device)
        label_y_input = torch.zeros(batch_y_input.size(0)).to(device)
    return batch_x_input, batch_y_input, label_x_input, label_y_input


def get_all_tensor(data_train_x, data_train_y, label_train_x=None, label_train_y=None):
    data_train_x_tensor = torch.tensor(data_train_x, dtype=torch.float32).to(device)
    data_train_y_tensor = torch.tensor(data_train_y, dtype=torch.float32).to(device)
    if label_train_x is not None and label_train_y is not None:
        label_train_x_tensor = torch.tensor(label_train_x, dtype=torch.float32).to(
            device
        )
        label_train_y_tensor = torch.tensor(label_train_y, dtype=torch.float32).to(
            device
        )
        return (
            data_train_x_tensor,
            data_train_y_tensor,
            label_train_x_tensor,
            label_train_y_tensor,
        )
    else:
        label_train_x_tensor = torch.zeros(data_train_x.shape[0]).to(device)
        label_train_y_tensor = torch.zeros(data_train_y.shape[0]).to(device)
    return (
        data_train_x_tensor,
        data_train_y_tensor,
        label_train_x_tensor,
        label_train_y_tensor,
    )


""" Model fitting and feature prediction of MUSE """


def muse_fit_predict(
    data_x,
    data_y,
    label_x,
    label_y,
    latent_dim=100,
    n_epochs=500,
    weight_penalty=5,
    triplet_lambda=5,
):

    # read data-specific parameters from inputs
    feature_dim_x = data_x.shape[1]
    feature_dim_y = data_y.shape[1]
    n_sample = data_x.shape[0]

    model = MUSE(
        feature_dim_x,
        feature_dim_y,
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
        data_train_x, data_train_y = randomly_permute_samples(n_sample, data_x, data_y)

        for i in range(total_batch):
            # input data batches
            batch_x_input, batch_y_input, label_x_input, label_y_input = (
                get_batch_tensors(i, n_sample, data_train_x, data_train_y)
            )

            optimizer.zero_grad()
            _, _, _, _, _, loss, _, _, _, _ = model(
                batch_x_input, batch_y_input, label_x_input, label_y_input
            )
            loss.backward()
            optimizer.step()

        # calculate and print loss terms for current epoch
        if epoch % print_epochs == 0:
            with torch.no_grad():
                (
                    data_train_x_tensor,
                    data_train_y_tensor,
                    label_train_x_tensor,
                    label_train_y_tensor,
                ) = get_all_tensor(data_train_x, data_train_y)

                (
                    _,
                    _,
                    _,
                    _,
                    _,
                    loss,
                    reconstruction_error,
                    sparse_penalty,
                    _,
                    _,
                ) = model(
                    data_train_x_tensor,
                    data_train_y_tensor,
                    label_train_x_tensor,
                    label_train_y_tensor,
                )

                print(
                    f"epoch: {epoch}, \t total loss: {loss.item():03.5f},\t reconstruction loss: {reconstruction_error.item():03.5f},\t sparse penalty: {sparse_penalty.item():03.5f}"
                )

    # estimate the margin for the triplet loss
    with torch.no_grad():
        data_x_tensor, data_y_tensor, label_x_tensor, label_y_tensor = get_all_tensor(
            data_x, data_y
        )

        latent, reconstruct_x, reconstruct_y, _, _, _, _, _, _, _ = model(
            data_x_tensor,
            data_y_tensor,
            label_x_tensor,
            label_y_tensor,
        )
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
        data_train_x, data_train_y, label_train_x, label_train_y = (
            randomly_permute_samples(n_sample, data_x, data_y, label_x, label_y)
        )

        for i in range(total_batch):
            # data batches
            batch_x_input, batch_y_input, label_x_input, label_y_input = (
                get_batch_tensors(
                    i,
                    n_sample,
                    data_train_x,
                    data_train_y,
                    label_train_x,
                    label_train_y,
                )
            )

            optimizer.zero_grad()
            _, _, _, _, _, loss, _, _, _, _ = model(
                batch_x_input,
                batch_y_input,
                label_x_input,
                label_y_input,
                triplet_margin=margin_estimate,
                triplet_lambda=triplet_lambda,
            )
            loss.backward()
            optimizer.step()

        # calculate loss on all input data for current epoch
        if epoch % print_epochs == 0:
            with torch.no_grad():
                (
                    data_train_x_tensor,
                    data_train_y_tensor,
                    label_train_x_tensor,
                    label_train_y_tensor,
                ) = get_all_tensor(
                    data_train_x, data_train_y, label_train_x, label_train_y
                )

                (
                    _,
                    _,
                    _,
                    _,
                    _,
                    loss,
                    reconstruction_error,
                    weight_penalty,
                    trip_loss_x,
                    trip_loss_y,
                ) = model(
                    data_train_x_tensor,
                    data_train_y_tensor,
                    label_train_x_tensor,
                    label_train_y_tensor,
                    triplet_margin=margin_estimate,
                    triplet_lambda=triplet_lambda,
                )

                print(
                    f"epoch: {epoch}, \t total loss: {loss.item():03.5f},\t reconstruction loss: {reconstruction_error.item():03.5f},\t sparse penalty: {weight_penalty.item():03.5f},\t x triplet: {trip_loss_x.item():03.5f},\t y triplet: {trip_loss_y.item():03.5f}"
                )

    # update cluster labels based modality-specific latents
    with torch.no_grad():
        data_x_tensor, data_y_tensor, label_x_tensor, label_y_tensor = get_all_tensor(
            data_x, data_y, label_x, label_y
        )

        _, _, _, latent_x, latent_y, _, _, _, _, _ = model(
            data_x_tensor,
            data_y_tensor,
            label_x_tensor,
            label_y_tensor,
            triplet_margin=margin_estimate,
            triplet_lambda=triplet_lambda,
        )
        latent_x = latent_x.cpu().numpy()
        latent_y = latent_y.cpu().numpy()

        # update cluster labels using PhenoGraph
        label_x_update, _, _ = phenograph.cluster(latent_x)
        label_y_update, _, _ = phenograph.cluster(latent_y)
        print("Finish initialization of MUSE")

    """ Training of MUSE """
    for epoch in range(n_epochs):
        # randomly permute samples
        data_train_x, data_train_y, label_train_x, label_train_y = (
            randomly_permute_samples(
                n_sample, data_x, data_y, label_x_update, label_y_update
            )
        )

        # loop over all batches
        for i in range(total_batch):
            # batch data
            batch_x_input, batch_y_input, batch_label_x_input, batch_label_y_input = (
                get_batch_tensors(
                    i,
                    n_sample,
                    data_train_x,
                    data_train_y,
                    label_train_x,
                    label_train_y,
                )
            )

            optimizer.zero_grad()
            _, _, _, _, _, loss, _, _, _, _ = model(
                batch_x_input,
                batch_y_input,
                batch_label_x_input,
                batch_label_y_input,
                triplet_margin=margin_estimate,
                triplet_lambda=triplet_lambda,
            )
            loss.backward()
            optimizer.step()

        # calculate and print losses on whole training dataset
        if epoch % print_epochs == 0:
            with torch.no_grad():
                (
                    data_train_x_tensor,
                    data_train_y_tensor,
                    label_train_x_tensor,
                    label_train_y_tensor,
                ) = get_all_tensor(
                    data_train_x, data_train_y, label_train_x, label_train_y
                )
                (
                    _,
                    _,
                    _,
                    _,
                    _,
                    loss,
                    reconstruction_error,
                    weight_penalty,
                    trip_loss_x,
                    trip_loss_y,
                ) = model(
                    data_train_x_tensor,
                    data_train_y_tensor,
                    label_train_x_tensor,
                    label_train_y_tensor,
                    triplet_margin=margin_estimate,
                    triplet_lambda=triplet_lambda,
                )

                print(
                    f"epoch: {epoch}, \t total loss: {loss.item():03.5f},\t reconstruction loss: {reconstruction_error.item():03.5f},\t sparse penalty: {weight_penalty.item():03.5f},\t x triplet loss: {trip_loss_x.item():03.5f},\t y triplet loss: {trip_loss_y.item():03.5f}"
                )

        # update cluster labels based on new modality-specific latent representations
        if epoch % cluster_update_epoch == 0:
            with torch.no_grad():
                data_x_tensor, data_y_tensor, label_x_tensor, label_y_tensor = (
                    get_all_tensor(data_x, data_y, label_x_update, label_y_update)
                )

                _, _, _, latent_x, latent_y, _, _, _, _, _ = model(
                    data_x_tensor,
                    data_y_tensor,
                    label_x_tensor,
                    label_y_tensor,
                    triplet_margin=margin_estimate,
                    triplet_lambda=triplet_lambda,
                )
                latent_x = latent_x.cpu().numpy()
                latent_y = latent_y.cpu().numpy()

                # use PhenoGraph to obtain cluster label
                label_x_update, _, _ = phenograph.cluster(latent_x)
                label_y_update, _, _ = phenograph.cluster(latent_y)

    """ MUSE output """
    with torch.no_grad():
        data_x_tensor, data_y_tensor, label_x_tensor, label_y_tensor = get_all_tensor(
            data_x, data_y, label_x_update, label_y_update
        )

        latent, reconstruct_x, reconstruct_y, latent_x, latent_y, _, _, _, _, _ = model(
            data_x_tensor,
            data_y_tensor,
            label_x_tensor,
            label_y_tensor,
            triplet_margin=margin_estimate,
            triplet_lambda=triplet_lambda,
        )
        latent = latent.cpu().numpy()
        reconstruct_x = reconstruct_x.cpu().numpy()
        reconstruct_y = reconstruct_y.cpu().numpy()
        latent_x = latent_x.cpu().numpy()
        latent_y = latent_y.cpu().numpy()

    print("++++++++++ MUSE completed ++++++++++")

    return latent, reconstruct_x, reconstruct_y, latent_x, latent_y
