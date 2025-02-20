import numpy as np
from .muse_architecture import MUSE
from scipy.spatial.distance import pdist
import phenograph
from sklearn.cluster import KMeans
import torch
import torch.optim as optim


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
    n_cluster=None,
):
    """
    MUSE model fitting and predicting:
      This function is used to train the MUSE model on multi-modality data

    Parameters:
      data_x:       input for transcript modality; matrix of  n * p, where n = number of cells, p = number of genes.
      data_y:       input for morphological modality; matrix of n * q, where n = number of cells, q is the feature dimension.
      label_x:      initial reference cluster label for transcriptional modality.
      label_y:      inital reference cluster label for morphological modality.
      latent_dim:   feature dimension of joint latent representation.
      n_epochs:     maximal epoch used in training.
      lambda_regul: weight for regularization term in the loss function.
      lambda_super: weight for supervised learning loss in the loss function.

    Output:
      latent:       joint latent representation learned by MUSE.
      reconstruct_x:reconstructed feature matrix corresponding to input data_x.
      reconstruct_y:reconstructed feature matrix corresponding to input data_y.
      latent_x:     modality-specific latent representation corresponding to data_x.
      latent_y:     modality-specific latent representation corresponding to data_y.

    Feng Bao @ Altschuler & Wu Lab @ UCSF 2022.
    Software provided as is under MIT License.
    """

    """ initial parameter setting """
    # parameter setting for neural network
    n_hidden = 128  # number of hidden node in neural network
    learn_rate = 1e-4  # learning rate in the optimization
    batch_size = 512  # number of cells in the training batch
    n_epochs_init = 200  # number of training epoch in model initialization
    print_epochs = 50  # epoch interval to display the current training loss
    cluster_update_epoch = 200  # epoch interval to update modality-specific clusters

    # read data-specific parameters from inputs
    feature_dim_x = data_x.shape[1]
    feature_dim_y = data_y.shape[1]
    n_sample = data_x.shape[0]

    """ construct computation graph using PyTorch """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # data_x = torch.tensor(data_x, dtype=torch.float32, device=device)
    # data_y = torch.tensor(data_y, dtype=torch.float32, device=device)
    # label_x = torch.tensor(label_x, dtype=torch.float32, device=device)
    # label_y = torch.tensor(label_y, dtype=torch.float32, device=device)

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
        random_idx = np.random.permutation(n_sample)
        data_train_x = data_x[random_idx, :]
        data_train_y = data_y[random_idx, :]

        for i in range(total_batch):
            # input data batches
            offset = (i * batch_size) % (n_sample)
            batch_x_input = torch.tensor(
                data_train_x[offset : (offset + batch_size), :], dtype=torch.float32
            ).to(device)
            batch_y_input = torch.tensor(
                data_train_y[offset : (offset + batch_size), :], dtype=torch.float32
            ).to(device)
            label_x_input = torch.zeros(batch_x_input.size(0)).to(device)
            label_y_input = torch.zeros(batch_y_input.size(0)).to(device)

            optimizer.zero_grad()
            _, _, _, _, _, loss, _, _, _, _ = model(
                batch_x_input, batch_y_input, label_x_input, label_y_input
            )
            loss.backward()
            optimizer.step()

        # calculate and print loss terms for current epoch
        if epoch % print_epochs == 0:
            with torch.no_grad():
                data_train_x_tensor = torch.tensor(
                    data_train_x, dtype=torch.float32
                ).to(device)
                data_train_y_tensor = torch.tensor(
                    data_train_y, dtype=torch.float32
                ).to(device)
                label_train_x_tensor = torch.zeros(data_train_x.shape[0]).to(device)
                label_train_y_tensor = torch.zeros(data_train_y.shape[0]).to(device)

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
        data_x_tensor = torch.tensor(data_x, dtype=torch.float32).to(device)
        data_y_tensor = torch.tensor(data_y, dtype=torch.float32).to(device)
        label_x_zero_tensor = torch.zeros(data_x.shape[0]).to(device)
        label_y_zero_tensor = torch.zeros(data_y.shape[0]).to(device)

        latent, reconstruct_x, reconstruct_y, _, _, _, _, _, _, _ = model(
            data_x_tensor,
            data_y_tensor,
            label_x_zero_tensor,
            label_y_zero_tensor,
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
        random_idx = np.random.permutation(n_sample)
        data_train_x = data_x[random_idx, :]
        data_train_y = data_y[random_idx, :]
        label_train_x = label_x[random_idx]
        label_train_y = label_y[random_idx]

        for i in range(total_batch):
            # data batches
            offset = (i * batch_size) % (n_sample)
            batch_x_input = torch.tensor(
                data_train_x[offset : (offset + batch_size), :], dtype=torch.float32
            ).to(device)
            batch_y_input = torch.tensor(
                data_train_y[offset : (offset + batch_size), :], dtype=torch.float32
            ).to(device)
            label_x_input = torch.tensor(
                label_train_x[offset : (offset + batch_size)], dtype=torch.float32
            ).to(device)
            label_y_input = torch.tensor(
                label_train_y[offset : (offset + batch_size)], dtype=torch.float32
            ).to(device)

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
                data_train_x_tensor = torch.tensor(
                    data_train_x, dtype=torch.float32
                ).to(device)
                data_train_y_tensor = torch.tensor(
                    data_train_y, dtype=torch.float32
                ).to(device)
                label_train_x_tensor = torch.tensor(
                    label_train_x, dtype=torch.float32
                ).to(device)
                label_train_y_tensor = torch.tensor(
                    label_train_y, dtype=torch.float32
                ).to(device)

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
        data_x_tensor = torch.tensor(data_x, dtype=torch.float32).to(device)
        data_y_tensor = torch.tensor(data_y, dtype=torch.float32).to(device)
        label_x_tensor = torch.tensor(label_x, dtype=torch.float32).to(device)
        label_y_tensor = torch.tensor(label_y, dtype=torch.float32).to(device)

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
        # label_x_update, _, _ = phenograph.cluster(latent_x)
        # label_y_update, _, _ = phenograph.cluster(latent_y)
        label_x_update = cluster(latent_x, n_cluster)
        label_y_update = cluster(latent_y, n_cluster)
        print("Finish initialization of MUSE")

    """ Training of MUSE """
    for epoch in range(n_epochs):
        # randomly permute samples
        random_idx = np.random.permutation(n_sample)
        data_train_x = data_x[random_idx, :]
        data_train_y = data_y[random_idx, :]
        label_train_x = label_x_update[random_idx]
        label_train_y = label_y_update[random_idx]

        # loop over all batches
        for i in range(total_batch):
            # batch data
            offset = (i * batch_size) % (n_sample)
            batch_x_input = torch.tensor(
                data_train_x[offset : (offset + batch_size), :], dtype=torch.float32
            ).to(device)
            batch_y_input = torch.tensor(
                data_train_y[offset : (offset + batch_size), :], dtype=torch.float32
            ).to(device)
            batch_label_x_input = torch.tensor(
                label_train_x[offset : (offset + batch_size)], dtype=torch.float32
            ).to(device)
            batch_label_y_input = torch.tensor(
                label_train_y[offset : (offset + batch_size)], dtype=torch.float32
            ).to(device)

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
                data_train_x_tensor = torch.tensor(
                    data_train_x, dtype=torch.float32
                ).to(device)
                data_train_y_tensor = torch.tensor(
                    data_train_y, dtype=torch.float32
                ).to(device)
                label_train_x_tensor = torch.tensor(
                    label_train_x, dtype=torch.float32
                ).to(device)
                label_train_y_tensor = torch.tensor(
                    label_train_y, dtype=torch.float32
                ).to(device)

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
                data_x_tensor = torch.tensor(data_x, dtype=torch.float32).to(device)
                data_y_tensor = torch.tensor(data_y, dtype=torch.float32).to(device)
                label_x_tensor = torch.tensor(label_x, dtype=torch.float32).to(device)
                label_y_tensor = torch.tensor(label_y, dtype=torch.float32).to(device)

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
                # label_x_update, _, _ = phenograph.cluster(latent_x)
                # label_y_update, _, _ = phenograph.cluster(latent_y)
                label_x_update = cluster(latent_x, n_cluster)
                label_y_update = cluster(latent_y, n_cluster)

    """ MUSE output """
    with torch.no_grad():
        data_x_tensor = torch.tensor(data_x, dtype=torch.float32).to(device)
        data_y_tensor = torch.tensor(data_y, dtype=torch.float32).to(device)
        label_x_tensor = torch.tensor(label_x, dtype=torch.float32).to(device)
        label_y_tensor = torch.tensor(label_y, dtype=torch.float32).to(device)

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


def cluster(feature, n_cluster=None):
    if n_cluster:
        return KMeans(n_cluster, random_state=42).fit_predict(feature)
    return phenograph.cluster(feature, seed=42)[0]
