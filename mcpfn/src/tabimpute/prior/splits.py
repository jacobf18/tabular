from __future__ import annotations

import torch


def create_train_test_sets(
    X: torch.Tensor, X_full: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    """Create train/test sets from a matrix with missing values."""
    missing_indices, non_missing_indices = torch.where(torch.isnan(X)), torch.where(
        ~torch.isnan(X)
    )
    train_X_list, test_X_list, train_y_list, test_y_list = [], [], [], []

    for i, j in zip(non_missing_indices[0], non_missing_indices[1]):
        row = X[i, :]
        col = X[:, j]
        train_X_list.append(
            torch.cat((torch.tensor([i, j], device=X.device), row, col))
        )
        train_y_list.append(X_full[i, j])

    for i, j in zip(missing_indices[0], missing_indices[1]):
        row = X[i, :]
        col = X[:, j]
        test_X_list.append(torch.cat((torch.tensor([i, j], device=X.device), row, col)))
        test_y_list.append(X_full[i, j])

    train_X = (
        torch.stack(train_X_list)
        if train_X_list
        else torch.empty(0, X.shape[0] + X.shape[1] + 2)
    )
    train_y = torch.stack(train_y_list) if train_y_list else torch.empty(0)
    test_X = (
        torch.stack(test_X_list)
        if test_X_list
        else torch.empty(0, X.shape[0] + X.shape[1] + 2)
    )
    test_y = torch.stack(test_y_list) if test_y_list else torch.empty(0)

    del train_X_list, train_y_list, test_X_list, test_y_list
    torch.cuda.empty_cache()

    return train_X, train_y, test_X, test_y
