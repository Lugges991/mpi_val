import torch


def encode_to_batch_times_class_times_spatial(
        tensor: torch.Tensor,
        num_classes: int,
        spatial_dim: int = 3) -> torch.Tensor:
    """
    Encode tensor to match shape [N x C x R].

    N for size of batch, C for number of classes, R for spatial expansion.

    Parameters
    ----------
    tensor : torch.Tensor
        tensor with shape [R] or [C, R] if N == 1, [N, R], or [N, C, R] else.
    num_classes : int
        number of classes C.
    spatial_dim : int, optional
        number of spatial dimensions. The default is 3.

    Raises
    ------
    ValueError
        If spatial dimensionality is too low.
        If spatial dimensionality is too high.

    Returns
    -------
    tensor_encoded : torch.Tensor
        Tensor with shape [N x C x spatial expansion].

    """
    device = tensor.device

    if tensor.dim() < spatial_dim:
        raise ValueError('Tensor spatial dimensionality is too low.')
    if tensor.dim() > spatial_dim + 2:
        raise ValueError('Tensor spatial dimensionality is too high.')

    if tensor.dim() == spatial_dim:
        tensor = tensor.unsqueeze(0).to(device)

    tensor_shape = list(tensor.shape)  # [N, R]
    if tensor.dim() == spatial_dim + 1:
        if tensor_shape[0] == num_classes:
            tensor_encoded = tensor.unsqueeze(0).to(device)
        else:
            tensor_encoded = torch.zeros([tensor_shape[0], num_classes] +
                                         tensor_shape[-spatial_dim:]).to(
                device)
            tensor_encoded.scatter_(1, tensor.long().unsqueeze(1), 1)
    elif tensor.dim() == spatial_dim + 2:
        if tensor_shape[1] == num_classes:
            tensor_encoded = tensor.clone()
        else:
            tensor_encoded = torch.zeros([tensor_shape[0], num_classes] +
                                         tensor_shape[-spatial_dim:]).to(
                device)
            tensor_encoded.scatter_(1, tensor.long(), 1)

    return tensor_encoded




def dice_coefficient(output: torch.Tensor,
                     target: torch.Tensor,
                     smooth: float = 10e-5) -> torch.Tensor:
    r"""
    Compute the Dice dissimilarity regarding a boolean tensor.

    The Dice dissimilarity between `X` and `Y` is

    .. math::

        \frac{2 | X \cap Y |}{|X| + |Y|}

    Parameters
    ----------
    output : torch.Tensor
        prediction of a model. Values can be discrete or contigious.
    target : torch.Tensor
        true values. They should be binary.
    smooth : float, optional
        softens the result. The default is 10e-5.

    Returns
    -------
    dice_score : torch.Tensor
        Dice similarity coefficient.

    """
    # Note, that .flatten is equivalent to .contiguous().view(-1)
    tmp_output = output.flatten()
    tmp_target = target.flatten()
    intersection = (tmp_output * tmp_target).sum()

    dice_score = ((2. * intersection).clamp(min=smooth) /
                  (tmp_output.sum() + tmp_target.sum()).clamp(min=smooth))

    return dice_score


def continous_dice_coefficient(output: torch.Tensor,
                               target: torch.Tensor,
                               smooth: float = 10e-5) -> torch.Tensor:
    r"""
    Extend the definition of the classical Dice coefficient.

    See paper: https://www.biorxiv.org/content/10.1101/306977v1

    .. math::

        \frac{2 | X \cap Y |}{c |X| + |Y|} \\

    Parameters
    ----------
    output : torch.Tensor
        DESCRIPTION.
    target : torch.Tensor
        DESCRIPTION.
    smooth : float, optional
        DESCRIPTION. The default is 10e-5.

    Returns
    -------
    continous_dice_score : TYPE
        DESCRIPTION.

    """
    tmp_output = output.flatten()
    tmp_target = target.flatten()
    intersection = (tmp_output * tmp_target).sum()

    continous_factor = (tmp_output * torch.sign(tmp_target)).sum()

    continous_dice_score = (
        (2. * intersection).clamp(min=smooth) /
        (continous_factor * tmp_output.sum()
         + tmp_target.sum()).clamp(min=smooth)
    )

    return continous_dice_score


def dice_score_per_class(output: torch.Tensor,
                         target: torch.Tensor,
                         class_dim: int = 1,
                         smooth: float = 10e-5) -> torch.Tensor:
    r"""
    Calculate dice score per class.

    Parameters
    ----------
    output : torch.Tensor
        tensor with classes one-hot encoded in class_dim.
    target : torch.Tensor
        tensor with classes one-hot encoded in class_dim.
    class_dim : int, optional
        specifies in which dim classes are specified. The default is 1.
    smooth : float, optional
        soften the result. The default is 10e-5.

    Raises
    ------
    ValueError
        If shape of output and target do not match.

    Returns
    -------
    dice_score_per_class : torch.Tensor
        dices of shape of num classes C.

    """
    if target.shape != output.shape:
        raise ValueError("Target shape and output shape do not match.")

    dims = torch.arange(output.dim())
    dims = tuple(dims[dims != class_dim])

    output_sum_per_class = output.sum(dim=dims)
    target_sum_per_class = target.sum(dim=dims)
    intersection_per_class = (output * target).sum(dim=dims)
    dice_score_per_class = (
        (2. * intersection_per_class).clamp(min=smooth) /
        (output_sum_per_class + target_sum_per_class).clamp(min=smooth))

    return dice_score_per_class


def generalized_dice_score(output: torch.Tensor,
                           target: torch.Tensor,
                           num_classes: int,
                           classes_to_ignore: List = [],
                           spatial_dim: int = 3) -> torch.Tensor:
    """
    Measure overlap of labelled regions by label.

    There are several variations of generalized Dice scores. Our function
    follows Crum et al. https://ieeexplore.ieee.org/document/1717643

    Parameters
    ----------
    output : torch.Tensor
        model output logits.
    target : torch.Tensor
        true values.
    num_classes : int
        number of classes to predict.
    classes_to_ignore : List, optional
        Classes which can be ignored. The default is [].
    spatial_dim : int, optional
        spatial dimensionality. The default is 3.

    Raises
    ------
    ValueError
        If target and output do not have the same spatial expansion..

    Returns
    -------
    dice_score_per_class: torch.Tensor
        dices with shape [num_classes - len(classes_to_ignore)].

    """
    if output.shape[-spatial_dim:] != target.shape[-spatial_dim:]:
        raise ValueError(
            "Target and output do not have the same spatial expansion.")

    # Fit target to B x C x spatial
    output_encoded = encode_to_batch_times_class_times_spatial(
        output, num_classes, spatial_dim)
    target_encoded = encode_to_batch_times_class_times_spatial(
        target, num_classes, spatial_dim)

    if classes_to_ignore != []:
        classes = torch.arange(num_classes)
        classes_index = torch.all(
            torch.stack(
                [classes != class_to_ignore
                 for class_to_ignore in classes_to_ignore]), dim=0)
        output_encoded = output_encoded[:, classes_index]
        target_encoded = target_encoded[:, classes_index]

    return dice_score_per_class(output_encoded, target_encoded, class_dim=1)


def intersection_over_union(
        output: torch.Tensor,
        target: torch.Tensor,
        labels: torch.Tensor = None,
        pos_label: int = 1,
        batch_iou: bool = False,
        non_presence_threshold: float = None,
        reduction: str = 'mean',
        smooth: float = 10e-6,
) -> torch.Tensor:
    r"""
    Measure the similarity of output and target.

    Parameters
    ----------
    output : torch.Tensor
        tensor with labels one-hot encoded in pos_label.
    target : torch.Tensor
        tensor with labels one-hot encoded in pos_label.
    labels : torch.Tensor, optional
        labels to include in IoU. The default is None.
    pos_label : int, optional
        specifies in which dim labels are specified. The default is 1.
    batch_iou : bool, optional
        Calculate IoU for each sample. The default is False.
    non_presence_threshold : float, optional
        Threshold at which number of pixels to include. The default is None.
    reduction : str, optional
        Modality how to reduce result. The default is 'mean'.
    smooth : float, optional
        soften the result. The default is 10e-6.

    Raises
    ------
    ValueError
        If reduction is not available.

    Returns
    -------
    jaccard_index : torch.Tensor
        Intersection over Union. NaN if classes are not presence.

    """
    if reduction not in ['mean', 'sum', 'none']:
        raise ValueError(f"Reduction {reduction} is not available.")

    if labels is not None:
        output = torch.index_select(output, dim=pos_label, index=labels)
        target = torch.index_select(target, dim=pos_label, index=labels)

    dims = torch.arange(output.dim())
    dims = tuple(dims[dims != pos_label])

    if batch_iou:
        dims = dims[1:]

    output_sum_per_class = output.sum(dim=dims)
    target_sum_per_class = target.sum(dim=dims)
    intersection_per_class = (output * target).sum(dim=dims)

    jaccard_index = (
        intersection_per_class.clamp(min=smooth)
        / (
            output_sum_per_class
            + target_sum_per_class
            - intersection_per_class
        ).clamp(min=smooth)
    )

    if non_presence_threshold is not None:
        jaccard_index[
            (target_sum_per_class + output_sum_per_class)
            <= non_presence_threshold
        ] = torch.nan

    if reduction == 'mean':
        jaccard_index = torch.nanmean(jaccard_index)
    elif reduction == 'sum':
        jaccard_index = torch.nansum(jaccard_index)

    return jaccard_index

