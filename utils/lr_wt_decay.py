# Modified from Masked Autoencoder: https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8/util/lr_decay.py

# References:
# ELECTRA https://github.com/google-research/electra
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# MAE: https://github.com/facebookresearch/mae

def param_groups_lrd(model, lr, weight_decay=1e-6, exclude_1d_params=True,
                     no_weight_decay_list=(), layer_decay=0.):
    """
    Parameter groups for layer-wise lr decay
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    """
    param_group_names = {}
    param_groups = {}

    num_layers = len(model.blocks) + 1

    layer_scales = list(layer_decay ** (num_layers - i)
                        for i in range(num_layers + 1))

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # no decay: all 1D parameters (batch/layer norm and bias)
        # and model specific ones
        if (exclude_1d_params and param.ndim == 1) or name in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.
        else:
            g_decay = "decay"
            this_decay = weight_decay

        layer_id = get_layer_id_for_vit(name, num_layers)
        group_name = f"layer_{layer_id}_{g_decay}"

        if group_name not in param_group_names:
            this_scale = layer_scales[layer_id]

            param_group_names[group_name] = {
                "lr": lr * this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
            param_groups[group_name] = {
                "lr": lr * this_scale,
                "weight_decay": this_decay,
                "params": [],
            }

        param_group_names[group_name]["params"].append(name)
        param_groups[group_name]["params"].append(param)

    return list(param_groups.values())


def get_layer_id_for_vit(name, num_layers):
    """
    Assign a parameter with its layer id
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
    """
    if name in {'cls_token', 'pos_embed'} or name.startswith('patch_embed'):
        return 0
    elif name.startswith('blocks'):
        return int(name.split('.')[1]) + 1
    else:
        return num_layers


def exclude_from_wt_decay(model, weight_decay, skip_list=()):
    params = []
    excluded_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if param.dim == 1 or name in skip_list:
            excluded_params.append(param)
        else:
            params.append(param)

    return [
        {"params": params, "weight_decay": weight_decay},
        {"params": excluded_params, "weight_decay": 0.0},
    ]
