def load_conv_weight(conv_block, state_dict):
    layers = conv_block._modules['layers'] # noqa

    conv = layers[0]
    state_dict_list = list(state_dict.items())

    if conv.bias is not None:
        conv.load_state_dict(
            dict([
                    ("weight", state_dict_list[0][1]),
                    ("bias", state_dict_list[1][1])
            ])
        )
        i = 2
    else:
        conv.load_state_dict(
            dict([
                ("weight", state_dict_list[0][1])
            ])
        )
        i = 1

    if len(layers) == 3:
        bn = layers[1]
        bn.load_state_dict(
            dict([
                ("weight", state_dict_list[i][1]),
                ("bias", state_dict_list[i+1][1]),
                ("running_mean", state_dict_list[i+2][1]),
                ("running_var", state_dict_list[i+3][1]),
                ("num_batches_tracked", state_dict_list[i+4][1])
            ])
        )
