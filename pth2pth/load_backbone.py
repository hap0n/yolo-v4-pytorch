def load_conv_weight(conv_block, conv_block_state_dict):
    layers = conv_block._modules['layers'] # noqa

    conv = layers[0]
    conv_block_state_dict_list = list(conv_block_state_dict.items())

    if conv.bias is not None:
        conv.load_state_dict(
            dict([
                    ("weight", conv_block_state_dict_list[0][1]),
                    ("bias", conv_block_state_dict_list[1][1])
            ])
        )
        i = 2
    else:
        conv.load_state_dict(
            dict([
                ("weight", conv_block_state_dict_list[0][1])
            ])
        )
        i = 1

    if len(layers) == 3:
        bn = layers[1]
        bn.load_state_dict(
            dict([
                ("weight", conv_block_state_dict_list[i][1]),
                ("bias", conv_block_state_dict_list[i+1][1]),
                ("running_mean", conv_block_state_dict_list[i+2][1]),
                ("running_var", conv_block_state_dict_list[i+3][1]),
                ("num_batches_tracked", conv_block_state_dict_list[i+4][1])
            ])
        )


def split_downsample_dict(state_dict):
    # [48, 54, 126, 126, 78]
    layers = list(state_dict.items())
    return (
        dict(layers[:48]),
        dict(layers[48:48+54]),
        dict(layers[48+54:48+54+126]),
        dict(layers[48+54+126:48+54+126+126]),
        dict(layers[48+54+126+126:]),
    )


def load_downsample_weights(down, state_dict):
    state_dict_list = list(state_dict.items())
    for i in range(8):
        conv_block = down._modules[f'conv{i}'] # noqa
        load_conv_weight(conv_block, dict(state_dict_list[i*6:(i+1)*6]))


def load_res_weights(res_block, state_dict):
    state_dict_list = list(state_dict.items())
    i = 0
    for module in res_block.module_list:
        j = 0
        for res in module:
            load_conv_weight(res, dict(state_dict_list[i*12+j*6:i*12+j*6+6]))
            j += 1
        i += 1


def load_csp_block_weights(scp_block, state_dict):
    state_dict_list = list(state_dict.items())

    load_conv_weight(scp_block.conv0, dict(state_dict_list[:6]))
    load_conv_weight(scp_block.conv1, dict(state_dict_list[6:12]))
    load_conv_weight(scp_block.conv2, dict(state_dict_list[12:18]))

    load_res_weights(scp_block.res_block, dict(state_dict_list[18:-12]))

    load_conv_weight(scp_block.conv3, dict(state_dict_list[-12:-6]))
    load_conv_weight(scp_block.conv4, dict(state_dict_list[-6:]))


def load_darknet_weights(darknet, state_dict):
    sd1, sd2, sd3, sd4, sd5 = split_downsample_dict(state_dict)

    load_downsample_weights(darknet.block0, sd1)
    load_csp_block_weights(darknet.block1, sd2)
    load_csp_block_weights(darknet.block2, sd3)
    load_csp_block_weights(darknet.block3, sd4)
    load_csp_block_weights(darknet.block4, sd5)
