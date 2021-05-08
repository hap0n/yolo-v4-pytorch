from pth2pth.load_conv import load_conv_weight


def load_spp_weights(spp_block, state_dict):
    state_dict_list = list(state_dict.items()) # noqa

    load_conv_weight(spp_block.conv0, dict(state_dict_list[:6]))
    load_conv_weight(spp_block.conv1, dict(state_dict_list[6:12]))
    load_conv_weight(spp_block.conv2, dict(state_dict_list[12:18]))
    load_conv_weight(spp_block.conv3, dict(state_dict_list[18:24]))
    load_conv_weight(spp_block.conv4, dict(state_dict_list[24:30]))
    load_conv_weight(spp_block.conv5, dict(state_dict_list[30:]))


def load_csp_up_weights(csp_block, state_dict):
    state_dict_list = list(state_dict.items()) # noqa

    load_conv_weight(csp_block.conv0, dict(state_dict_list[:6]))
    load_conv_weight(csp_block.conv1, dict(state_dict_list[6:12]))
    load_conv_weight(csp_block.conv2, dict(state_dict_list[12:18]))
    load_conv_weight(csp_block.conv3, dict(state_dict_list[18:24]))
    load_conv_weight(csp_block.conv4, dict(state_dict_list[24:30]))
    load_conv_weight(csp_block.conv5, dict(state_dict_list[30:36]))
    load_conv_weight(csp_block.conv6, dict(state_dict_list[36:]))


def load_neck_weights(neck, state_dict):
    state_dict_list = list(state_dict.items())  # noqa

    spp_state_dict = dict(state_dict_list[:36])
    up1_state_dict = dict(state_dict_list[36:78])
    up2_state_dict = dict(state_dict_list[78:])

    load_spp_weights(neck.spp, spp_state_dict)
    load_csp_up_weights(neck.up1, up1_state_dict)
    load_csp_up_weights(neck.up2, up2_state_dict)
