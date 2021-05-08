from pth2pth.load_conv import load_conv_weight


def load_csp_down_weights(csp_block, state_dict):
    state_dict_list = list(state_dict.items())  # noqa

    load_conv_weight(csp_block.conv0, dict(state_dict_list[:6]))
    load_conv_weight(csp_block.conv1, dict(state_dict_list[6:12]))
    load_conv_weight(csp_block.conv2, dict(state_dict_list[12:18]))
    load_conv_weight(csp_block.conv3, dict(state_dict_list[18:24]))
    load_conv_weight(csp_block.conv4, dict(state_dict_list[24:30]))
    load_conv_weight(csp_block.conv5, dict(state_dict_list[30:]))


def load_head_weights(head, state_dict):
    state_dict_list = list(state_dict.items())  # noqa

    conv1_state_dict = dict(state_dict_list[:6])
    conv2_state_dict = dict(state_dict_list[6:8])
    down1_state_dict = dict(state_dict_list[8:44])

    conv3_state_dict = dict(state_dict_list[44:50])
    conv4_state_dict = dict(state_dict_list[50:52])
    down2_state_dict = dict(state_dict_list[52:88])

    conv5_state_dict = dict(state_dict_list[88:94])
    conv6_state_dict = dict(state_dict_list[94:])

    load_csp_down_weights(head.down1, down1_state_dict)
    load_csp_down_weights(head.down2, down2_state_dict)

    load_conv_weight(head.conv1, conv1_state_dict)
    load_conv_weight(head.conv2, conv2_state_dict)
    load_conv_weight(head.conv3, conv3_state_dict)
    load_conv_weight(head.conv4, conv4_state_dict)
    load_conv_weight(head.conv5, conv5_state_dict)
    load_conv_weight(head.conv6, conv6_state_dict)
