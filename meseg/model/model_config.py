import ml_collections


def UNETR():
    config = {}
    config["in_channels"] = 1
    config["out_channels"] = 14
    config["img_size"] = (96,96,96)
    config["feature_size"] = 16
    config["hidden_size"] = 768
    config["mlp_dim"] = 3072
    config["num_heads"] = 12
    config["pos_embed"] = "perceptron"
    config["norm_name"] = "instance"
    config["res_block"] = True
    config["dropout_rate"] = 0.0
    return config