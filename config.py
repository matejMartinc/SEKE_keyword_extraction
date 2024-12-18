import os





TrainConfig = {
    "pretrained_model": "microsoft/deberta-v3-base",
    "num_labels": 3,
    "lr": 2e-4,
    "max_length": 256,
    "batch_size": 8,
    "num_workers": os.cpu_count(),
    "max_epochs": 20,
    "debug_mode_sample": None,
    "max_time": None,
    "min_delta": 0.005,
    "patience": 5,
    "seed": 42,
    "model_type": "deberta-v2",
    "attention_probs_dropout_prob": 0.1,
    "hidden_dropout_prob": 0.1,
    "hidden_act": "gelu",
    "hidden_size": 768,
    "initializer_range": 0.02,
    "intermediate_size": 3072,
    "max_position_embeddings": 256,
    "relative_attention": True,
    "position_buckets": 256,
    "norm_rel_ebd": "layer_norm",
    "share_att_key": True,
    "pos_att_type": "p2c|c2p",
    "layer_norm_eps": 1e-7,
    "max_relative_positions": -1,
    "position_biased_input": False,
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
    "type_vocab_size": 0,
    "vocab_size": 128100,
    "use_return_dict": True,
    "top_k": 2,
    #If more than 4, you should add additional layers to the LORA config
    "num_experts": 4,
}

TestConfig = {
    "pretrained_model": "microsoft/deberta-v3-base",
    "num_labels": 3,
    "max_length": 256,
    "batch_size": 4,
    "lr": 2e-4,
    "num_workers": os.cpu_count(),
    "debug_mode_sample": None,
    "max_time": None,
    "seed": 42,
    "model_type": "deberta-v2",
    "attention_probs_dropout_prob": 0.1,
    "hidden_dropout_prob": 0.1,
    "hidden_act": "gelu",
    "hidden_size": 768,
    "initializer_range": 0.02,
    "intermediate_size": 3072,
    "max_position_embeddings": 256,
    "relative_attention": True,
    "position_buckets": 256,
    "norm_rel_ebd": "layer_norm",
    "share_att_key": True,
    "pos_att_type": "p2c|c2p",
    "layer_norm_eps": 1e-7,
    "max_relative_positions": -1,
    "position_biased_input": False,
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
    "type_vocab_size": 0,
    "vocab_size": 128100,
    "use_return_dict": True,
    "top_k": 2,
    #If more than 4, you should add additional layers to the LORA config
    "num_experts": 4,
}



