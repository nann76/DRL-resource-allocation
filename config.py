



model_paras={

    # "dim_ope_in": env_paras['dim_feature_operation'],
    # "dim_ope_out": 16-8,
    # "dim_agent_in": env_paras['dim_feature_agent'],
    # "dim_agent_out": 16-8,
    # # "dim_edge": 1,
    # "mlp_hidden_dim": 128,
    #
    # "num_heads": [1, 1],
    # "dropout": 0.0,

    "num_layers_actor": 3,
    "num_layers_critic": 3,
    "hidden_dim_actor": 128,
    "hidden_dim_critic": 128,
    "output_dim_actor": 1,
    "output_dim_critic": 1,

    # "device":'cuda:0',
    # "device":'cpu',

}

train_paras = {

    "device":'cuda:0',

    "batch_size_sample": 20 ,  # 训练时，一次sample的案例数。
    "repeat_time_sample": 1,  # 在一次iter中对一个insatnce的重复探索次数

    "batch_size_validate": 100,  # 验证集的案例数
    "max_iterations": 1000,     # 迭代次数
    "policy_update_timestep": 1,    # 策略更新的时间步
    "validate_timestep": 10,        # validate的时间步，同时根据validate的结果决定是否保存当前模型
    "train_env_update_timestep": 20,   # 用于采样的env更新的时间步

    "maxlen_best_model": 5,
    "update_mini_batch_size": 2**10,  # 模型更新时，一次送入的batch

    # "gamma": 0.99,
    "gamma": 1.0,
    "lr": 2e-4,
    "betas": [0.9, 0.999],
    "K_epochs": 3,
    "eps_clip": 0.2,
    "A_coeff": 1.0,
    "vf_coeff": 0.5,
    # "A_coeff": 2.0,
    # "vf_coeff": 1.0,
    "entropy_coeff": 0.01,


}
