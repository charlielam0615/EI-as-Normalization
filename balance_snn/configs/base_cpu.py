from balance_snn.utils import Config

model_config = Config(
    tau = 2.0, 
    n_layer = 5,
    n_neuron = [50, 40, 30, 20, 10],
    has_I = True,
)

global_config = Config(
    T = 100,
    device = 'cpu',
    batch = 64,
    epochs = 1,
    lr = 1e-3,
    toggle_global_balance_reg = True,
    global_balance_reg_scale = 1e-2,
    toggle_detailed_balance_reg = True,
    detailed_balance_l1_reg_scale = 1e-2,
    detailed_balance_l2_reg_scale = 1e-2,
    kappa = 1.0,
)
