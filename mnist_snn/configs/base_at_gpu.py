from mnist_snn.utils import Config

model_config = Config(
    tau = 2.0, 
    n_layer = 5,
    n_neuron = [50, 40, 30, 20, 10],
    has_I = True,
)

global_config = Config(
    T = 100,
    device = 'gpu',
    batch = 64,
    epochs = 15,
    lr = 1e-3,
    toggle_global_balance_reg_w_l2 = True,
    global_balance_reg_scale = 1e-4,
    kappa = 1.0,
    toggle_detailed_balance_reg_l1_l2_spike = False,
    detailed_balance_l1_reg_sp_scale = 1e-5,
    detailed_balance_l2_reg_sp_scale = 1e-6,
    toggle_detailed_balance_reg_input = True,
    detailed_balance_l1_reg_inp_scale = 1e-5,
)
