from mnist_snn.utils import Config

model_config = Config(
    tau = 2.0, 
    n_layer = 5,
    neuron_type = "LIF",    # "LIF" or "GIFa" (wm GIF), "GIFb" (bp.GIF)
    n_neuron = [50, 40, 30, 20, 10],
    # n_rate_neuron = [100, 10],
    SI_k = 0.0,
    balance_weights = True,
)

global_config = Config(
    T = 100,
    device = 'gpu',
    batch = 64,
    epochs = 15,
    lr = 1e-3,
    toggle_global_balance_reg_w_l2 = False,
    toggle_detailed_balance_reg_l1_l2_spike = False,
    toggle_detailed_balance_reg_input = False,
)
