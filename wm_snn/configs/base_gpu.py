from wm_snn.utils import Config

model_config = Config(
    tau = 20.0, 
    n_layer = 3,
    n_neuron = [80, 60, 2],
    neuron_type = "GIFa",    # "LIF" or "GIFa" (wm GIF), "GIFb" (bp.GIF)
    has_I = True,
)

global_config = Config(
    T = 2500,
    dt = 1.0,
    device = 'gpu',
    batch = 64,
    epochs = 10,
    lr = 1e-4,
    toggle_global_balance_reg_w_l2 = True,
    global_balance_reg_scale = 1e-2,
    kappa = 1.0,
    toggle_detailed_balance_reg_l1_l2_spike = False,
    detailed_balance_l1_reg_sp_scale = 1e-6,
    detailed_balance_l2_reg_sp_scale = 1e-9,
    toggle_detailed_balance_reg_input = True,
    detailed_balance_l1_reg_inp_scale = 1e-6,
    toggle_membrane_reg = False,
    membrane_reg_scale = 1.,
)
