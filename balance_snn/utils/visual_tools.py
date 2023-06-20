import matplotlib.pyplot as plt
import numpy as np
import brainpy as bp
import brainpy.math as bm
import os


def epoch_visual(model, inputs, epoch_id, save_dir):
    show_and_save_weights(model, os.path.join(save_dir, f'weights_{epoch_id}.png'))
    show_and_save_activity(model, inputs, os.path.join(save_dir, f'activity_{epoch_id}.png'))
    show_and_save_current(model, inputs, os.path.join(save_dir, f'currents_{epoch_id}.png'))


def show_and_save_weights(model, file_name):
    n_layer = model.n_layer
    fig = plt.figure(figsize=(11, 3*n_layer))

    for i in range(n_layer):
        ff_w = getattr(model, f'layer{i}_ff').W.value
        ee_w = getattr(model, f'layer{i}_ee').W.value
        try:
            ei_w = getattr(model, f'layer{i}_ei').W.value
            ie_w = getattr(model, f'layer{i}_ie').W.value
        except:
            pass

        ax1 = fig.add_subplot(n_layer, 4, 1+4*i)
        ff_fig = ax1.imshow(ff_w, aspect='auto')
        ax1.set_title("ff weights")
        plt.colorbar(ff_fig, ax=ax1)
        try:
            ax2 = fig.add_subplot(n_layer, 4, 2+4*i)
            ei_fig = ax2.imshow(ei_w, aspect='auto')
            ax2.set_title("ei weights")
            plt.colorbar(ei_fig, ax=ax2)
            ax3 = fig.add_subplot(n_layer, 4, 3+4*i)
            ie_fig = ax3.imshow(ie_w, aspect='auto')
            ax3.set_title("ie weights")
            plt.colorbar(ie_fig, ax=ax3)
        except:
            pass

        ax4 = fig.add_subplot(n_layer, 4, 4+4*i)
        ee_fig = ax4.imshow(ee_w, aspect='auto')
        ax4.set_title("ee weights")
        plt.colorbar(ee_fig, ax=ax4)

    plt.savefig(file_name)
    plt.close()


def show_and_save_activity(model, inputs, file_name):
    n_layer = model.n_layer
    T = inputs.shape[0]
    fig = plt.figure(figsize=(6, 3*n_layer))

    mon_vars = [f'layer{i}_e_neu.spike' for i in range(n_layer)]
    runner = bp.DSRunner(model, data_first_axis='T', monitors=mon_vars, progress_bar=False)
    runner.predict(inputs=inputs, reset_state=True)

    for i in range(n_layer):
        ax = fig.add_subplot(n_layer, 1, 1+i)
        bp.visualize.raster_plot(ts=bm.arange(0, T), sp_matrix=runner.mon[f'layer{i}_e_neu.spike'][:,0], ax=ax)
        plt.xlim([0, T])

    plt.savefig(file_name)
    plt.close()


def show_and_save_current(model, inputs, file_name):
    n_layer = model.n_layer
    fig = plt.figure(figsize=(6, 3*n_layer))

    mon_vars = [
        [f'layer{i}_inp_ff' for i in range(n_layer)],
        [f'layer{i}_inp_ee' for i in range(n_layer)],
        [f'layer{i}_inp_ei' for i in range(n_layer)],
        [f'layer{i}_inp_ie' for i in range(n_layer)],
        [f'layer{i}_inp_ii' for i in range(n_layer)],
    ]
    mon_vars = sum(mon_vars, [])
    runner = bp.DSRunner(model, data_first_axis='T', monitors=mon_vars, progress_bar=False)
    runner.predict(inputs=inputs, reset_state=True)
    
    for i in range(n_layer):
        # E neuron
        ax1 = fig.add_subplot(n_layer, 2, 1+i*2)
        total_e = np.sum(runner.mon[f'layer{i}_inp_ff'], axis=0) \
            + np.sum(runner.mon[f'layer{i}_inp_ee'], axis=0)
        total_i = np.sum(runner.mon[f'layer{i}_inp_ie'], axis=0)
        ax1.plot(total_e, label='E', color='r')
        ax1.plot(total_i, label='I', color='b')
        ax1.plot(total_e+total_i, label='total', color='k')
        # I neuron
        ax2 = fig.add_subplot(n_layer, 2, 2+i*2)
        total_e = np.sum(runner.mon[f'layer{i}_inp_ei'], axis=0)
        total_i = np.sum(runner.mon[f'layer{i}_inp_ii'], axis=0)
        ax2.plot(total_e, label='E', color='r', linestyle='--')
        ax2.plot(total_i, label='I', color='b', linestyle='--')
        ax2.plot(total_e+total_i, label='total', color='k', linestyle='--')       

    plt.savefig(file_name)
    plt.close()
