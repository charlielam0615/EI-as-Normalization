import matplotlib.pyplot as plt
import numpy as np
import brainpy as bp
import brainpy.math as bm
import os


def epoch_visual(model, trainer, inputs, global_config, epoch_id, save_dir):
    show_and_save_weights(model, os.path.join(save_dir, f'weights_{epoch_id}.png'))
    show_and_save_activity(model, inputs, global_config, os.path.join(save_dir, f'activity_{epoch_id}.png'))
    show_and_save_current(model, inputs, global_config, os.path.join(save_dir, f'currents_{epoch_id}.png'))
    show_and_save_wl2(model, global_config, os.path.join(save_dir, f'wl2_{epoch_id}.png'))
    show_and_save_loss_reg(trainer, inputs, os.path.join(save_dir, f'loss_comp_{epoch_id}.png'))
    

def show_and_save_loss_reg(trainer, inputs, filename):
    inputs, label = inputs['data'], inputs['label']
    _, linfo = trainer.calculate_loss(inputs, label)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    # loss_comp is a dict
    keys = ['ce_loss', 'global_balance_reg', 'detailed_balance_l1_reg_sp', 
            'detailed_balance_l2_reg_sp', 'detailed_balance_reg_inp', 'loss_reg_v']
    key_abbrev = ['ce', 'gb', 'dbsp1', 'dbsp2', 'dbinp', 'regv']
    # bar plot of loss components
    ax.bar(key_abbrev, [linfo[k] for k in keys])
    plt.savefig(filename)
    plt.close()


def show_and_save_wl2(model, global_config, filename):
    w_l2 = []
    w_category = {'ff':[], 'ee':[], 'ei':[], 'ie':[], 'ii':[]}
    for i in range(model.n_layer):
        for wp in model.w_pattern:
            w = getattr(model, wp.format(i)).W
            l2_value = bm.sqrt(bm.sum(bm.square(w))).item()
            w_l2.append(l2_value)
            w_category[wp[-2:]].append(l2_value)

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(w_l2)
    ax1.plot([0, len(w_l2)], [global_config.kappa, global_config.kappa], 'r--')
    ax1.set_title('L2 norm of weights')

    ax2 = fig.add_subplot(1, 2, 2)
    # plot bar plot of mean L2 norm of weights in w_category with error bars
    ax2.bar(w_category.keys(), [np.mean(w_category[k]) for k in w_category.keys()])
    ax2.errorbar(w_category.keys(), [np.mean(w_category[k]) for k in w_category.keys()],
                    yerr=[np.std(w_category[k]) for k in w_category.keys()], fmt='none', ecolor='r')
    ax2.plot([0, len(w_category.keys())], [global_config.kappa, global_config.kappa], 'r--')
    ax2.set_title('In each category')
    plt.savefig(filename)
    

def save_loss_and_acc_figure(trainer, filename):
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(trainer.hist_train_loss)
    ax1.set_title('train loss')
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(trainer.hist_train_acc)
    ax2.set_title('train acc')
    ax2.set_ylim(0, 1)
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(trainer.hist_val_loss)
    ax3.set_title('val loss')
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(trainer.hist_val_acc)
    ax4.set_ylim(0, 1)
    ax4.set_title('val acc')
    plt.tight_layout()
    plt.savefig(filename)


def show_and_save_weights(model, file_name):
    n_layer = model.n_layer
    fig = plt.figure(figsize=(16, 3*n_layer))

    for i in range(n_layer):
        ff_w = getattr(model, f'layer{i}_ff').W.value
        ee_w = getattr(model, f'layer{i}_ee').W.value
        try:
            ei_w = getattr(model, f'layer{i}_ei').W.value
            ie_w = getattr(model, f'layer{i}_ie').W.value
        except:
            pass

        ax1 = fig.add_subplot(n_layer, 4, 1+4*i)
        ff_fig = ax1.imshow(ff_w, aspect='auto', cmap='YlOrBr')
        ax1.set_title("ff weights")
        plt.colorbar(ff_fig, ax=ax1)
        try:
            ax2 = fig.add_subplot(n_layer, 4, 2+4*i)
            ei_fig = ax2.imshow(ei_w, aspect='auto', cmap='YlOrBr')
            ax2.set_title("ei weights")
            plt.colorbar(ei_fig, ax=ax2)
            ax3 = fig.add_subplot(n_layer, 4, 3+4*i)
            ie_fig = ax3.imshow(ie_w, aspect='auto', cmap='YlGnBu_r')
            ax3.set_title("ie weights")
            plt.colorbar(ie_fig, ax=ax3)
        except:
            pass

        ax4 = fig.add_subplot(n_layer, 4, 4+4*i)
        ee_fig = ax4.imshow(ee_w, aspect='auto', cmap='YlOrBr')
        ax4.set_title("ee weights")
        plt.colorbar(ee_fig, ax=ax4)

    plt.savefig(file_name)
    plt.close()


def show_and_save_activity(model, inputs, global_config, file_name):
    inputs = inputs['data']
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


def show_and_save_current(model, inputs, global_config, file_name):
    inputs = inputs['data']
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
