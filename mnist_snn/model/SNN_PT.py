import brainpy as bp
import brainpy.math as bm
from wm_snn.model.GIF import GIF as GIFa
from brainpy.neurons import GIF as GIFb


class SNN(bp.DynamicalSystem):
    def __init__(self, config):
        super().__init__()

        assert len(config.n_neuron) == config.n_layer
        self.config = config
        self.n_layer = config.n_layer
        self.n_neuron = config.n_neuron
        self.tau = config.tau
        self.encoder = bp.encoding.PoissonEncoder(min_val=0., max_val=1.)

        for i in range(config.n_layer):
            if i == 0:
                self.init_connection(layer_index=i, in_num=28*28, out_num=config.n_neuron[i], 
                                     balance_weights=config.balance_weights, suffix='ff')
            else:
                # feedforward connection from layer i-1 to layer i
                self.init_connection(layer_index=i, in_num=config.n_neuron[i-1], out_num=config.n_neuron[i], 
                                     balance_weights=config.balance_weights, suffix='ff')

            # recurrent connection in layer i from E to E
            self.init_connection(layer_index=i, in_num=config.n_neuron[i], out_num=config.n_neuron[i], 
                                 balance_weights=config.balance_weights, suffix='ee')
            # recurrent connection in layer i from E to I
            self.init_connection(layer_index=i, in_num=config.n_neuron[i], out_num=config.n_neuron[i], 
                                 balance_weights=config.balance_weights, suffix='ei')
            # recurrent connection in layer i from I to E
            self.init_connection(layer_index=i, in_num=config.n_neuron[i], out_num=config.n_neuron[i], 
                                 balance_weights=config.balance_weights, suffix='ie')
            # recurrent connection in layer i from I to I
            self.init_connection(layer_index=i, in_num=config.n_neuron[i], out_num=config.n_neuron[i], 
                                 balance_weights=config.balance_weights, suffix='ii')
                
            # E neurons in layer i
            E_neuron = self.select_neuron(i, name=f'layer{i}_E', E_or_I='E')
            setattr(self, f'layer{i}_neu_e', E_neuron)

            # I neurons in layer i
            I_neuron = self.select_neuron(i, name=f'layer{i}_I', E_or_I='I')
            setattr(self, f'layer{i}_neu_i', I_neuron)

            # exponential kernel for synaptic currents
            self.init_exp_kernel(layer_index=i, fast_tau=2.0, slow_tau=10.0, suffix='neu_e_epsc')
            self.init_exp_kernel(layer_index=i, fast_tau=2.0, slow_tau=2.0, suffix='neu_e_ipsc')
            self.init_exp_kernel(layer_index=i, fast_tau=2.0, slow_tau=10.0, suffix='neu_i_epsc')
            self.init_exp_kernel(layer_index=i, fast_tau=2.0, slow_tau=2.0, suffix='neu_i_ipsc')

            
            # input current for visualization
            setattr(self, f'layer{i}_neu_e_inp_fast_e', bm.Variable(bm.zeros(config.n_neuron[i])))
            setattr(self, f'layer{i}_neu_e_inp_slow_e', bm.Variable(bm.zeros(config.n_neuron[i])))
            setattr(self, f'layer{i}_neu_e_inp_fast_i', bm.Variable(bm.zeros(config.n_neuron[i])))
            setattr(self, f'layer{i}_neu_e_inp_slow_i', bm.Variable(bm.zeros(config.n_neuron[i])))
            setattr(self, f'layer{i}_neu_i_inp_fast_e', bm.Variable(bm.zeros(config.n_neuron[i])))
            setattr(self, f'layer{i}_neu_i_inp_slow_e', bm.Variable(bm.zeros(config.n_neuron[i])))
            setattr(self, f'layer{i}_neu_i_inp_fast_i', bm.Variable(bm.zeros(config.n_neuron[i])))
            setattr(self, f'layer{i}_neu_i_inp_slow_i', bm.Variable(bm.zeros(config.n_neuron[i])))


            # patterns for weights and neurons
            self.w_pattern = [r"layer{}_ff", r"layer{}_ee", r"layer{}_ei", r"layer{}_ie", r"layer{}_ii"]
            self.neu_pattern = [r"layer{}_neu_e", r"layer{}_neu_e"]
            self.mask_pattern = {
                r"layer{}_ff": lambda x: x < 0, 
                r"layer{}_ee": lambda x: x < 0,
                r"layer{}_ei": lambda x: x < 0, 
                r"layer{}_ie": lambda x: x > 0, 
                r"layer{}_ii": lambda x: x > 0
            }
            

    def init_connection(self, layer_index: int, in_num: int, out_num: int, 
                        suffix: str|None = None, balance_weights=True):
        if suffix is None:
            suffix = ''
        else:
            suffix = '_' + suffix

        setattr(self, 
                f'layer{layer_index}{suffix}', 
                bp.layers.Dense(
                    in_num, 
                    out_num, 
                    W_initializer=bp.initialize.Uniform(min_val=0, max_val=1.0/in_num), 
                    b_initializer=None, 
                    name=f'layer{layer_index}{suffix}', 
                )
        )
    
        if balance_weights:
            setattr(self,
                    f'layer{layer_index}_balance{suffix}',
                    bp.dnn.JitFPHomoLinear(
                        in_num, 
                        out_num, 
                        prob=0.1,
                        weight=1/bm.sqrt(in_num),
                        seed=0,
                        mode=bm.BatchingMode()
                    )
            )
            
    def init_exp_kernel(self, layer_index: int, fast_tau: int|None =None, 
                        slow_tau: int|None = None, suffix: str|None = None):
        if suffix is None:
            suffix = ''
        else:
            suffix = '_' + suffix

        if fast_tau is not None:
            setattr(self,
                    f'layer{layer_index}_fast_exp{suffix}',
                    bp.dyn.Expon(
                        size=self.config.n_neuron[layer_index],
                        name=f'layer{layer_index}_fast_exp{suffix}',
                        tau=fast_tau,
                    )
            )

        if slow_tau is not None:
            setattr(self,
                    f'layer{layer_index}_slow_exp{suffix}',
                    bp.dyn.Expon(
                        size=self.config.n_neuron[layer_index],
                        name=f'layer{layer_index}_slow_exp{suffix}',
                        tau=slow_tau,
                    )
            )
        
    def run_dynamics(self, layer_index: int, inp: bm.Variable, slow_or_fast: str):
        if slow_or_fast not in ['slow', 'fast']:
            raise ValueError(f'slow_or_fast should be one of [slow, fast], but got {slow_or_fast}')
        if slow_or_fast == 'fast':
            wk = 'balance_'
        else:
            wk = ''

        i = layer_index
        expk_ee = getattr(self, f'layer{i}_{slow_or_fast}_exp_neu_e_epsc')
        expk_ie = getattr(self, f'layer{i}_{slow_or_fast}_exp_neu_e_ipsc')
        expk_ei = getattr(self, f'layer{i}_{slow_or_fast}_exp_neu_i_epsc')
        expk_ii = getattr(self, f'layer{i}_{slow_or_fast}_exp_neu_i_ipsc')
        # feedforward input
        conn_ff = getattr(self, f'layer{i}_{wk}ff')   # feedforward connection from layer i-1 to layer i
        ff_sp = conn_ff(inp)
        # excitatory input
        conn_ee = getattr(self, f'layer{i}_{wk}ee')   # recurrent connection in layer i from E to E
        ee_sp = conn_ee(getattr(self, f'layer{i}_neu_e').spike)
        current_ee = expk_ee(ff_sp + ee_sp)
        # inhibitory input
        conn_ie = getattr(self, f'layer{i}_{wk}ie')   # recurrent input from I to E neurons
        ie_sp = conn_ie(getattr(self, f'layer{i}_neu_i').spike)   
        current_ie = expk_ie(ie_sp)
        # recurrent input to inhibitory neurons
        conn_ei = getattr(self, f'layer{i}_{wk}ei')   # recurrent input from E to I neurons
        conn_ii = getattr(self, f'layer{i}_{wk}ii')   # recurrent input from I to I neurons
        ei_sp = conn_ei(getattr(self, f'layer{i}_neu_e').spike)
        ii_sp = conn_ii(getattr(self, f'layer{i}_neu_i').spike)
        current_ei = expk_ei(ei_sp)
        current_ii = expk_ii(ii_sp)

        return current_ee, current_ei, current_ie, current_ii
        

    def update(self, p, x):
        # mask weights
        for i in range(self.n_layer):
            for wp in self.w_pattern:
                w = getattr(self, wp.format(i)).W
                mask_f = self.mask_pattern[wp]
                w.value = bm.where(mask_f(w), 0, w)
        # update
        bp.share.save(t=p)
        inp = x
        for i in range(self.n_layer):
            slow_ee, slow_ei, slow_ie, slow_ii = self.run_dynamics(layer_index=i, inp=inp, slow_or_fast='slow')
            if self.config.balance_weights:
                fast_ee, fast_ei, fast_ie, fast_ii = self.run_dynamics(layer_index=i, inp=inp, slow_or_fast='fast')
            else:
                fast_ee, fast_ei, fast_ie, fast_ii = [bm.zeros([x.shape[0], self.config.n_neuron[i]])] * 4
            SI = self.config.SI_k * fast_ie * (fast_ee + slow_ee)

            # update neuron's dynamics
            e_neu = getattr(self, f'layer{i}_neu_e')
            e_neu.update(slow_ee + fast_ee + slow_ie + fast_ie + SI)
            i_neu = getattr(self, f'layer{i}_neu_i')
            i_neu.update(slow_ei + fast_ei + slow_ii + fast_ii)

            inp = e_neu.spike

            # save input current for visualization
            getattr(self, f'layer{i}_neu_e_inp_fast_e').value = fast_ee[0]
            getattr(self, f'layer{i}_neu_e_inp_slow_e').value = slow_ee[0]
            getattr(self, f'layer{i}_neu_i_inp_fast_e').value = fast_ei[0]
            getattr(self, f'layer{i}_neu_i_inp_slow_e').value = slow_ei[0]
            getattr(self, f'layer{i}_neu_e_inp_fast_i').value = fast_ie[0] + SI[0]
            getattr(self, f'layer{i}_neu_e_inp_slow_i').value = slow_ie[0]
            getattr(self, f'layer{i}_neu_i_inp_fast_i').value = fast_ii[0]
            getattr(self, f'layer{i}_neu_i_inp_slow_i').value = slow_ii[0]

        outputs = {
            "e_neu":{
                "spike": [getattr(self, f'layer{i}_neu_e').spike.value for i in range(self.n_layer)],
                "V": [getattr(self, f'layer{i}_neu_e').V.value for i in range(self.n_layer)],
                "inp": [getattr(self, f'layer{i}_neu_e').input.value for i in range(self.n_layer)],
                },
            "i_neu":{
                "spike": [getattr(self, f'layer{i}_neu_i').spike.value for i in range(self.n_layer)],
                "V": [getattr(self, f'layer{i}_neu_i').V.value for i in range(self.n_layer)],
                "inp": [getattr(self, f'layer{i}_neu_i').input.value for i in range(self.n_layer)],
                }
            }

        return outputs
    
    def select_neuron(self, i, name, E_or_I):
        if E_or_I == "E":
            A1 = bm.zeros(self.config.n_neuron[i])
        elif E_or_I == "I":
            A1 = bm.ones(self.config.n_neuron[i])

        gif_pars = dict(
            Ath=1, A2=-0.6, adaptive_th=False, tau_I1=10., v_scale_var=True,
            tau_I2=bm.random.uniform(100, 3000, self.config.n_neuron[i]), A1=A1
        )

        if self.config.neuron_type == "LIF":
            neuron = bp.neurons.LIF(
                self.config.n_neuron[i],
                V_rest=0.,
                V_reset=0.,
                V_th=1.,
                tau=self.config.tau,
                spike_fun=bm.surrogate.arctan,
                name=name
            )
        elif self.config.neuron_type == "GIFa":
            neuron = GIFa(
                self.config.n_neuron[i], 
                V_rest=0., 
                V_th_inf=1., 
                tau=self.config.tau, 
                spike_fun=bm.surrogate.arctan,
                V_initializer=bp.init.ZeroInit(),
                Vth_initializer=bp.init.OneInit(1.),
                name=name,
                **gif_pars,
            )
        elif self.config.neuron_type == "GIFb":
            neuron = GIFb(
                self.config.n_neuron[i], 
                V_rest=0., 
                V_reset=0.,
                V_th_inf=1., 
                tau=self.config.tau, 
                spike_fun=bm.surrogate.arctan,
                name=name,
            )
        return neuron
    