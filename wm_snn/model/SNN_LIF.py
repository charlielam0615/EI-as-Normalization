import brainpy as bp
import brainpy.math as bm


class SNN_LIF(bp.DynamicalSystem):
    def __init__(self, config):
        super().__init__()

        assert len(config.n_neuron) == config.n_layer
        self.config = config
        self.has_I = config.has_I
        self.n_layer = config.n_layer
        self.n_neuron = config.n_neuron
        self.tau = config.tau

        for i in range(config.n_layer):
            if i == 0:
                setattr(self, 
                    f'layer{i}_ff', 
                    bp.layers.Dense(
                        100, 
                        config.n_neuron[i], 
                        W_initializer=bp.initialize.Uniform(min_val=0, max_val=1/100), 
                        b_initializer=None, 
                        name=f'layer{i}_ff', 
                    )
                )
            else:
                # feedforward connection from layer i-1 to layer i
                setattr(self, 
                        f'layer{i}_ff', 
                        bp.layers.Dense(
                            config.n_neuron[i-1], 
                            config.n_neuron[i], 
                            W_initializer=bp.initialize.Uniform(min_val=0, max_val=1/config.n_neuron[i-1]), 
                            b_initializer=None,
                            name=f'layer{i}_ff',
                        )
                )
            # recurrent connection in layer i from E to E
            setattr(self,
                    f'layer{i}_ee',
                    bp.layers.Dense(
                        config.n_neuron[i], 
                        config.n_neuron[i], 
                        W_initializer=bp.initialize.Uniform(min_val=0, max_val=1/config.n_neuron[i]), 
                        b_initializer=None,
                        name=f'layer{i}_ee',
                    )
            )
            # recurrent connection in layer i from E to I
            if config.has_I:
                setattr(self,
                        f'layer{i}_ei',
                        bp.layers.Dense(
                            config.n_neuron[i], 
                            config.n_neuron[i], 
                            W_initializer=bp.initialize.Uniform(min_val=0, max_val=1/config.n_neuron[i]), 
                            b_initializer=None,
                            name=f'layer{i}_ei',
                        )
                )
                # recurrent connection in layer i from I to E
                setattr(self,
                        f'layer{i}_ie',
                        bp.layers.Dense(
                            config.n_neuron[i], 
                            config.n_neuron[i], 
                            W_initializer=bp.initialize.Uniform(min_val=-1/config.n_neuron[i], max_val=0), 
                            b_initializer=None,
                            name=f'layer{i}_ie',
                        )
                )
                # recurrent connection in layer i from I to I
                setattr(self,
                        f'layer{i}_ii',
                        bp.layers.Dense(
                            config.n_neuron[i], 
                            config.n_neuron[i], 
                            W_initializer=bp.initialize.Uniform(min_val=-1/config.n_neuron[i], max_val=0), 
                            b_initializer=None,
                            name=f'layer{i}_ii',
                        )
                )

                # E neurons in layer i
                setattr(self,
                        f'layer{i}_e_neu',
                        bp.neurons.LIF(
                            config.n_neuron[i], 
                            V_rest=0., 
                            V_reset=0., 
                            V_th=1., 
                            tau=config.tau, 
                            spike_fun=bm.surrogate.arctan,
                            name=f'layer{i}_E',
                        )
                )

            if config.has_I:
                # I neurons in layer i
                setattr(self,
                        f'layer{i}_i_neu',
                        bp.neurons.LIF(
                            config.n_neuron[i], 
                            V_rest=0., 
                            V_reset=0., 
                            V_th=1., 
                            tau=config.tau, 
                            spike_fun=bm.surrogate.arctan,
                            name=f'layer{i}_I',
                        )
                )
            
            # input current
            setattr(self,
                    f'layer{i}_inp_ff',
                    bm.Variable(bm.zeros(config.n_neuron[i])),
            )

            setattr(self,
                    f'layer{i}_inp_ee',
                    bm.Variable(bm.zeros(config.n_neuron[i])),
            )

            setattr(self,
                    f'layer{i}_inp_ie',
                    bm.Variable(bm.zeros(config.n_neuron[i])),
            )

            setattr(self,
                    f'layer{i}_inp_ii',
                    bm.Variable(bm.zeros(config.n_neuron[i])),
            )

            setattr(self,
                    f'layer{i}_inp_ei',
                    bm.Variable(bm.zeros(config.n_neuron[i])),
            )

            # patterns
            if config.has_I:
                self.w_pattern = [r"layer{}_ff", r"layer{}_ee", r"layer{}_ei", r"layer{}_ie", r"layer{}_ii"]
                self.neu_pattern = [r"layer{}_e_neu", r"layer{}_i_neu"]
                self.mask_pattern = {
                    r"layer{}_ff": lambda x: x < 0, 
                    r"layer{}_ee": lambda x: x < 0,
                    r"layer{}_ei": lambda x: x < 0, 
                    r"layer{}_ie": lambda x: x > 0, 
                    r"layer{}_ii": lambda x: x > 0
                }
            else:
                self.w_pattern = [r"layer{}_ff", r"layer{}_ee"]
                self.neu_pattern = [r"layer{}_e_neu"]
                self.mask_pattern = {r"layer{}_ff": lambda x: x < 0}
        
    def update(self, p, x):
        # mask weights
        for i in range(self.n_layer):
            for wp in self.w_pattern:
                w = getattr(self, wp.format(i)).W
                mask_f = self.mask_pattern[wp]
                w.value = bm.where(mask_f(w), 0, w)
        # update
        inp = x
        for i in range(self.n_layer):
            conn_ff = getattr(self, f'layer{i}_ff')   # feedforward connection from layer i-1 to layer i
            ff_inp = conn_ff(inp)
            getattr(self, f'layer{i}_inp_ff').value = ff_inp[0]   # only save the first batch item
            conn_ee = getattr(self, f'layer{i}_ee')   # recurrent connection in layer i from E to E
            r_ee = conn_ee(getattr(self, f'layer{i}_e_neu').spike.value)
            getattr(self, f'layer{i}_inp_ee').value = r_ee[0]
            if self.has_I:
                conn_ie = getattr(self, f'layer{i}_ie')   # recurrent input from I to E neurons
                r_ie = conn_ie(getattr(self, f'layer{i}_i_neu').spike.value)   
                getattr(self, f'layer{i}_inp_ie').value = r_ie[0]
                conn_ei = getattr(self, f'layer{i}_ei')   # recurrent input from E to I neurons
                conn_ii = getattr(self, f'layer{i}_ii')   # recurrent input from I to I neurons
                r_ei = conn_ei(getattr(self, f'layer{i}_e_neu').spike.value)
                getattr(self, f'layer{i}_inp_ei').value = r_ei[0]
                r_ii = conn_ii(getattr(self, f'layer{i}_i_neu').spike.value)
                getattr(self, f'layer{i}_inp_ii').value = r_ii[0]
            else:
                r_ie = 0.
            
            bp.share.save(t=p)
            e_neu = getattr(self, f'layer{i}_e_neu')
            e_neu.update(ff_inp + r_ie + r_ee)
            if self.has_I:
                i_neu = getattr(self, f'layer{i}_i_neu')
                i_neu.update(r_ei + r_ii)
            inp = e_neu.spike.value

        outputs = {
            "e_neu":{
                "spike": [getattr(self, f'layer{i}_e_neu').spike.value for i in range(self.n_layer)],
                "inp": [getattr(self, f'layer{i}_e_neu').input.value for i in range(self.n_layer)],
                },
            "i_neu":{
                "spike": [getattr(self, f'layer{i}_i_neu').spike.value for i in range(self.n_layer)],
                "inp": [getattr(self, f'layer{i}_i_neu').input.value for i in range(self.n_layer)],
                }
            }

        return outputs
    