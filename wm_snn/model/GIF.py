import brainpy as bp
import brainpy.math as bm
import brainpy_datasets as bd
import jax


class GIF(bp.NeuGroupNS):
    def __init__(
            self, size, V_rest=0., V_th_inf=1., R=1., tau=20.,
            tau_th=100., Ath=1., tau_I1=5., A1=0., tau_I2=50., A2=0.,
            adaptive_th=False, V_initializer=bp.init.OneInit(0.), I1_initializer=bp.init.ZeroInit(),
            I2_initializer=bp.init.ZeroInit(), Vth_initializer=bp.init.OneInit(1.),
            method='exp_auto', keep_size=False, name=None, mode=None,
            spike_fun=bm.surrogate.ReluGrad(), v_scale_var: bool = False, input_var=True
    ):
        super().__init__(size=size, keep_size=keep_size, name=name, mode=mode)
        assert self.mode.is_parent_of(bm.TrainingMode, bm.NonBatchingMode)

        # params
        self.V_rest = bp.init.parameter(V_rest, self.varshape, allow_none=False)
        self.V_th_inf = bp.init.parameter(V_th_inf, self.varshape, allow_none=False)
        self.R = bp.init.parameter(R, self.varshape, allow_none=False)
        self.tau = bp.init.parameter(tau, self.varshape, allow_none=False)
        self.tau_th = bp.init.parameter(tau_th, self.varshape, allow_none=False)
        self.tau_I1 = bp.init.parameter(tau_I1, self.varshape, allow_none=False)
        self.tau_I2 = bp.init.parameter(tau_I2, self.varshape, allow_none=False)
        self.Ath = bp.init.parameter(Ath, self.varshape, allow_none=False)
        self.A1 = bp.init.parameter(A1, self.varshape, allow_none=False)
        self.A2 = bp.init.parameter(A2, self.varshape, allow_none=False)
        self.spike_fun = bp.check.is_callable(spike_fun, 'spike_fun')
        self.adaptive_th = adaptive_th
        self.v_scale_var = v_scale_var
        self.input_var = input_var

        # initializers
        self._V_initializer = bp.check.is_initializer(V_initializer)
        self._I1_initializer = bp.check.is_initializer(I1_initializer)
        self._I2_initializer = bp.check.is_initializer(I2_initializer)
        self._Vth_initializer = bp.check.is_initializer(Vth_initializer)

        # variables
        self.reset_state(self.mode)

        # integral
        self.int_V = bp.odeint(method=method, f=self.dV)
        self.int_I1 = bp.odeint(method=method, f=self.dI1)
        self.int_I2 = bp.odeint(method=method, f=self.dI2)
        self.int_Vth = bp.odeint(method=method, f=self.dVth)

    def reset_state(self, batch_size=None):
        self.V = bp.init.variable_(self._V_initializer, self.varshape, batch_size)
        self.I1 = bp.init.variable_(self._I1_initializer, self.varshape, batch_size)
        self.I2 = bp.init.variable_(self._I2_initializer, self.varshape, batch_size)
        if self.adaptive_th:
            self.V_th = bp.init.variable_(self._Vth_initializer, self.varshape, batch_size)
        if self.v_scale_var:
            self.Vs = bp.init.variable_(bm.zeros, self.varshape, batch_size)
        self.spike = bp.init.variable_(bm.zeros, self.varshape, batch_size)
        self.input = bp.init.variable_(bm.zeros, self.varshape, batch_size)

    def dI1(self, I1, t):
        return - I1 / self.tau_I1

    def dI2(self, I2, t):
        return - I2 / self.tau_I2

    def dVth(self, V_th, t):
        return -(V_th - self.V_th_inf) / self.tau_th

    def dV(self, V, t, I_ext):
        return (- V + self.V_rest + self.R * I_ext) / self.tau

    def update(self, x=None):
        t = bp.share.load('t')
        dt = bp.share.load('dt')
        if self.input_var:
            if x is not None:
                self.input += x
            x = self.input.value
        else:
            x = 0. if x is None else x

        I1 = jax.lax.stop_gradient(bm.where(self.spike, self.A1, self.int_I1(self.I1.value, t, dt)).value)
        I2 = self.int_I2(self.I2.value, t, dt) + self.A2 * self.spike
        V = self.int_V(self.V.value, t, I_ext=(x + I1 + I2), dt=dt)
        if self.adaptive_th:
            V_th = self.int_Vth(self.V_th.value, t, dt) + self.Ath * self.spike
            V_th_ng = jax.lax.stop_gradient(V_th)
            vs = (V - V_th) / V_th_ng
            if self.v_scale_var:
                self.Vs.value = vs
            spike = self.spike_fun(vs)
            V -= V_th_ng * spike
            self.V_th.value = V_th
        else:
            vs = (V - self.V_th_inf) / self.V_th_inf
            if self.v_scale_var:
                self.Vs.value = vs
            spike = self.spike_fun(vs)
            V -= self.V_th_inf * spike
        self.spike.value = spike
        self.I1.value = I1
        self.I2.value = I2
        self.V.value = V
        return spike