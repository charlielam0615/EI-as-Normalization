import numpy as np
import brainpy as bp
import brainpy.math as bm
import brainpy_datasets as bd  # pip install brainpy-datasets


@bp.tools.numba_jit
def _dms(num_steps, num_inputs, n_motion_choice, motion_tuning, is_spiking_mode,
         sample_time, test_time, fr, bg_fr, rotate_dir):
  # data
  X = np.zeros((num_steps, num_inputs))

  # sample
  match = np.random.randint(2)
  sample_dir = np.random.randint(n_motion_choice)

  # Generate the sample and test stimuli based on the rule
  if match == 1:  # match trial
    test_dir = (sample_dir + rotate_dir) % n_motion_choice
  else:
    test_dir = np.random.randint(n_motion_choice)
    while test_dir == ((sample_dir + rotate_dir) % n_motion_choice):
      test_dir = np.random.randint(n_motion_choice)

  # SAMPLE stimulus
  X[sample_time] += motion_tuning[sample_dir] * fr
  # TEST stimulus
  X[test_time] += motion_tuning[test_dir] * fr
  X += bg_fr

  # to spiking
  if is_spiking_mode:
    X = np.random.random(X.shape) < X
    X = X.astype(np.float_)

  # can use a greater weight for test period if needed
  return X, match


_rotate_choice = {
    '0': 0,
    '45': 1,
    '90': 2,
    '135': 3,
    '180': 4,
    '225': 5,
    '270': 6,
    '315': 7,
    '360': 8,
}


class DMS(bd.cognitive.CognitiveTask):
  times = ('dead', 'fixation', 'sample', 'delay', 'test')
  output_features = ('non-match', 'match')

  def __init__(
      self,
      dt=100., t_fixation=500., t_sample=500., t_delay=1000., t_test=500.,
      limits=(0., np.pi * 2), rotation_match='0', kappa=2,
      bg_fr=1., ft_motion=bd.cognitive.Feature(24, 100, 40.),
      num_trial=1024, mode='rate', seed=None,
  ):
    super().__init__(dt=dt, num_trial=num_trial, seed=seed)
    # time
    self.t_fixation = int(t_fixation / dt)
    self.t_sample = int(t_sample / dt)
    self.t_delay = int(t_delay / dt)
    self.t_test = int(t_test / dt)
    self.num_steps = self.t_fixation + self.t_sample + self.t_delay + self.t_test
    self._times = {
      'fixation': self.t_fixation,
      'sample': self.t_sample,
      'delay': self.t_delay,
      'test': self.t_test,
    }
    test_onset = self.t_fixation + self.t_sample + self.t_delay
    self.test_time = slice(test_onset, test_onset + self.t_test)
    self.fix_time = slice(0, test_onset)
    self.sample_time = slice(self.t_fixation, self.t_fixation + self.t_sample)

    # input shape
    self.features = ft_motion.set_name('motion')
    self.features.set_mode(mode)
    self.rotation_match = rotation_match
    self._rotate = _rotate_choice[rotation_match]
    self.bg_fr = bg_fr  # background firing rate
    self.v_min = limits[0]
    self.v_max = limits[1]
    self.v_range = limits[1] - limits[0]

    # Tuning function data
    self.n_motion_choice = 8
    self.kappa = kappa  # concentration scaling factor for von Mises

    # Generate list of preferred directions
    # dividing neurons by 2 since two equal
    # groups representing two modalities
    pref_dirs = np.arange(self.v_min, self.v_max, self.v_range / ft_motion.num)

    # Generate list of possible stimulus directions
    stim_dirs = np.arange(self.v_min, self.v_max, self.v_range / self.n_motion_choice)

    d = np.cos(np.expand_dims(stim_dirs, 1) - pref_dirs)
    self.motion_tuning = np.exp(self.kappa * d) / np.exp(self.kappa)

  @property
  def num_inputs(self) -> int:
    return self.features.num

  @property
  def num_outputs(self) -> int:
    return 2

  def sample_a_trial(self, index):
    fr = self.features.fr(self.dt)
    bg_fr = bd.cognitive.firing_rate(self.bg_fr, self.dt, self.features.mode)
    return _dms(self.num_steps, self.num_inputs, self.n_motion_choice,
                self.motion_tuning, self.features.is_spiking_mode,
                self.sample_time, self.test_time, fr, bg_fr, self._rotate)

