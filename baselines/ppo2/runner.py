import numpy as np
from baselines.common.runners import AbstractEnvRunner


class Runner(AbstractEnvRunner):
    """
    We use this object to make a mini batch of experiences
    __init__:
    - Initialize the runner

    run():
    - Make a mini batch
    """
    def __init__(self, *, env, model, nsteps, gamma, lam):
        super().__init__(env=env, model=model, nsteps=nsteps)
        # Lambda used in GAE (General Advantage Estimation)
        self.lam = lam
        # Discount rate
        self.gamma = gamma

    def run(self):
        # Here, we init the lists that will contain the mb of experiences
        mb_obs, mb_dones, mb_rewards = [], [], []
        pro_mb_actions, pro_mb_values, pro_mb_neglogpacs = [], [], []
        adv_mb_actions, adv_mb_values, adv_mb_neglogpacs = [], [], []
        mb_states = self.states
        epinfos = []
        # For n in range number of steps
        for _ in range(self.nsteps):
            # Given observations, get action value and neglopacs
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
            #TODO: pro_actions and adv_actions has the same dimension
            pro_actions, pro_values, self.states, pro_neglogpacs = self.model.pro_step(self.obs, S=self.states, M=self.dones)
            adv_actions, adv_values, self.states, adv_neglogpacs = self.model.adv_step(self.obs, S=self.states, M=self.dones)

            mb_obs.append(self.obs.copy())
            pro_mb_actions.append(pro_actions)
            adv_mb_actions.append(adv_actions)
            pro_mb_values.append(pro_values)
            adv_mb_values.append(adv_values)
            pro_mb_neglogpacs.append(pro_neglogpacs)
            adv_mb_neglogpacs.append(adv_neglogpacs)
            mb_dones.append(self.dones)

            # Take actions in env and look the results
            # Infos contains a ton of useful informations
            actions = np.append(pro_actions, adv_actions)
            actions = [0, 0, 0]
            self.obs[:], rewards, self.dones, infos = self.env.step(actions)

            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)
            mb_rewards.append(rewards)

        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)

        pro_mb_actions = np.asarray(pro_mb_actions)
        adv_mb_actions = np.asarray(adv_mb_actions)

        pro_mb_values = np.asarray(pro_mb_values, dtype=np.float32)
        adv_mb_values = np.asarray(adv_mb_values, dtype=np.float32)

        pro_mb_neglogpacs = np.asarray(pro_mb_neglogpacs, dtype=np.float32)
        adv_mb_neglogpacs = np.asarray(adv_mb_neglogpacs, dtype=np.float32)

        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        pro_last_values = self.model.pro_value(self.obs, S=self.states, M=self.dones)
        adv_last_values = self.model.adv_value(self.obs, S=self.states, M=self.dones)

        # discount/bootstrap off value fn
        pro_mb_returns = np.zeros_like(mb_rewards)
        pro_mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = pro_last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = pro_mb_values[t+1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - pro_mb_values[t]
            pro_mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        pro_mb_returns = pro_mb_advs + pro_mb_values

        adv_mb_returns = np.zeros_like(mb_rewards)
        adv_mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = adv_last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = adv_mb_values[t+1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - adv_mb_values[t]
            adv_mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        adv_mb_returns = adv_mb_advs + adv_mb_values

        return (*map(sf01, (mb_obs, pro_mb_returns, adv_mb_returns, mb_dones, pro_mb_actions, adv_mb_actions,
                            pro_mb_values, adv_mb_values, pro_mb_neglogpacs, adv_mb_neglogpacs)), mb_states, epinfos)
# obs, returns, masks, actions, values, neglogpacs, states = runner.run()
def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])


