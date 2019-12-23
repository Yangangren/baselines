import tensorflow as tf
import functools
import numpy as np

from baselines.common.tf_util import get_session, save_variables, load_variables
from baselines.common.tf_util import initialize

try:
    from baselines.common.mpi_adam_optimizer import MpiAdamOptimizer
    from mpi4py import MPI
    from baselines.common.mpi_util import sync_from_root
except ImportError:
    MPI = None


class Model(object):
    """
    We use this object to :
    __init__:
    - Creates the step_model
    - Creates the train_model

    train():
    - Make the training part (feedforward and retropropagation of gradients)

    save/load():
    - Save load the model
    """
    def __init__(self, *, pro_policy, adv_policy, ob_space, ac_space, nbatch_act, nbatch_train,
                nsteps, ent_coef, vf_coef, max_grad_norm, mpi_rank_weight=1, comm=None, microbatch_size=None):
        self.sess = sess = get_session()

        if MPI is not None and comm is None:
            comm = MPI.COMM_WORLD

        with tf.variable_scope('ppo2_model', reuse=tf.AUTO_REUSE):
            # CREATE OUR TWO MODELS
            # act_model that is used for sampling
            act_model = pro_policy(nbatch_act, 1, sess)

            # Train model for training
            if microbatch_size is None:
                train_model = pro_policy(nbatch_train, nsteps, sess)
            else:
                train_model = pro_policy(microbatch_size, nsteps, sess)

        with tf.variable_scope('ppo2_adv', reuse=tf.AUTO_REUSE):
            # CREATE OUR TWO MODELS
            # act_model that is used for sampling
            act_adv = adv_policy(nbatch_act, 1, sess)
            # Train model for training
            if microbatch_size is None:
                train_adv = adv_policy(nbatch_train, nsteps, sess)
            else:
                train_adv = adv_policy(microbatch_size, nsteps, sess)

        # CREATE THE PLACEHOLDERS
        # actions
        self.A1 = A1 = train_model.pdtype.sample_placeholder([None])
        self.A2 = A2 = train_adv.pdtype.sample_placeholder([None])
        # advantages
        self.ADV1 = ADV1 = tf.placeholder(tf.float32, [None])
        self.ADV2 = ADV2 = tf.placeholder(tf.float32, [None])
        # return
        self.R1 = R1 = tf.placeholder(tf.float32, [None])
        self.R2 = R2 = tf.placeholder(tf.float32, [None])
        # Keep track of old actor
        self.OLDNEGLOGPAC1 = OLDNEGLOGPAC1 = tf.placeholder(tf.float32, [None])
        self.OLDNEGLOGPAC2 = OLDNEGLOGPAC2 = tf.placeholder(tf.float32, [None])

        # Keep track of old critic
        self.OLDVPRED1 = OLDVPRED1 = tf.placeholder(tf.float32, [None])
        self.OLDVPRED2 = OLDVPRED2 = tf.placeholder(tf.float32, [None])

        # learning rate
        self.LR = LR = tf.placeholder(tf.float32, [])

        # Cliprange
        self.CLIPRANGE = CLIPRANGE = tf.placeholder(tf.float32, [])

        neglogpac = train_model.pd.neglogp(A1)
        neglogpac_adv = train_adv.pd.neglogp(A2)

        # Calculate the entropy
        # Entropy is used to improve exploration by limiting the premature convergence to suboptimal policy.
        entropy = tf.reduce_mean(train_model.pd.entropy())
        entropy_adv = tf.reduce_mean(train_adv.pd.entropy())

        # CALCULATE THE LOSS
        # Total loss = Policy gradient loss - entropy * entropy coefficient + Value coefficient * value loss

        # Clip the value to reduce variability during Critic training
        # Get the predicted value
        vpred = train_model.vf
        vpredclipped = OLDVPRED1 + tf.clip_by_value(train_model.vf - OLDVPRED1, - CLIPRANGE, CLIPRANGE)
        # Unclipped value
        vf_losses1 = tf.square(vpred - R1)
        # Clipped value
        vf_losses2 = tf.square(vpredclipped - R1)

        vf_loss = 0.5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))

        vpred_adv = train_adv.vf
        vpredclipped_adv = OLDVPRED2 + tf.clip_by_value(train_adv.vf - OLDVPRED2, - CLIPRANGE, CLIPRANGE)
        # Unclipped value
        vf_losses1_adv = tf.square(vpred_adv - R2)
        # Clipped value
        vf_losses2_adv = tf.square(vpredclipped_adv - R2)
        vf_loss_adv = 0.5 * tf.reduce_mean(tf.maximum(vf_losses1_adv, vf_losses2_adv))

        # Calculate ratio (pi current policy / pi old policy)
        ratio = tf.exp(OLDNEGLOGPAC1 - neglogpac)
        ratio_adv = tf.exp(OLDNEGLOGPAC2 - neglogpac_adv)

        pg_losses = ADV1 * ratio
        pg_losses2 = ADV1 * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
        pg_loss = -tf.reduce_mean(tf.minimum(pg_losses, pg_losses2))

        pg_losses_adv = ADV2 * ratio_adv
        pg_losses2_adv = ADV2 * tf.clip_by_value(ratio_adv, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
        pg_loss_adv = tf.reduce_mean(tf.minimum(pg_losses_adv, pg_losses2_adv))

        # Final PG loss
        approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC1))
        clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))
        approxkl_adv = .5 * tf.reduce_mean(tf.square(neglogpac_adv - OLDNEGLOGPAC2))
        clipfrac_adv = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio_adv - 1.0), CLIPRANGE)))

        # Total loss
        pro_loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef
        adv_loss = pg_loss_adv - entropy_adv * ent_coef + vf_loss_adv * vf_coef

        # UPDATE THE PARAMETERS USING LOSS
        # 1. Get the model parameters
        params1 = tf.trainable_variables('ppo2_model')
        params2 = tf.trainable_variables('ppo2_adv')
        # 2. Build our trainer
        if comm is not None and comm.Get_size() > 1:
            self.trainer1 = MpiAdamOptimizer(comm, learning_rate=LR, mpi_rank_weight=mpi_rank_weight, epsilon=1e-5)
            self.trainer2 = MpiAdamOptimizer(comm, learning_rate=LR, mpi_rank_weight=mpi_rank_weight, epsilon=1e-5)
        else:
            self.trainer1 = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
            self.trainer2 = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
        # 3. Calculate the gradients
        grads_and_var = self.trainer1.compute_gradients(pro_loss, var_list=params1)
        grads_and_var2 = self.trainer2.compute_gradients(adv_loss, var_list=params2)

        grads1, var1 = zip(*grads_and_var)
        grads2, var2 = zip(*grads_and_var2)

        if max_grad_norm is not None:
            # Clip the gradients (normalize)
            grads1, _grad_norm1 = tf.clip_by_global_norm(grads1, max_grad_norm)
            grads2, _grad_norm2 = tf.clip_by_global_norm(grads2, max_grad_norm)
        grads_and_var1 = list(zip(grads1, var1))
        grads_and_var2 = list(zip(grads2, var2))

        # zip aggregate each gradient with parameters associated
        # For instance zip(ABCD, xyza) => Ax, By, Cz, Da

        self.pro_grads = grads1
        self.pro_var = var1
        self.adv_grads = grads2
        self.adv_var = var2

        self.pro_train_op = self.trainer1.apply_gradients(grads_and_var1)
        self.adv_train_op = self.trainer2.apply_gradients(grads_and_var2)

        self.pro_loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac']
        self.adv_loss_names = ['policy_loss_adv', 'value_loss_adv', 'policy_entropy_adv', 'approxkl_adv', 'clipfrac_adv']
        self.pro_stats_list = [pg_loss, vf_loss, entropy, approxkl, clipfrac]
        self.adv_stats_list = [pg_loss_adv, vf_loss_adv, entropy_adv, approxkl_adv, clipfrac_adv]

        self.pro_train_model = train_model
        self.adv_train_model = train_adv
        self.pro_act_model = act_model
        self.adv_act_model = act_adv

        self.pro_step = act_model.step
        self.adv_step = act_adv.step
        self.pro_value = act_model.value
        self.adv_value = act_adv.value

        self.initial_state = act_model.initial_state

        self.save = functools.partial(save_variables, sess=sess)
        self.load = functools.partial(load_variables, sess=sess)

        initialize()
        global_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="")
        if MPI is not None:
            sync_from_root(sess, global_variables, comm=comm) #pylint: disable=E1101

    def pro_train(self, lr, cliprange, obs, returns, masks, actions, values, neglogpacs, states=None):
        # Here we calculate advantage A(s,a) = R + yV(s') - V(s)
        # Returns = R + yV(s')
        advs = returns - values

        # Normalize the advantages
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        td_map = {
            self.pro_train_model.X: obs,
            self.A1: actions,
            self.ADV1: advs,
            self.R1: returns,
            self.LR: lr,
            self.CLIPRANGE: cliprange,
            self.OLDNEGLOGPAC1: neglogpacs,
            self.OLDVPRED1: values
        }
        if states is not None:
            td_map[self.pro_train_model.S] = states
            td_map[self.pro_train_model.M] = masks

        return self.sess.run(
            self.pro_stats_list + [self.pro_train_op],
            td_map
        )[:-1]

    def adv_train(self, lr, cliprange, obs, returns, masks, actions, values, neglogpacs, states=None):
        # Here we calculate advantage A(s,a) = R + yV(s') - V(s)
        # Returns = R + yV(s')
        advs = returns - values

        # Normalize the advantages
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        td_map = {
            self.adv_train_model.X: obs,
            self.A2: actions,
            self.ADV2: advs,
            self.R2: returns,
            self.LR: lr,
            self.CLIPRANGE: cliprange,
            self.OLDNEGLOGPAC2: neglogpacs,
            self.OLDVPRED2: values
        }
        if states is not None:
            td_map[self.adv_train_model.S] = states
            td_map[self.adv_train_model.M] = masks

        return self.sess.run(
            self.adv_stats_list + [self.adv_train_op],
            td_map
        )[:-1]
