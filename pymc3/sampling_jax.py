# pylint: skip-file
import os
import re
import warnings

xla_flags = os.getenv("XLA_FLAGS", "").lstrip("--")
xla_flags = re.sub(r"xla_force_host_platform_device_count=.+\s", "", xla_flags).split()
os.environ["XLA_FLAGS"] = " ".join(["--xla_force_host_platform_device_count={}".format(100)])

import arviz as az
import jax
import numpy as np
import pandas as pd
import theano.graph.fg

from theano.link.jax.jax_dispatch import jax_funcify

import pymc3 as pm

from pymc3 import modelcontext

warnings.warn("This module is experimental.")

# Disable C compilation by default
# theano.config.cxx = ""
# This will make the JAX Linker the default
# theano.config.mode = "JAX"


def sample_tfp_nuts(
    draws=1000,
    tune=1000,
    chains=4,
    target_accept=0.8,
    random_seed=10,
    model=None,
    num_tuning_epoch=2,
    num_compute_step_size=500,
):
    import jax

    from tensorflow_probability.substrates import jax as tfp

    model = modelcontext(model)

    seed = jax.random.PRNGKey(random_seed)

    fgraph = theano.graph.fg.FunctionGraph(model.free_RVs, [model.logpt])
    fns = jax_funcify(fgraph)
    logp_fn_jax = fns[0]

    rv_names = [rv.name for rv in model.free_RVs]
    init_state = [model.test_point[rv_name] for rv_name in rv_names]
    init_state_batched = jax.tree_map(lambda x: np.repeat(x[None, ...], chains, axis=0), init_state)

    @jax.pmap
    def _sample(init_state, seed):
        def gen_kernel(step_size):
            hmc = tfp.mcmc.NoUTurnSampler(target_log_prob_fn=logp_fn_jax, step_size=step_size)
            return tfp.mcmc.DualAveragingStepSizeAdaptation(
                hmc, tune // num_tuning_epoch, target_accept_prob=target_accept
            )

        def trace_fn(_, pkr):
            return pkr.new_step_size

        def get_tuned_stepsize(samples, step_size):
            return step_size[-1] * jax.numpy.std(samples[-num_compute_step_size:])

        step_size = jax.tree_map(jax.numpy.ones_like, init_state)
        for i in range(num_tuning_epoch - 1):
            tuning_hmc = gen_kernel(step_size)
            init_samples, tuning_result, kernel_results = tfp.mcmc.sample_chain(
                num_results=tune // num_tuning_epoch,
                current_state=init_state,
                kernel=tuning_hmc,
                trace_fn=trace_fn,
                return_final_kernel_results=True,
                seed=seed,
            )

            step_size = jax.tree_multimap(get_tuned_stepsize, list(init_samples), tuning_result)
            init_state = [x[-1] for x in init_samples]

        # Run inference
        sample_kernel = gen_kernel(step_size)
        mcmc_samples, leapfrog_num = tfp.mcmc.sample_chain(
            num_results=draws,
            num_burnin_steps=tune // num_tuning_epoch,
            current_state=init_state,
            kernel=sample_kernel,
            trace_fn=lambda _, pkr: pkr.inner_results.leapfrogs_taken,
            seed=seed,
        )

        return mcmc_samples, leapfrog_num

    print("Compiling and sampling...")
    tic2 = pd.Timestamp.now()
    map_seed = jax.random.split(seed, chains)
    mcmc_samples, leapfrog_num = _sample(init_state_batched, map_seed)

    # map_seed = jax.random.split(seed, chains)
    # mcmc_samples = _sample(init_state_batched, map_seed)
    # tic4 = pd.Timestamp.now()
    # print("Sampling time = ", tic4 - tic3)

    posterior = {k: v for k, v in zip(rv_names, mcmc_samples)}

    az_trace = az.from_dict(posterior=posterior)
    tic3 = pd.Timestamp.now()
    print("Compilation + sampling time = ", tic3 - tic2)
    return az_trace  # , leapfrog_num, tic3 - tic2

def sample_numpyro_nuts(
    draws=1000,
    tune=1000,
    chains=4,
    target_accept=0.8,
    random_seed=10,
    model=None,
    progress_bar=True,
):
    from numpyro.infer import MCMC, NUTS

    from pymc3 import modelcontext

    model = modelcontext(model)

    seed = jax.random.PRNGKey(random_seed)

    fgraph = theano.graph.fg.FunctionGraph(model.free_RVs, [model.logpt])
    fns = jax_funcify(fgraph)
    logp_fn_jax = fns[0]

    rv_names = [rv.name for rv in model.free_RVs]
    init_state = [model.test_point[rv_name] for rv_name in rv_names]
    init_state_batched = jax.tree_map(lambda x: np.repeat(x[None, ...], chains, axis=0), init_state)

    @jax.jit
    def _sample(current_state, seed):
        step_size = jax.tree_map(jax.numpy.ones_like, init_state)
        nuts_kernel = NUTS(
            potential_fn=lambda x: -logp_fn_jax(*x),
            # model=model,
            target_accept_prob=target_accept,
            adapt_step_size=True,
            adapt_mass_matrix=True,
            dense_mass=False,
        )

        pmap_numpyro = MCMC(
            nuts_kernel,
            num_warmup=tune,
            num_samples=draws,
            num_chains=chains,
            postprocess_fn=None,
            chain_method="parallel",
            progress_bar=progress_bar,
        )

        pmap_numpyro.run(seed, init_params=current_state, extra_fields=("num_steps",))
        samples = pmap_numpyro.get_samples(group_by_chain=True)
        leapfrogs_taken = pmap_numpyro.get_extra_fields(group_by_chain=True)["num_steps"]
        return samples, leapfrogs_taken

    print("Compiling and sampling...")
    tic2 = pd.Timestamp.now()
    map_seed = jax.random.split(seed, chains)
    mcmc_samples, leapfrogs_taken = _sample(init_state_batched, map_seed)
    # map_seed = jax.random.split(seed, chains)
    # mcmc_samples = _sample(init_state_batched, map_seed)
    # tic4 = pd.Timestamp.now()
    # print("Sampling time = ", tic4 - tic3)

    posterior = {k: v for k, v in zip(rv_names, mcmc_samples)}

    az_trace = az.from_dict(posterior=posterior)
    tic3 = pd.Timestamp.now()
    print("Compilation + sampling time = ", tic3 - tic2)
    return az_trace  # , leapfrogs_taken, tic3 - tic2


#
# Better batch versions
#
def sample_tfp_nuts_jit_vmap(
    draws=1000,
    tune=1000,
    chains=4,
    target_accept=0.8,
    random_seed=10,
    model=None,
    num_tuning_epoch=2,
    num_compute_step_size=500,
):
    import jax

    from tensorflow_probability.substrates import jax as tfp

    model = modelcontext(model)

    seed = jax.random.PRNGKey(random_seed)

    fgraph = theano.graph.fg.FunctionGraph(model.free_RVs, [model.logpt])
    fns = jax_funcify(fgraph)
    logp_fn_jax = fns[0]

    rv_names = [rv.name for rv in model.free_RVs]
    init_state = [model.test_point[rv_name] for rv_name in rv_names]
    init_state_batched = jax.tree_map(lambda x: np.repeat(x[None, ...], chains, axis=0), init_state)

    def _sample(init_state, seed):
        def gen_kernel(step_size):
            hmc = tfp.mcmc.NoUTurnSampler(target_log_prob_fn=logp_fn_jax, step_size=step_size)
            return tfp.mcmc.DualAveragingStepSizeAdaptation(
                hmc, jax.numpy.array(tune // num_tuning_epoch, dtype=jax.numpy.int32), 
                target_accept_prob=target_accept
            )

        def trace_fn(_, pkr):
            return pkr.new_step_size

        def get_tuned_stepsize(samples, step_size):
            return step_size[-1] * jax.numpy.std(samples[-num_compute_step_size:])

        step_size = jax.tree_map(jax.numpy.ones_like, init_state)
        for i in range(num_tuning_epoch - 1):
            tuning_hmc = gen_kernel(step_size)
            init_samples, tuning_result, kernel_results = tfp.mcmc.sample_chain(
                num_results=jax.numpy.array(tune // num_tuning_epoch, dtype=jax.numpy.int32),
                current_state=init_state,
                kernel=tuning_hmc,
                trace_fn=trace_fn,
                return_final_kernel_results=True,
                seed=seed,
            )

            step_size = jax.tree_multimap(get_tuned_stepsize, list(init_samples), tuning_result)
            init_state = [x[-1] for x in init_samples]

        # Run inference
        sample_kernel = gen_kernel(step_size)
        mcmc_samples, leapfrog_num = tfp.mcmc.sample_chain(
            num_results=draws,
            num_burnin_steps=tune // num_tuning_epoch,
            current_state=init_state,
            kernel=sample_kernel,
            trace_fn=lambda _, pkr: pkr.inner_results.leapfrogs_taken,
            seed=seed,
        )

        return mcmc_samples, leapfrog_num

    _sample_jit_vmap = jax.jit(jax.vmap(_sample))

    print("Compiling and sampling...")
    tic2 = pd.Timestamp.now()
    map_seed = jax.random.split(seed, chains)
    mcmc_samples, leapfrog_num = _sample_jit_vmap(init_state_batched, map_seed)

    # map_seed = jax.random.split(seed, chains)
    # mcmc_samples = _sample(init_state_batched, map_seed)
    # tic4 = pd.Timestamp.now()
    # print("Sampling time = ", tic4 - tic3)

    posterior = {k: v for k, v in zip(rv_names, mcmc_samples)}

    az_trace = az.from_dict(posterior=posterior)
    tic3 = pd.Timestamp.now()
    print("Compilation + sampling time = ", tic3 - tic2)
    return az_trace  # , leapfrog_num, tic3 - tic2


def sample_numpyro_nuts_vmap(
    draws=1000,
    tune=1000,
    chains=4,
    target_accept=0.8,
    random_seed=10,
    model=None,
    progress_bar=True,
    chain_method="parallel"
):
    from numpyro.infer import MCMC, NUTS

    from pymc3 import modelcontext

    model = modelcontext(model)

    seed = jax.random.PRNGKey(random_seed)

    fgraph = theano.graph.fg.FunctionGraph(model.free_RVs, [model.logpt])
    fns = jax_funcify(fgraph)
    logp_fn_jax = fns[0]

    rv_names = [rv.name for rv in model.free_RVs]
    init_state = [model.test_point[rv_name] for rv_name in rv_names]
    init_state_batched = jax.tree_map(lambda x: np.repeat(x[None, ...], chains, axis=0), init_state)

    @jax.jit
    def _sample(current_state, seed):
        step_size = jax.tree_map(jax.numpy.ones_like, init_state)
        nuts_kernel = NUTS(
            potential_fn=lambda x: -logp_fn_jax(*x),
            # model=model,
            target_accept_prob=target_accept,
            adapt_step_size=True,
            adapt_mass_matrix=True,
            dense_mass=False,
        )

        pmap_numpyro = MCMC(
            nuts_kernel,
            num_warmup=tune,
            num_samples=draws,
            num_chains=chains,
            postprocess_fn=None,
            chain_method=chain_method,
            progress_bar=progress_bar,
        )

        pmap_numpyro.run(seed, init_params=current_state, extra_fields=("num_steps",))
        samples = pmap_numpyro.get_samples(group_by_chain=True)
        leapfrogs_taken = pmap_numpyro.get_extra_fields(group_by_chain=True)["num_steps"]
        return samples, leapfrogs_taken

    print("Compiling and sampling...")
    tic2 = pd.Timestamp.now()
    map_seed = jax.random.split(seed, chains)
    mcmc_samples, leapfrogs_taken = _sample(init_state_batched, map_seed)
    # map_seed = jax.random.split(seed, chains)
    # mcmc_samples = _sample(init_state_batched, map_seed)
    # tic4 = pd.Timestamp.now()
    # print("Sampling time = ", tic4 - tic3)

    posterior = {k: v for k, v in zip(rv_names, mcmc_samples)}

    az_trace = az.from_dict(posterior=posterior)
    tic3 = pd.Timestamp.now()
    print("Compilation + sampling time = ", tic3 - tic2)
    return az_trace  # , leapfrogs_taken, tic3 - tic2

##
## This is for mhrw in tf
##
def sample_tfp_mhwr(
    draws=1000,
    tune=1000,
    chains=4,
    random_seed=10,
    model=None,
    num_tuning_epoch=2,
):
    import jax

    from tensorflow_probability.substrates import jax as tfp

    model = modelcontext(model)

    seed = jax.random.PRNGKey(random_seed)

    fgraph = theano.graph.fg.FunctionGraph(model.free_RVs, [model.logpt])
    fns = jax_funcify(fgraph)
    logp_fn_jax = fns[0]

    rv_names = [rv.name for rv in model.free_RVs]
    init_state = [model.test_point[rv_name] for rv_name in rv_names]
    init_state_batched = jax.tree_map(lambda x: np.repeat(x[None, ...], chains, axis=0), init_state)

### begin harvested code
              
        @tf.function(experimental_compile=True)
        def sampler_club_mhrw(init_vals, scale_):

            @tf.function
            def trace_is_accepted(states, previous_kernel_results):
                return previous_kernel_results.is_accepted, previous_kernel_results.accepted_results.target_log_prob
            
            # prior on epsilon
            #@tf.function
            #def log_prior_epsilon_(error, sigmae):
            #    return tf.math.reduce_sum(tfp.distributions.Normal(0,sigmae).log_prob(error), axis=-1)
            @tf.function
            def log_prior_epsilon_(error):
                return tf.math.reduce_sum(tfp.distributions.Normal(mean_eps, std_eps).log_prob(error), axis=-1)
            
            # prior on sigmae
            #@tf.function
            #def log_prior_sigmae_(sigmae):
            #    return tfp.distributions.HalfNormal(scale=5.).log_prob(sigmae)
            
            # log likelihood
            @tf.function
            def log_like_(beta_, Wvec_, delta_, epsilon_):
                # project delta into K space (rather than K-1)
                delta__ = tf.scatter_nd(site_idx_, delta_, shape = tf.constant([K]))
                # apportion delta__ to each row
                delta___ = tf.gather_nd(delta__, site_idx)
                        
                Wmat_upper = tf.scatter_nd(W_idx, Wvec_, (N,N))
                Wmat = Wmat_upper + tf.transpose(Wmat_upper)  
                # spatial granularity condition (Pesaran 2006) that row and column norms are uniformly bounded. 
                # In the case of unknown W, this implies that row and column norms are less than unity 
                # note: if symmetric these row and col norms for W will be equal
                max_norm_row = tf.math.reduce_max(tf.linalg.norm(Wmat, axis=1, ord=1))
                max_norm_col = tf.math.reduce_max(tf.linalg.norm(Wmat, axis=0, ord=1))
                # norm_check == 1, reject
                norm_check = tf.cond(tf.greater_equal(max_norm_row, c_one) | tf.greater_equal(max_norm_col, c_one), lambda: c_one, lambda: c_zero)
            
                # Reshape Errors for Club
                err_expanded = tf.gather_nd(epsilon_, err_idx_1)
                epsilon_club = tf.scatter_nd(err_idx, err_expanded, (dfs.shape[0],N))
            
                # multiply errors x W for each row
                W_ = tf.gather_nd(Wmat, id_n)
                club_err_contrib = tf.reduce_sum(tf.math.multiply(W_, epsilon_club), axis=1)
            
                # Logit Mean
                mu = tf.linalg.matvec(x, beta_) + epsilon_ + club_err_contrib + delta___
            
                exp_xb_all = tf.math.exp(mu)
                exp_xb_chosen = tf.multiply(exp_xb_all, choice)

                # sum each using bincount
                denom = tf.math.bincount(choice_idx,weights=exp_xb_all)
                num = tf.math.bincount(choice_idx,weights=exp_xb_chosen)
                prob = tf.math.divide(num, denom)
                # if Pasaran condition rejected return -np.inf
                logL = tf.math.reduce_sum(tf.math.log(prob), axis=-1)
                LogL = tf.cond(tf.equal(norm_check, c_zero), lambda: logL, 
                               lambda: tf.constant(-np.inf, dtype=dtype))
                return LogL 

            # @tf.function
            # def log_posterior_rum_club_(beta, Wvec, sigmae, delta, epsilon):
            #     # priors
            #     logp_epsilon = log_prior_epsilon_(epsilon, sigmae)
            #     logp_sigmae = log_prior_sigmae_(sigmae)
            #     # log like
            #     logl = log_like_(beta, Wvec, delta, epsilon)
            #     return logl + logp_epsilon + logp_sigmae

            @tf.function
            def log_posterior_rum_club_(beta, Wvec, delta, epsilon):
                # log like
                logl = log_like_(beta, Wvec, delta, epsilon)
                # priors
                logp_epsilon = log_prior_epsilon_(epsilon)
                return logl + logp_epsilon

            kernel_cl = tfp.mcmc.RandomWalkMetropolis(
                target_log_prob_fn=log_like_, #log_posterior_rum_club_,
                new_state_fn=tfp.mcmc.random_walk_normal_fn(scale=scale_))
               
            samples, stats = tfp.mcmc.sample_chain(num_results=num_samples,
                                                   current_state=init_vals,
                                                   kernel = kernel_cl,
                                                   num_burnin_steps=burnin,
                                                   num_steps_between_results=thin_amount,
                                                   trace_fn = trace_is_accepted)
            return samples, stats

### end harvested code

    
    @jax.pmap
    def _sample(init_state, seed):
    
        def trace_is_accepted(states, previous_kernel_results):
            return previous_kernel_results.is_accepted

        def gen_kernel(step_size):
            kernel_ = tfp.mcmc.RandomWalkMetropolis(
                target_log_prob_fn=logp_fn_jax,
                new_state_fn=tfp.mcmc.random_walk_normal_fn(scale=step_size))
            return kernel_
        
        for i in range(num_tuning_epoch):
             tuning_mhrw = gen_kernel(step_size)
             samples, stats = tfp.mcmc.sample_chain(num_results=num_samples,
                                                   current_state=init_vals,
                                                   kernel = tuning_mhrw,
                                                   num_burnin_steps=burnin,
                                                   num_steps_between_results=thin,
                                                   trace_fn = trace_is_accepted)
             accept_rate = stats.mean(axis=0)
             step_size = tune(step_size, accept_rate)

        # Run inference
        sample_kernel = gen_kernel(step_size)
        mcmc_samples, mcmc_stats = tfp.mcmc.sample_chain(
            num_results=draws,
            num_burnin_steps=tune // num_tuning_epoch,
            current_state=init_state,
            kernel=sample_kernel,
            trace_fn=trace_is_accepted,
            seed=seed,
        )
        return mcmc_samples, leapfrog_num

    print("Compiling and sampling...")
    tic2 = pd.Timestamp.now()
    map_seed = jax.random.split(seed, chains)
    mcmc_samples, leapfrog_num = _sample(init_state_batched, map_seed)

    # map_seed = jax.random.split(seed, chains)
    # mcmc_samples = _sample(init_state_batched, map_seed)
    # tic4 = pd.Timestamp.now()
    # print("Sampling time = ", tic4 - tic3)

    posterior = {k: v for k, v in zip(rv_names, mcmc_samples)}

    az_trace = az.from_dict(posterior=posterior)
    tic3 = pd.Timestamp.now()
    print("Compilation + sampling time = ", tic3 - tic2)
    return az_trace 

def tune(scale, acc_rate):
    """
    This is a pymc3 function
    
    Tunes the scaling parameter for the proposal distribution
    according to the acceptance rate over the last tune_interval:
    Rate    Variance adaptation
    ----    -------------------
    <0.001        x 0.1
    <0.05         x 0.5
    <0.2          x 0.9
    >0.5          x 1.1
    >0.75         x 2
    >0.95         x 10
    """
    if acc_rate < 0.001:
        # reduce by 90 percent
        return scale * 0.1
    elif acc_rate < 0.05:
        # reduce by 50 percent
        return scale * 0.5
    elif acc_rate < 0.24:
        # reduce by ten percent
        return scale * 0.9
    elif acc_rate > 0.95:
        # increase by factor of ten
        return scale * 10.0
    elif acc_rate > 0.75:
        # increase by double
        return scale * 2.0
    elif acc_rate > 0.5:
        # increase by ten percent
        return scale * 1.1

    return scale
