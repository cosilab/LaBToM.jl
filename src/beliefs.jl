using PDDL
using Gen, GenParticleFilters
using InversePlanning

using GenParticleFilters: softmax, logsumexp
using InversePlanning: BeliefConfig, maybe_sample
using Combinatorics: with_replacement_combinations

include("utils.jl")

"""
    ParticleBeliefState(env_states, log_weights)

Particle belief representation of a distribution of environment states,
consisting of `env_states` and their unnormalized `log_weights`. 
"""
struct ParticleBeliefState{S}
    env_states::Vector{S}
    log_weights::Vector{Float64}
end

ParticleBeliefState(env_states::Vector{S}, log_weights::Vector{Float64}) where {S} =
    ParticleBeliefState{S}(env_states, log_weights)
ParticleBeliefState{S}(env_states::Vector{S}) where {S} = 
    ParticleBeliefState{S}(env_states, zeros(length(env_states)))
ParticleBeliefState(env_states::Vector{S}) where {S} = 
    ParticleBeliefState{S}(env_states, zeros(length(env_states)))

function Base.getproperty(b::ParticleBeliefState, name::Symbol)
    if name == :weights
        return exp.(b.log_weights)
    elseif name == :probs
        return softmax(b.log_weights)
    elseif name == :log_probs
        return b.log_weights .- logsumexp(b.log_weights)
    else
        return getfield(b, name)
    end
end

"""
    ParticleBeliefConfig(
        domain, obs_model, possible_envs,
        [belief_dists, belief_prior]
    )

Constructs a `BeliefConfig` where the agent's beliefs are represented as a 
particle collection. Each particle corresponds to a possible environment state, 
and has an associated weight.

# Arguments
- `domain`: PDDL domain that defines the semantics of the environment.
- `obs_model`: Function that returs a dictionary of observed fluents and their
    values given a PDDL domain and environment state.
- `possible_envs`: Vector of possible initial environment states.
- `belief_dists`: Vector of log probability vectors over environments. Defaults
    to enumerating all ways `N` samples distribute over `N` possible states.
- `belief_prior`: Vector of prior probabilities over `belief_dists`. Defaults
    to the uniform distribution.
"""
function ParticleBeliefConfig(
    domain::Domain, obs_model, possible_envs::Vector,
    belief_dists::Vector = enumerate_belief_dists(length(possible_envs)),
    belief_prior::Vector = ones(length(belief_dists)) ./ length(belief_dists)
)
    return BeliefConfig(
        env_independent_particle_belief_init,
        (possible_envs, belief_dists, belief_prior),
        partial_obs_particle_belief_step,
        (domain, obs_model)
    )
end


"""
    ParticleTrueBeliefConfig(domain, obs_model)

Constructs a `BeliefConfig` where the agent's beliefs are always true, 
represented as a particle collection assigning probability of 1.0 to the
true environment state for that trace.
"""
function ParticleTrueBeliefConfig(
    domain::Domain, obs_model
)
    possible_env_fn(env_state) = [env_state]
    belief_dist_fn(env_states) = [[0.0]]
    belief_prior_fn(env_states, belief_dists) = [1.0]
    return BeliefConfig(
        env_conditioned_particle_belief_init,
        (possible_env_fn, belief_dist_fn, belief_prior_fn),
        partial_obs_particle_belief_step,
        (domain, obs_model)
    )
end

"""
    env_independent_particle_belief_init(env_state, possible_envs,
                                         belief_dists, belief_prior)

Given a list of possible environment states (`possible_envs`), a list of
log probability vectors (`belief_dists`) and a prior over those probability
vectors (`belief_prior`), samples a belief distribution over environment states
(indexed by the address `belief_idx`), and returns `ParticleBeliefState` with
that distribution.
"""
@gen function env_independent_particle_belief_init(
    env_state, possible_envs,
    belief_dists::Vector{Vector{Float64}}, belief_prior::Vector{Float64}
)
    # Sample a distribution over possible environments
    belief_id ~ categorical(belief_prior)
    log_weights = belief_dists[belief_id]
    # Construct and return belief state
    belief_state = ParticleBeliefState(possible_envs, log_weights)
    return belief_state
end

"""
    env_conditioned_particle_belief_init(env_state, possible_env_fn,
                                         belief_dist_fn, belief_prior_fn)

Given the true initial environment `env_state`, uses `possible_env_fn` to 
construct a list of possible environments, then uses `belief_dist_fn` to compute
a list of log probability vectors over those environments. A prior over those
belief vectors is then computed using `belief_prior_fn`. Finally, a distribution
over environment states is sampled and returned as a `ParticleBeliefState`.
"""
@gen function env_conditioned_particle_belief_init(
    env_state, possible_env_fn,
    belief_dist_fn, belief_prior_fn
)
    # Compute the set of possible environment states and belief distributions
    possible_envs = possible_env_fn(env_state)
    belief_dists = belief_dist_fn(possible_envs)
    belief_prior = belief_prior_fn(possible_envs, belief_dists)
    # Sample a distribution over possible environments
    belief_id ~ categorical(belief_prior)
    log_weights = belief_dists[belief_id]
    # Construct and return belief state
    belief_state = ParticleBeliefState(possible_envs, log_weights)
    return belief_state
end

"""
    partial_obs_particle_belief_step(t, belief_state, act_state, env_state,
                                     domain, obs_model)

Belief update under noiseless but partial observation of the true `env_state`,
given by the `obs_model`. The weight of each particle in the agent's belief
is either kept the same or set to zero if it is inconsistent with the
observation or most recent action. 
"""
@gen function partial_obs_particle_belief_step(
    t, belief_state::ParticleBeliefState, act_state, env_state,
    domain, obs_model
)
    act = convert(Term, act_state)
    # Compute observed fluents and their values
    obs = obs_model(domain, env_state)
    # Update each particle's state and weight
    next_states = copy(belief_state.env_states)
    next_log_weights = copy(belief_state.log_weights)
    for i in 1:length(belief_state.env_states)
        # Check if action is possible
        if t > 1 && !available(domain, belief_state.env_states[i], act)
            next_log_weights[i] = -Inf
            continue
        end
        # Simulate next state for particle
        next_states[i] = transition(domain, belief_state.env_states[i], act)
        # Skip checking if this state is already impossible
        if next_log_weights[i] == -Inf
            continue
        end
        # Check for consistency with observations
        if any(PDDL.evaluate(domain, next_states[i], f) != v for (f, v) in obs) 
            next_log_weights[i] = -Inf
        end 
    end
    next_belief_state = ParticleBeliefState(next_states, next_log_weights)
    return next_belief_state
end

"""
    enumerate_belief_dists(n_envs::Int, n_samples::Int = n_envs;
                           no_zeros::Bool = true)

Enumerate across all belief distributions that can be formed by distributing
`n_samples` across all `n_envs` environments.

Returns a list of log probabliity vectors, each of which corresponds to a
distribution over `n_env` environment states. If `no_zeros` is `true`, zero
probability states will instead be assigned a probability of `eps(0.0)`.
"""
function enumerate_belief_dists(n_envs::Int, n_samples::Int = n_envs;
                                no_zeros::Bool = false)
    env_multisets = with_replacement_combinations(1:n_envs, n_samples)
    env_log_probs = map(env_multisets) do mset
        counts = [sum(mset .== i) for i in 1:n_envs]
        probs = counts ./ sum(counts) .+ (no_zeros ? eps(0.0) : 0.0)
        return log.(probs)
    end
    return env_log_probs
end

"""
    dkg_enumerate_possible_envs(
        ref_state::State;
        min_keys, max_keys,
        min_color_keys,max_color_keys
    )

Enumerate all possible environment states consistent with the reference state
`ref_state` in the Doors, Keys and Gems domain. Places between `min_keys` and
`max_keys` keys among the boxes in the reference state, ensuring that no less
than `min_color_keys` and no more than `max_color_keys` of each color is present.
"""
function dkg_enumerate_possible_envs(
    ref_state::State;
    boxes = PDDL.get_objects(ref_state, :box),
    colors = PDDL.get_objects(ref_state, :color),
    min_keys = 1,
    max_keys = min(2, length(boxes)),
    min_color_keys = Dict{Const, Int}(c => min_keys for c in colors),
    max_color_keys = Dict{Const, Int}(c => max_keys for c in colors),
)
    env_states = Vector{typeof(ref_state)}()
    n_boxes = length(boxes)
    max_keys = min(max_keys, sum(values(max_color_keys)))
    # Extract keys that correspond to boxes
    keys = [k for k in PDDL.get_objects(ref_state, :key)
            if ref_state[pddl"(offgrid $k)"] || ref_state[pddl"(hidden $k)"]]
    if length(keys) < length(boxes)
        error("Not enough keys to fill boxes.")
    elseif length(keys) > length(boxes)
        resize!(keys, length(boxes))
    end
    # Create base state with no boxes
    base_state = copy(ref_state)
    for box in boxes
        empty_box!(base_state, box)
    end
    # Enumerate over number of keys
    for n_keys in min_keys:max_keys
        color_iter = Iterators.product((colors for _ in 1:n_keys)...)
        # Enumerate over subsets of boxes to place keys in
        for key_idxs in IterTools.subsets(1:n_boxes, n_keys)
            # Enumerate over color assignments to keys within subset
            for key_colors in color_iter
                # Skip if too little or too many of any one color
                counts = [sum(==(c), key_colors, init=0) for c in colors]
                if (any(n < min_color_keys[c] for (c, n) in zip(colors, counts)) ||
                    any(n > max_color_keys[c] for (c, n) in zip(colors, counts)))
                    continue
                end
                # Create new state and set key colors and locations
                s = copy(base_state)
                for (idx, color) in zip(key_idxs, key_colors)
                    key = keys[idx]
                    set_color!(s, key, color)
                    place_key_in_box!(s, key, boxes[idx])
                end
                push!(env_states, s)
            end
        end
    end
    return env_states
end

"""
    dkg_observation_model(domain::Domain, env_state::State)

Observation model for agents in the Doors, Keys and Gems domain. Returns a 
dictionary containing key locations and colors for all keys that are not hidden
or off the grid (i.e. picked up by an agent).
"""
function dkg_observation_model(domain::Domain, env_state::State)
    observations = Dict{Term, Union{Bool, Int, Float64}}()
    for key in PDDL.get_objects(env_state, :key)
        if (env_state[pddl"(hidden $key)"] || env_state[pddl"(offgrid $key)"])
            continue
        end
        observations[pddl"(xloc $key)"] = env_state[pddl"(xloc $key)"]
        observations[pddl"(yloc $key)"] = env_state[pddl"(yloc $key)"]
        for color in PDDL.get_objects(env_state, :color)
            observations[pddl"(iscolor $key $color)"] =
                env_state[pddl"(iscolor $key $color)"]
        end
        for box in PDDL.get_objects(env_state, :box)
            observations[pddl"(inside $key $box)"] =
                env_state[pddl"(inside $key $box)"]
        end
    end
    for box in PDDL.get_objects(env_state, :box)
        env_state[pddl"(closed $box)"] && continue
        observations[pddl"(empty $box)"] =
            PDDL.evaluate(domain, env_state, pddl"(empty $box)")
        for key in PDDL.get_objects(env_state, :key)
            observations[pddl"(inside $key $box)"] =
                env_state[pddl"(inside $key $box)"]
        end
    end
    return observations
end
