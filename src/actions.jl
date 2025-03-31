using PDDL, SymbolicPlanners
using InversePlanning
using GenParticleFilters: softmax

using SymbolicPlanners: get_action_probs, get_action_values
using InversePlanning: ActConfig

"""
    BoltzmannThompsonActConfig(temperature::Real)

Constructs an `ActConfig` which selects an environment state from the agent's
belief state via Thompson sampling, then samples actions according to the
Boltzmann distribution over the Q-values for that state.
"""
function BoltzmannThompsonActConfig(temperature::Real)
    return ActConfig(PDDL.no_op, (), boltzmann_thompson_act_step, (temperature,))
end

"""
    boltzmann_thompson_act_step(t, act_state, agent_state, env_state,
                                temperature)

Samples an environment state from the agent's belief state via Thompson sampling
then samples actions according to the Boltzmann distribution over the Q-values
for that state. To reduce variance, this marginalizes out the sampling of 
environment states.
"""
@gen function boltzmann_thompson_act_step(t, act_state, agent_state, env_state,
                                          temperature::Real)
    plan_state = agent_state.plan_state
    belief_state = agent_state.belief_state
    action_probs = Dict{Term, Float64}()
    for i in 1:length(belief_state.env_states)
        (belief_state.log_weights[i] == -Inf) && continue
        policy = BoltzmannPolicy(plan_state.solutions[i], temperature)
        for (act, p) in get_action_probs(policy, belief_state.env_states[i])
            action_probs[act] =
                get(action_probs, act, 0.0) + p * belief_state.probs[i]
        end
    end
    act = {:act} ~ action_categorical(action_probs)
    return act
end

"""
    BoltzmannQMDPActConfig(temperature::Real, min_q::Real = -Inf)

Constructs an `ActConfig` which averages the Q-values across all environment
states in the agent's belief state, then samples actions according to the
Boltzmann distribution over the averaged Q-values. To avoid negative infinite
Q-values, set `min_q` to a finite negative number.
"""
function BoltzmannQMDPActConfig(temperature::Real, min_q::Real = -Inf)
    return ActConfig(PDDL.no_op, (),
                     boltzmann_qmdp_act_step, (temperature, min_q))
end

"""
    boltzmann_qmdp_act_step(t, act_state, agent_state, env_state,
                            temperature)

Samples actions according to the Boltzmann distribution over the averaged
Q-values across all environment states in the agent's belief state.
"""
@gen function boltzmann_qmdp_act_step(t, act_state, agent_state, env_state,
                                      temperature::Real, min_q::Real)
    plan_state = agent_state.plan_state
    belief_state = agent_state.belief_state
    # Average Q-values across environment states
    # TODO: Better handling of actions possible in one state but not another
    q_values = Dict{Term, Float64}()
    for i in 1:length(belief_state.env_states)
        belief_state.log_weights[i] == -Inf && continue
        qs = get_action_values(plan_state.solutions[i], belief_state.env_states[i])
        for (act, q) in qs
            q = max(q, min_q)
            q_values[act] = get(q_values, act, 0.0) + q * belief_state.probs[i]
        end
    end
    # Compute action probabilities
    action_probs = softmax(collect(values(q_values)) ./ temperature)
    action_probs = Dict(zip(keys(q_values), action_probs))
    act = {:act} ~ action_categorical(action_probs)
    return act
end

# Action categorical distribution #

struct ActionCategorical <: Gen.Distribution{Term} end

"""
    action_categorical(action_probs)

Gen `Distribution` that samples an action from a categorical distribution given 
a dictionary mapping action terms to their probabilities.
"""
const action_categorical = ActionCategorical()

(d::ActionCategorical)(args...) = Gen.random(d, args...)

@inline function Gen.random(::ActionCategorical, action_probs::Dict)
    u = rand()
    for (act, prob) in action_probs
        u -= prob
        (u < 0) && return act
    end
    return PDDL.no_op
end

@inline function Gen.logpdf(::ActionCategorical, act::Term, action_probs::Dict)
    act.name == InversePlanning.DO_SYMBOL && return 0.0
    return log(get(action_probs, act, 0.0))
end

Gen.logpdf_grad(::ActionCategorical, act::Term, action_probs::Dict) =
    (nothing, nothing)
Gen.has_output_grad(::ActionCategorical) =
    false
Gen.has_argument_grads(::ActionCategorical) =
    (false,)
