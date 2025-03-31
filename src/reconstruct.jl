using PDDL, SymbolicPlanners
using GenParticleFilters
using JSON3

using JSON3: StructTypes

"Struct for storing JSON-serialized inference results."
struct InferenceResult
    plan_id::String
    state_ids::Vector{Int}
    goal_ids::Vector{Int}
    config::Dict{String, Any}
    t::Vector{Int}
    action::Vector{String}
    lml_est::Vector{Float64}
    log_weights::Vector{Vector{Float64}}
    belief_dists::Vector{Vector{Vector{Float64}}}
end

StructTypes.StructType(::Type{InferenceResult}) = StructTypes.Struct()

"Reconstruct environment state histories from initial states."
function reconstruct_env_hists(domain, initial_states, plan, state_ids)
    env_hists = map(initial_states) do state
        hist = [state]
        for act in plan
            if PDDL.available(domain, state, act)
                state = PDDL.transition(domain, state, act)
            end
            push!(hist, state)
        end
        return hist    
    end
    trace_env_hists = env_hists[state_ids]
    trace_env_hists = permutedims(reduce(hcat, trace_env_hists))
    return (env_hists, trace_env_hists)
end

function reconstruct_env_hists(result::InferenceResult, domain, initial_states)
    plan = parse_pddl.(result.action)[2:end]
    return reconstruct_env_hists(domain, initial_states, plan, result.state_ids)
end

"Reconstruct belief state histories from belief distribution vectors."
function reconstruct_belief_hists(belief_dists, env_hists)
    belief_hists = reduce(hcat, belief_dists)
    belief_hists = map(enumerate(eachcol(belief_hists))) do (t, belief_states)
        map(belief_states) do log_weights
            env_states = [h[t] for h in env_hists]
            return ParticleBeliefState(env_states, collect(Float64, log_weights))
        end
    end
    return reduce(hcat, belief_hists)
end

function reconstruct_belief_hists(result::InferenceResult, env_hists)
    return reconstruct_belief_hists(result.belief_dists, env_hists)
end

