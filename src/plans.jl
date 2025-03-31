using PDDL, SymbolicPlanners
using Gen
using InversePlanning

using InversePlanning: PlanConfig

include("utils.jl")
# include("beliefs.jl")

struct ParticleBeliefPlanState{S}
    "Initial timestep of the current plan."
    init_step::Int
    "Solution returned by the planner for each environment state."
    solutions::Vector{S}
    "Specification that the solution is intended to satisfy."
    spec::Specification
end

"""
    ParticleBeliefPolicyConfig(domain::Domain, planner::Planner)

Constructs a `PlanConfig` that refines a policy at every timestep for each 
environment state tracked in a particle belief representation. If `plan_at_init`
is true, then the policies are initialized at timestep zero.
"""
function ParticleBeliefPolicyConfig(
    domain::Domain, planner::Planner
)
    return PlanConfig(
        particle_belief_policy_init, (domain, planner),
        particle_belief_policy_step, (domain, planner)
    )
end

"""
    particle_belief_policy_init(plan_state, belief_state, goal_state,
                                domain, planner)

Policy initialization for a particle belief policy. The planner is used to
compute the regular state-action policy for each environment state
with non-zero probability in the particle belief representation.
"""
@gen function particle_belief_policy_init(
    belief_state::ParticleBeliefState, goal_state,
    domain::Domain, planner::Planner
)   
    # Refine each policy starting at each corresponding environment state
    spec = convert(Specification, goal_state)
    solutions = map(belief_state.env_states) do env_state
        planner(domain, env_state, spec)
    end
    return ParticleBeliefPlanState(0, solutions, spec)
end

"""
    particle_belief_policy_step(t, plan_state, belief_state, goal_state,
                                domain, planner)

Policy update step for a particle belief policy. At each timestep, the planner
is used to refine the regular state-action policy for each environment state
with non-zero probability in the particle belief representation.
"""
@gen function particle_belief_policy_step(
    t::Int, plan_state::ParticleBeliefPlanState,
    belief_state::ParticleBeliefState, goal_state,
    domain::Domain, planner::Planner
)   
    # Ensure that number of policies and particles are matched
    @assert length(plan_state.solutions) == length(belief_state.env_states)
    spec = convert(Specification, goal_state)
    solutions = copy(plan_state.solutions)
    if spec == plan_state.spec
        # Refine each policy starting at each corresponding environment state
        for i in 1:length(belief_state.env_states)
            belief_state.log_weights[i] == -Inf && continue
            env_state = belief_state.env_states[i]
            sol = copy(plan_state.solutions[i])
            refine!(sol, planner, domain, env_state, spec)
            solutions[i] = sol
        end
    else
        # Recompute policies for new goal specification
        for i in 1:length(belief_state.env_states)
            belief_state.log_weights[i] == -Inf && continue
            env_state = belief_state.env_states[i]
            solutions[i] = planner(domain, env_state, spec)
        end
    end
    return ParticleBeliefPlanState(plan_state.init_step, solutions, spec)
end
