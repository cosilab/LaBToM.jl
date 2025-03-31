using PDDL, SymbolicPlanners

import SymbolicPlanners:
    compute, precompute!, filter_available,
    get_goal_terms, set_goal_terms

include("utils.jl")

"""
    GoalManhattan

Custom relaxed distance heuristic to goal objects. Estimates the cost of 
collecting all goal objects by computing the distance between all goal objects
and the agent, then returning the minimum distance plus the number of remaining
goals to satisfy.
"""
struct GoalManhattan <: Heuristic end

function compute(heuristic::GoalManhattan,
                 domain::Domain, state::State, spec::Specification)
    # Count number of remaining goals to satisfy
    goal_count = GoalCountHeuristic()(domain, state, spec)
    # Determine goal objects to collect
    goals = get_goal_terms(spec)
    isempty(goals) && return goal_count
    # Compute minimum distance to goal objects
    min_dist = minimum(goals) do g
        g.name != :has && return 0.0f0
        state[g] && return 0.0f0
        agent, obj = g.args[1], g.args[2]
        agent_loc = get_obj_loc(state, agent)
        obj_loc = get_obj_loc(state, obj)
        return Float32(sum(abs.(agent_loc .- obj_loc)))
    end
    return min_dist + goal_count
end
