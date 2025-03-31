using PDDL, SymbolicPlanners
using IterTools

"Returns the color of an object."
function get_obj_color(state::State, obj::Const)
    for color in PDDL.get_objects(state, :color)
        if state[Compound(:iscolor, Term[obj, color])]
            return color
        end
    end
    return Const(:none)
end

"Returns the location of an object."
function get_obj_loc(state::State, obj::Const; check_has::Bool=false)
    x = state[Compound(:xloc, Term[obj])]
    y = state[Compound(:yloc, Term[obj])]
    # Check if object is held by an agent, and return agent's location if so
    if check_has && PDDL.get_objtype(state, obj) in (:gem, :key)
        for agent in PDDL.get_objects(state, :agent)
            if state[Compound(:has, Term[agent, obj])]
                x, y = get_obj_loc(state, agent)
                break
            end
        end
    end
    return (x, y)
end

"Sets the location of an object."
function set_obj_loc!(state::State, obj::Const, loc::Tuple{Int,Int})
    state[pddl"(xloc $obj)"] = loc[1]
    state[pddl"(yloc $obj)"] = loc[2]
    return loc
end

"Removes the color of an object."
function remove_color!(state::State, obj::Const)
    for color in PDDL.get_objects(state, :color)
        state[pddl"(iscolor $obj $color)"] = false
    end
    return state
end

"Sets the color of an object."
function set_color!(state::State, obj::Const, color::Const)
    remove_color!(state, obj)
    state[pddl"(iscolor $obj $color)"] = true
    return color
end

"Empties a box of all keys."
function empty_box!(state::State, box::Const)
    for key in PDDL.get_objects(state, :key)
        state[pddl"(inside $key $box)"] || continue
        state[pddl"(inside $key $box)"] = false
        state[pddl"(hidden $key)"] = false
        state[pddl"(offgrid $key)"] = true
        set_obj_loc!(state, key, (-1, -1))
        remove_color!(state, key)
    end
    return state
end

"Places a key in a box."
function place_key_in_box!(state::State, key::Const, box::Const)
    box_loc = get_obj_loc(state, box)
    set_obj_loc!(state, key, box_loc)
    state[pddl"(inside $key $box)"] = true
    state[pddl"(hidden $key)"] = true
    state[pddl"(offgrid $key)"] = false
    return state
end

"Returns a string representation of the box contents in the state."
function box_contents_str(state::State)
    keys = sort(collect(PDDL.get_objects(state, :key)), by=string)
    boxes = sort(collect(PDDL.get_objects(state, :box)), by=string)
    str = map(boxes) do box
        for k in keys
            state[pddl"(inside $k $box)"] || continue
            color = get_obj_color(state, k)
            return first(string(color))
        end
        return '_'
    end |> join
    return str
end

"Extracts keys and boxes relevant to a goal or plan."
function extract_relevant_keys_and_boxes(
    domain::Domain, state::State, plan::AbstractVector{<:Term};
    hidden_only = true
)
    plan_keys = [act.args[2] for act in plan if act.name == :pickup]
    filter!(k -> PDDL.get_objtype(state, k) == :key, plan_keys)
    if hidden_only
        filter!(k -> state[pddl"(hidden $k)"], plan_keys)
    end
    key_colors = [get_obj_color(state, k) for k in plan_keys]
    plan_boxes = map(plan_keys) do k
        for b in PDDL.get_objects(state, :box)
            if state[pddl"(inside $k $b)"]
                return b
            end
        end
        return Const(:none)
    end
    return key_colors, plan_keys, plan_boxes
end

function extract_relevant_keys_and_boxes(
    domain::Domain, state::State, goal;
    hidden_only = true, planner = AStarPlanner(GoalManhattan())
)
    sol = planner(domain, state, goal)
    plan = collect(sol)
    return extract_relevant_keys_and_boxes(domain, state, plan; hidden_only)
end

"Counts the number of keys of each color necessary to reach a set of goals."
function count_necessary_key_colors(domain::Domain, state::State, goals)
    key_counts = Dict{Const, Int}()
    for goal in goals
        key_colors, _, _ = extract_relevant_keys_and_boxes(domain, state, goal;
                                                           hidden_only = false)
        goal_key_counts = Dict{Const, Int}()
        for color in key_colors
            goal_key_counts[color] = get(goal_key_counts, color, 0) + 1
        end
        for (key, count) in goal_key_counts
            key_counts[key] = max(get(key_counts, key, 0), count)
        end
    end
    return key_counts
end

"Counts the number of keys of each color that are visible in the state."
function count_visible_key_colors(domain::Domain, state::State)
    key_counts = Dict{Const, Int}()
    for key in PDDL.get_objects(state, :key)
        if !(state[pddl"(hidden $key)"] || state[pddl"(offgrid $key)"]) 
            color = get_obj_color(state, key)
            key_counts[color] = get(key_counts, color, 0) + 1
        end
    end
    return key_counts
end