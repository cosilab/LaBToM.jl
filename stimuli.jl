using PDDL, SymbolicPlanners
using PDDLViz, GLMakie
using JSON3
using Random

include("src/render.jl")
include("src/plan_io.jl")
include("src/utils.jl")
include("src/heuristics.jl")
include("src/interpreter.jl")

PDDL.Arrays.register!()

"Generates stimulus animation from a initial state and plan."
function generate_stim_anim(
    path::Union{AbstractString, Nothing},
    domain::Domain,
    state::State,
    plan::AbstractVector{<:Term};
    i = 1,
    renderer = RENDERER,
    trail_length = 10,
    show_inventory = true,
    inventory_titlesize = 28,
    framerate = 3,
    format = "gif",
    loop = -1,
    kwargs...
)
    # Render initial state
    canvas = PDDLViz.new_canvas(renderer)
    anim_initialize!(canvas, renderer, domain, state;
                     show_inventory = show_inventory, trail_length, kwargs...)
    # Set inventory title font size 
    if show_inventory
        canvas.blocks[2].titlesize = inventory_titlesize
    end
    # Animate plan
    anim = anim_plan!(canvas, renderer, domain, state, plan;
                      trail_length, show_inventory,
                      framerate, format, loop, kwargs...)
    # Save animation
        if !isnothing(path)
            save("$(path)_$i.$format", anim)
        end
    return anim
end

function generate_stim_anim(
    path::Union{AbstractString, Nothing},
    domain::Domain,
    problem::Problem,
    plan::AbstractVector{<:Term};
    kwargs...
)
    state = initstate(domain, problem)
    anim = generate_stim_anim(path, domain, state, plan; kwargs...)
    return anim
end

"Create storyboard plot from a stimulus."
function generate_stim_storyboard(
    domain::Domain,
    problem::Problem,
    plan::AbstractVector{<:Term},
    timesteps::AbstractVector{Int};
    subtitles = fill("", length(timesteps)),
    xlabels = fill("", length(timesteps)),
    xlabelsize = 20, subtitlesize = 24, n_rows = 1,
    kwargs...
)
    # Generate animation without smooth transitions
    anim = generate_stim_anim(nothing, domain, problem, plan; kwargs...)
    # Create goal inference storyboard
    timesteps = timesteps .+ 1
    storyboard = render_storyboard(
        anim, timesteps;
        subtitles, xlabels, xlabelsize, subtitlesize, n_rows
    )
    return storyboard
end

"Generates animation segments from an initial state, plan, and splitpoints."
function generate_stim_anim_segments(
    basepath::Union{AbstractString, Nothing},
    domain::Domain,
    state::State,
    plan::AbstractVector{<:Term},
    splitpoints::AbstractVector{Int};
    renderer = RENDERER,
    trail_length = 5,
    show_inventory = true,
    inventory_titlesize = 28,
    framerate = 3,
    format = "gif",
    loop = -1,
    track_stopmarker = ' ',
    kwargs...
)
    trajectory = PDDL.simulate(domain, state, plan)
    # Adjust splitpoints to include the start and end of the trajectory
    splitpoints = copy(splitpoints) .+ 1
    if isempty(splitpoints) || last(splitpoints) != length(trajectory)
        push!(splitpoints, length(trajectory))
    end
    pushfirst!(splitpoints, 1)
    # Render initial state
    canvas = PDDLViz.new_canvas(renderer)
    anim_initialize!(canvas, renderer, domain, state;
                     show_inventory, trail_length,
                     track_stopmarker, kwargs...)
    # Set inventory title font size 
    if show_inventory
        canvas.blocks[2].titlesize = inventory_titlesize
    end
    # Animate trajectory segments between each splitpoint
    anims = PDDLViz.Animation[]
    for i in 1:length(splitpoints)-1
        a, b = splitpoints[i:i+1]
        anim = anim_trajectory!(canvas, renderer, domain, trajectory[a:b];
                                framerate, format, loop, show=false,
                                trail_length, track_stopmarker, kwargs...)
        if !isnothing(basepath)
            save("$(basepath)_$i.$format", anim)
        end
        push!(anims, anim)
    end
    return anims
end

function generate_stim_anim_segments(
    basepath::Union{AbstractString, Nothing},
    domain::Domain,
    problem::Problem,
    plan::AbstractVector{<:Term},
    splitpoints::AbstractVector{Int};
    kwargs...
)
    state = initstate(domain, problem)
    anims = generate_stim_anim_segments(
        basepath, domain, state, plan, splitpoints; kwargs...
    )
    return anims
end

"Generate stimuli JSON metadata."
function generate_stim_json(
    name::String,
    problem::Problem,
    goal:: Int,
    splitpoints::AbstractVector{Int},
    statements::AbstractVector{<:AbstractString},
    key_colors::AbstractVector{<:Term} = Const[],
    plan_boxes::AbstractVector{<:Term} = Const[]
)
    times = splitpoints .+ 1
    n_images = length(splitpoints)
    images = ["stimuli/segments/p$(name)_$i.gif" for i in 1:n_images]
    # goal = parse(Int, string(problem.goal.args[2])[end])
    key_colors = string.(key_colors)
    box_ids = [parse(Int, string(box)[end]) for box in plan_boxes]
    json = (
        name = name,
        goal = goal,
        images = images,
        times = times,
        statements = statements,
        relevant_colors = key_colors,
        relevant_boxes = box_ids,
        length = length(splitpoints)-1,
    )
    return json
end

# Define directory paths
EXPERIMENT_ID = "exp2_current"
PROBLEM_DIR = joinpath(@__DIR__, "dataset", "problems")
PLAN_DIR = joinpath(@__DIR__, "dataset", "plans", EXPERIMENT_ID)
STATEMENT_DIR = joinpath(@__DIR__, "dataset", "statements", EXPERIMENT_ID)
STIMULI_DIR = joinpath(@__DIR__, "dataset", "stimuli")

# Load domain
domain = load_domain(joinpath(@__DIR__, "dataset", "domain.pddl"))

## Generate animations for single plan / stimulus

# Load problem
problem_path = joinpath(PROBLEM_DIR, "demo.pddl")
problem = load_problem(problem_path)

# Load plan
plan_path = joinpath(PLAN_DIR, "demo.pddl")
plan, _, splitpoints = load_plan(plan_path) 
p_id = splitext(basename(plan_path))[1]

# Generate stimulus animation
mkpath(joinpath(STIMULI_DIR, "full"))
path = joinpath(STIMULI_DIR, "full", p_id == "demo" ? "demo.gif" : "p$p_id.gif")
anim = generate_stim_anim(path, domain, problem, plan;
                          format="gif", framerate=3)

# Generate completion of plan
planner = AStarPlanner(GoalManhattan())
state = initstate(domain, problem)
plan_end_state = PDDL.simulate(EndStateSimulator(), domain, state, plan)
sol = planner(domain, plan_end_state, problem.goal)
completion = collect(sol)
full_plan = [plan; completion]

# Generate stimulus animation segments
mkpath(joinpath(STIMULI_DIR, "segments"))
basepath = joinpath(STIMULI_DIR, "segments", p_id == "demo" ? "demo" : "p$p_id")
anims = generate_stim_anim_segments(
    basepath, domain, problem, full_plan, splitpoints;
    format="gif", framerate=3
)

## Generate all animations (except demo)

mkpath(STIMULI_DIR)
mkpath(joinpath(STIMULI_DIR, "full"))
mkpath(joinpath(STIMULI_DIR, "segments"))

problem_paths = readdir(PROBLEM_DIR, join=true)
filter!(p -> !startswith(basename(p), "demo") && endswith(p, ".pddl"), problem_paths)

plan_paths = readdir(PLAN_DIR, join=true)
filter!(p -> !startswith(basename(p), "demo") && endswith(p, ".pddl"), plan_paths)

all_metadata = []

planner = AStarPlanner(GoalManhattan())

for (problem_path, plan_path) in collect(zip(problem_paths, plan_paths))
    # Load problem
    problem = load_problem(problem_path)
    # Load plan
    plan, _, splitpoints = load_plan(plan_path) 
    p_id = splitext(basename(plan_path))[1]
    println("Generating stimuli for $(p_id) ...")
    # Load statements
    statements = readlines(joinpath(STATEMENT_DIR, "$(p_id).txt"))
    # Generate stimulus animation
    path = joinpath(STIMULI_DIR, "full", "p$p_id.gif")
    generate_stim_anim(path, domain, problem, plan)
    # Generate plan completion
    state = initstate(domain, problem)
    plan_end_state = PDDL.simulate(EndStateSimulator(), domain, state, plan)
    sol = planner(domain, plan_end_state, problem.goal)
    completion = collect(sol)
    full_plan = [plan; completion]
    # Extract relevant keys and boxes
    key_colors, _, plan_boxes =
        extract_relevant_keys_and_boxes(domain, plan_end_state, completion)
    # Generate stimulus animation segments
    basepath = joinpath(STIMULI_DIR, "segments", "p$p_id")
    generate_stim_anim_segments(basepath, domain, problem,
                                full_plan, splitpoints)
    # Generate stimuli JSON metadata
    json = generate_stim_json(p_id, problem, splitpoints, statements,
                              key_colors, plan_boxes)
    push!(all_metadata, json)
    # Sleep for 0.5 seconds
    sleep(0.5)
end

# Save JSON metadata
metadata_path = joinpath(STIMULI_DIR, "stimuli.json")
open(metadata_path, "w") do io
    JSON3.pretty(io, all_metadata, JSON3.AlignmentContext(indent=2))
end

## Generate stimuli ordering ##

Random.seed!(0)

mutexes = [[1, 2], [3, 6], [4, 5], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]
mutex_choices_1 =
    [[fill(1, n); fill(2, length(mutexes)-n)] for n in 1:length(mutexes)]
mutex_choices_2 =
    [[fill(2, n); fill(1, length(mutexes)-n)] for n in 1:length(mutexes)]
mutex_choices =
    reduce(vcat, [[c1, c2] for (c1, c2) in zip(mutex_choices_1, mutex_choices_2)])

stimuli_orders = map(mutex_choices) do choices
    idxs = [m[is] for (m, is) in zip(mutexes, choices)]
    return shuffle!(reduce(vcat, idxs))
end

for (i, order) in enumerate(stimuli_orders)
    idx = rand(2:6)
    control_idx = mod(i, 2) == 0 ? 17 : 18
    insert!(order, idx, control_idx)
end

stimuli_orders_str = replace(string(stimuli_orders), "], [" => "],\n [")
println(stimuli_orders_str)