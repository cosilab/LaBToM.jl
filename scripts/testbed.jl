using PDDL, SymbolicPlanners
using Gen, GenParticleFilters
using InversePlanning
using JSON3
using CSV, DataFrames
# using PDDLViz, GLMakie

# Register PDDL array theory
PDDL.Arrays.register!()

include("src/plan_io.jl")
include("src/utils.jl")
include("src/heuristics.jl")
include("src/beliefs.jl")
include("src/plans.jl")
include("src/actions.jl")
include("src/interpreter.jl")
include("src/reconstruct.jl")
# include("src/render.jl")

# Define directory paths
EXPERIMENT_ID = "exp2_current"
PROJECT_DIR = joinpath(@__DIR__, "..")
PROBLEM_DIR = joinpath(PROJECT_DIR, "dataset", "problems")
PLAN_DIR = joinpath(PROJECT_DIR, "dataset", "plans", EXPERIMENT_ID)
STATEMENT_DIR = joinpath(PROJECT_DIR, "dataset", "statements", EXPERIMENT_ID)

TRANSLATIONS_DF =
    CSV.read(joinpath(PROJECT_DIR, "dataset", "statements", "$(EXPERIMENT_ID)_translations.csv"), DataFrame)
TRANSLATIONS_GDF = DataFrames.groupby(TRANSLATIONS_DF, :plan_id)
TRANSLATIONS = Dict{String, Vector{String}}(
    key.plan_id => df.translation_elot for (key, df) in pairs(TRANSLATIONS_GDF)
)

#--- Initial Setup ---#

# Load domain
domain = load_domain(joinpath(PROJECT_DIR, "dataset", "domain.pddl"))

# Load problem
p_id = "1_1"
problem = load_problem(joinpath(PROBLEM_DIR, "$(p_id).pddl"))
state = initstate(domain, problem)

# Load plan
plan, _, splitpoints = load_plan(joinpath(PLAN_DIR, "$(p_id).pddl"))

# Load belief statements and pre-translated logical forms
statements = load_statements(joinpath(STATEMENT_DIR, "$(p_id).txt"))
logical_statements = map(TRANSLATIONS["$(p_id)"]) do stmt
    stmt |> Meta.parse |> BELIEF_REWRITER |> expr_to_pddl
end

# Initialize and compile reference state
state = initstate(domain, problem)
domain, state = PDDL.compiled(domain, problem)

# Simulate trajectory and end state
trajectory = PDDL.simulate(domain, state, plan)
end_state = trajectory[end]

# Render initial state
# canvas = RENDERER(domain, state)

#--- Goal Inference Setup ---#

# Specify possible goals
goals = @pddl(
    "(has human gem1)",
    "(has human gem2)",
    "(has human gem3)",
    "(has human gem4)"
)
goal_names = ["tri", "square", "hex", "circle"]

# Define uniform prior over possible goals
@gen function goal_prior()
    goal_id ~ uniform_discrete(1, length(goals))
    return Specification(goals[goal_id])
end

# Determine minimum and maximum count for each key color
max_color_keys = count_necessary_key_colors(domain, state, goals)
vis_color_keys = count_visible_key_colors(domain, state)
min_color_keys = Dict(c => n - get(vis_color_keys, c, 0) for (c, n) in max_color_keys)

# Enumerate over possible initial states
initial_states = dkg_enumerate_possible_envs(
    state;
    colors = collect(keys(max_color_keys)),
    min_keys = 1, max_keys = 2,
    min_color_keys, max_color_keys
)
state_names = map(box_contents_str, initial_states)

# Define uniform prior over possible initial states
state_probs = fill(1/length(initial_states), length(initial_states))
@gen function state_prior()
    state_id ~ uniform_discrete(1, length(initial_states))
    return initial_states[state_id]
end

# Set up particle belief configuration with uniform prior over beliefs
n_agent_particles = 3 
initial_belief_dists =
    enumerate_belief_dists(length(initial_states), n_agent_particles;
                           no_zeros=true)
belief_names = map(initial_belief_dists) do dist
    counts = round.(Int, exp.(dist) .* n_agent_particles)
    return join(string.(counts))
end
belief_config = ParticleBeliefConfig(
    domain, dkg_observation_model,
    initial_states, initial_belief_dists
)

# Set up agent's planning configuration
manhattan = GoalManhattan()
heuristic = memoized(PlannerHeuristic(AStarPlanner(manhattan, max_nodes=2^15)))
planner = RTDP(heuristic=heuristic, n_rollouts=0, max_depth=0)
plan_config = ParticleBeliefPolicyConfig(domain, planner)

# Define action noise model
act_config = BoltzmannQMDPActConfig(1.0, -100)

# Define agent configuration
agent_config = AgentConfig(
    goal_config = StaticGoalConfig(goal_prior),
    belief_config = belief_config,
    plan_config = plan_config,
    act_config = act_config
)

# Define observation noise model
obs_params = ObsNoiseParams(
    (pddl"(forall (?k - key ?b - box) (inside ?k ?b))", 0.00),
    (pddl"(forall (?k - key ?c - color) (iscolor ?k ?c))", 0.00),
)
obs_params = ground_obs_params(obs_params, domain, state)
obs_terms = collect(keys(obs_params))

# Configure world model with agent configuration and initial state prior
world_config = WorldConfig(
    agent_config = agent_config,
    env_config = PDDLEnvConfig(domain, state_prior),
    obs_config = MarkovObsConfig(domain, obs_params)
)

## Run goal and belief inference ##

# Construct iterator over initial choicemaps for stratified sampling
init_state_addr = :init => :env => :state_id
goal_addr = :init => :agent => :goal => :goal_id
belief_addr = :init => :agent => :belief => :belief_id
init_strata = choiceproduct(
    (goal_addr, 1:length(goals)),
    (init_state_addr, 1:length(initial_states)),
    (belief_addr, 1:length(initial_belief_dists))
)

# Construct iterator over observation timesteps and choicemaps 
t_obs_iter = act_choicemap_pairs(plan)
for (t, choices) in t_obs_iter
    obs = dkg_observation_model(domain, trajectory[t + 1])
    for (term, val) in obs
        choices[:timestep => t => :obs => term] = val
    end
end

# Set up logging callback
n_goals = length(goals)
n_init_states = length(initial_states)
n_beliefs = length(initial_belief_dists)
belief_type = EXPERIMENT_ID == "exp2_current" ? :current : :initial
logger_cb = DataLoggerCallback(
    t = (t, pf) -> t::Int,
    goal_probs = pf -> probvec(pf, goal_addr, 1:n_goals)::Vector{Float64},
    state_probs = pf -> probvec(pf, init_state_addr, 1:n_init_states)::Vector{Float64},
    statement_probs = (t, pf) -> begin
        t_query = belief_type == :current ? t : 0    
        map(logical_statements) do stmt
            eval_epistemic_formula_prob(domain, pf, stmt, t_query,
                                        normalize_prior = true)
        end::Vector{Float64}
    end,
    action = (t, obs, pf) -> begin
        t == 0 && return write_pddl(PDDL.no_op)
        addr = (:timestep => t => :act => :act)
        act = Gen.has_value(obs, addr) ? obs[addr] : PDDL.no_op
        return write_pddl(act)
    end,
    verbose = true
)
silent_cb = DataLoggerCallback(
    belief_probs = pf -> probvec(pf, belief_addr, 1:n_beliefs)::Vector{Float64},
    belief_dists = (t, pf) -> begin
        belief_addr = t == 0 ?
            (:init => :agent => :belief) : (:timestep => t => :agent => :belief)
        idxs = 1:(n_goals * n_init_states):length(init_strata)
        belief_dists = map(get_traces(pf)[idxs]) do trace
            belief_state = trace[belief_addr]
            return replace(belief_state.log_weights, -Inf => -1000.0)            
        end
        return belief_dists::Vector{Vector{Float64}}
    end,
    verbose = false
)
callback = CombinedCallback(logger=logger_cb, silent=silent_cb)

# Configure SIPS particle filter
sips = SIPS(world_config, resample_cond=:none, rejuv_cond=:none)

# Run particle filter
n_samples = length(init_strata)
pf_state = sips(
    n_samples, t_obs_iter;
    init_args = (init_strata = init_strata,),
    callback = callback
);

# Extract goal inferences
goal_probs = reduce(hcat, callback.logger.data[:goal_probs])

# Extract initial state inferences
state_probs = reduce(hcat, callback.logger.data[:state_probs])

# Extract belief inferences
belief_probs = reduce(hcat, callback.silent.data[:belief_probs])

# Extract agent's belief states
belief_states = callback.silent.data[:belief_dists]

# Extract statement probabilities
statement_probs = reduce(hcat, callback.logger.data[:statement_probs])

## Visualize inferences ##

using PDDLViz, CairoMakie
using CairoMakie: FileIO
include("src/render.jl")

# Render trajectory segments as images
h, w = size(state[pddl"(walls)"])
img_paths = String[]
for judgment_id in 2:length(splitpoints)-1
    t_plot = splitpoints[judgment_id] + 1
    prev_t_plot = splitpoints[judgment_id - 1] + 1
            cfigure = Figure(resolution=(100*w, 100*h), backgroundcolor=:transparent)
    canvas = new_canvas(RENDERER, cfigure)
    render_state!(canvas, RENDERER, domain, trajectory[t_plot], show_inventory=false)
    render_trajectory!(canvas, RENDERER, domain, trajectory[prev_t_plot:t_plot],
                    track_stopmarker=' ', object_start_colors=[(:black, 0.25)], 
                    show_state=false)
    tmp_path = tempname(PROJECT_DIR) * ".png"
    push!(img_paths, tmp_path)
    save(tmp_path, cfigure)
end

# Render inferences under each segment
n_segments = length(img_paths)
figure = Figure(resolution=(1500 * n_segments, 2000),
                backgroundcolor=:transparent, figure_padding=0)
for (i, judgment_id) in enumerate(2:length(splitpoints)-1)
    t_plot = splitpoints[judgment_id] + 1
    ax = Axis(figure[1, i], xlabel= "t = $(t_plot)",
              xlabelsize=60, xlabelfont=:italic,
              backgroundcolor=:transparent)
    hidedecorations!(ax)
    hidespines!(ax)
    ax.xlabelvisible = true
    image!(ax, rotr90(FileIO.load(img_paths[i])))
    layout = GridLayout(figure[2, i])
    plot_step_inferences!(layout,
        state_probs[:, t_plot], 
        goal_probs[:, t_plot],
        belief_probs[:, t_plot],
        belief_states[t_plot];
        backgroundcolor = :white
    )
end

# Adjust row sizes
rowsize!(figure.layout, 1, Aspect(1, h/w))
rowsize!(figure.layout, 2, 600)
display(figure)

# Save figure
FIGURE_DIR = joinpath(PROJECT_DIR, "figures")
mkpath(FIGURE_DIR)
save(joinpath(FIGURE_DIR, "step_by_step_$(p_id).pdf"), figure)

# Remove temporary images
for path in img_paths
    rm(path)
end

## Save particle weights and values for post-hoc analysis ##

using DataStructures: OrderedDict

particle_logger_cb = DataLoggerCallback(
    t = (t, pf) -> t::Int,
    action = (t, obs, pf) -> begin
        t == 0 && return write_pddl(PDDL.no_op)
        addr = (:timestep => t => :act => :act)
        act = Gen.has_value(obs, addr) ? obs[addr] : PDDL.no_op
        return write_pddl(act)
    end,
    lml_est =
        pf -> log_ml_estimate(pf)::Float64,
    log_weights =
        pf -> replace(get_log_weights(pf), -Inf => -1000.0)::Vector{Float64},
    belief_dists = (t, pf) -> begin
        belief_addr = t == 0 ?
            (:init => :agent => :belief) : (:timestep => t => :agent => :belief)
        belief_dists = map(get_traces(pf)) do trace
            belief_state = trace[belief_addr]
            return replace(belief_state.log_weights, -Inf => -1000.0)            
        end
        return belief_dists::Vector{Vector{Float64}}
    end
)

# Configure SIPS particle filter
sips = SIPS(world_config, resample_cond=:none, rejuv_cond=:none) 

# Run particle filter
n_samples = length(init_strata)
pf_state = sips(
    n_samples, t_obs_iter;
    init_args = (init_strata = init_strata,),
    callback = particle_logger_cb
);

# Construct JSON dictionary from results
header = OrderedDict(
    :plan_id => p_id,
    :state_ids => [trace[init_state_addr] for trace in get_traces(pf_state)],
    :goal_ids => [trace[goal_addr] for trace in get_traces(pf_state)]
)
header[:config] = OrderedDict()
data = sort(particle_logger_cb.data, by = x -> (length(string(x)), x))
data = merge(header, data)

# Write out and read as JSON
json = JSON3.read(JSON3.write(data), InferenceResult)

# Reconstruct environment trajectories
initial_states = dkg_enumerate_possible_envs(
    state;
    colors = collect(keys(max_color_keys)),
    min_keys = 1, max_keys = 2,
    min_color_keys, max_color_keys
)
env_hists, trace_env_hists =
    reconstruct_env_hists(json, domain, initial_states)

# Reconstruct belief trajectories
trace_belief_hists = reconstruct_belief_hists(json, env_hists)

# Extract particle weights over time
log_weight_hist = reduce(hcat, json.log_weights)

# Evaluate statement probabilities post-hoc
times = splitpoints[2:end-1]
statement_probs = map(Iterators.product(times, logical_statements)) do (t, stmt)
    t_query = t
    env_states = trace_env_hists[:, t_query + 1]
    belief_states = trace_belief_hists[:, t_query + 1]
    log_weights = log_weight_hist[:, t + 1]
    return eval_epistemic_formula_prob(
        domain, env_states, belief_states, log_weights, stmt;
        normalize_prior = false
    )
end
norm_statement_probs = map(Iterators.product(times, logical_statements)) do (t, stmt)
    t_query = t
    env_states = trace_env_hists[:, t_query + 1]
    belief_states = trace_belief_hists[:, t_query + 1]
    log_weights = log_weight_hist[:, t + 1]
    return eval_epistemic_formula_prob(
        domain, env_states, belief_states, log_weights, stmt;
        normalize_prior = true
    )
end
