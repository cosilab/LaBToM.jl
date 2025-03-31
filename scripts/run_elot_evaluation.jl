using PDDL, SymbolicPlanners
using Gen, GenParticleFilters
using InversePlanning
using CSV, DataFrames, Dates
using JSON3

using DataStructures: OrderedDict

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

# Define directory paths
PROJECT_DIR = joinpath(@__DIR__, "..")
PROBLEM_DIR = joinpath(PROJECT_DIR, "dataset", "problems")

CURR_PLAN_DIR = joinpath(PROJECT_DIR, "dataset", "plans", "exp2_current")
CURR_STATEMENT_DIR = joinpath(PROJECT_DIR, "dataset", "statements", "exp2_current")

INIT_PLAN_DIR = joinpath(PROJECT_DIR, "dataset", "plans", "exp2_initial")
INIT_STATEMENT_DIR = joinpath(PROJECT_DIR, "dataset", "statements", "exp2_initial")

RESULTS_DIR = joinpath(PROJECT_DIR, "results")

# Load domain
DOMAIN = load_domain(joinpath(PROJECT_DIR, "dataset", "domain.pddl"))
COMPILED_DOMAINS = Dict{String, Domain}()

# Load problems
PROBLEMS = Dict{String, Problem}()
for path in readdir(PROBLEM_DIR)
    name, ext = splitext(path)
    ext == ".pddl" || continue
    PROBLEMS[name] = load_problem(joinpath(PROBLEM_DIR, path))
end

# Load plans, judgement points, and belief statements
PLAN_IDS, PLANS, _, CURR_SPLITPOINTS = load_plan_dataset(CURR_PLAN_DIR)
_, CURR_STATEMENTS = load_statement_dataset(CURR_STATEMENT_DIR)

_, _, _, INIT_SPLITPOINTS = load_plan_dataset(INIT_PLAN_DIR)
_, INIT_STATEMENTS = load_statement_dataset(INIT_STATEMENT_DIR)

# Load translated statements
CURR_TRANSLATIONS_DF =
    CSV.read(joinpath(PROJECT_DIR, "dataset", "statements", "exp2_current_translations.csv"), DataFrame)
CURR_TRANSLATIONS_GDF = DataFrames.groupby(CURR_TRANSLATIONS_DF, :plan_id)
CURR_TRANSLATIONS = Dict{String, Vector{String}}(
    key.plan_id => df.translation_elot for (key, df) in pairs(CURR_TRANSLATIONS_GDF)
)
INIT_TRANSLATIONS_DF =
    CSV.read(joinpath(PROJECT_DIR, "dataset", "statements", "exp2_current_translations.csv"), DataFrame)
INIT_TRANSLATIONS_GDF = DataFrames.groupby(INIT_TRANSLATIONS_DF, :plan_id)
INIT_TRANSLATIONS = Dict{String, Vector{String}}(
    key.plan_id => df.translation_elot for (key, df) in pairs(INIT_TRANSLATIONS_GDF)
)

# Load threshold values
THRESHOLDS_PATH = joinpath(RESULTS_DIR, "best_thresholds.json")
belief_thresholds = OrderedDict{Symbol, Float64}(JSON3.read(read(THRESHOLDS_PATH, String))[:thresholds])
belief_threshold_values = collect(values(belief_thresholds))

## Run main experiment loop ##

df = DataFrame(
    # Plan info
    plan_id = String[],
    belief_type = String[],
    true_goal = Int[],
    # Model parameters
    goal_prior = String[],
    state_prior = String[],
    belief_prior = String[],
    n_agent_particles = Int[],
    policy_type = String[],
    value_function = String[],
    act_temperature = Float64[],
    modal_thresholds = String[],
    # Timesteps
    timestep = Int[],
    is_judgment = Bool[],
    action = String[],
    # Goal inference probabilities
    goal_probs_1 = Float64[],
    goal_probs_2 = Float64[],
    goal_probs_3 = Float64[],
    goal_probs_4 = Float64[],
    true_goal_probs = Float64[],
    lml_est = Float64[],
    # Belief statement probabilities
    statement_probs_1 = Float64[],
    statement_probs_2 = Float64[],
    statement_probs_3 = Float64[],
    statement_probs_4 = Float64[],
    statement_probs_5 = Float64[],
    # Normalized belief statement probabilities
    norm_statement_probs_1 = Float64[],
    norm_statement_probs_2 = Float64[],
    norm_statement_probs_3 = Float64[],
    norm_statement_probs_4 = Float64[],
    norm_statement_probs_5 = Float64[],
)
datetime = Dates.format(Dates.now(), "yyyy-mm-ddTHH-MM-SS")
df_types = eltype.(eachcol(df))
df_path = "results_btom_$datetime.csv"
df_path = joinpath(RESULTS_DIR, df_path)

# Select inference runs to evaluate
RUN_DIR = joinpath(RESULTS_DIR, "runs")
RUN_IDS = [
    # "uniform-uniform-uniform-3-qmdp-optimal-0.25",
    # "uniform-uniform-uniform-3-qmdp-optimal-0.354",
    # "uniform-uniform-uniform-3-qmdp-optimal-0.5",
    # "uniform-uniform-uniform-3-qmdp-optimal-0.707",
    # "uniform-uniform-uniform-3-qmdp-optimal-1.0",
    # "uniform-uniform-uniform-3-qmdp-optimal-1.414",
    # "uniform-uniform-uniform-3-qmdp-optimal-2.0",
    "uniform-uniform-uniform-3-qmdp-optimal-0.354",
    "uniform-uniform-true_belief-3-qmdp-optimal-0.354",
    "uniform-uniform-uniform-3-qmdp-manhattan-0.354",
]

for run_id in RUN_IDS
    println("===== Run $run_id =====")
    println()

    for plan_id in PLAN_IDS
        println("=== Plan $plan_id ===")
        println()

        # Load inference result for plan
        json_path = joinpath(RUN_DIR, run_id, "$plan_id.json")
        json = JSON3.read(read(json_path, String), InferenceResult)
        config = json.config

        # Load problem, plan, splitpoints, and statements
        problem = PROBLEMS[plan_id]
        plan = PLANS[plan_id]
        curr_splitpoints = CURR_SPLITPOINTS[plan_id]
        init_splitpoints = INIT_SPLITPOINTS[plan_id]
        curr_statements = CURR_STATEMENTS[plan_id]
        init_statements = INIT_STATEMENTS[plan_id]
        curr_logical_statements = map(CURR_TRANSLATIONS[plan_id]) do stmt
            stmt |> Meta.parse |> BELIEF_REWRITER |> expr_to_pddl
        end
        init_logical_statements = map(INIT_TRANSLATIONS[plan_id]) do stmt
            stmt |> Meta.parse |> BELIEF_REWRITER |> expr_to_pddl
        end

        # Display statements and translations
        println("Current Belief Statements:")
        for stmt in curr_statements
            println("  ", stmt)
        end
        println()
        println("Translations:")
        for stmt in curr_logical_statements
            println("  ", stmt)
        end
        println("\n")

        println("Initial Belief Statements:")
        for stmt in init_statements
            println("  ", stmt)
        end
        println()
        println("Translations:")
        for stmt in init_logical_statements
            println("  ", stmt)
        end
        println("\n")

        # Extract true goal index from problem
        true_goal = parse(Int, string(PLANS[plan_id][end].args[2])[end])

        # Compile domain for problem
        domain = get!(COMPILED_DOMAINS, plan_id) do
            println("Compiling domain for problem $plan_id...")
            state = initstate(DOMAIN, problem)
            domain, _ = PDDL.compiled(DOMAIN, state)
            return domain
        end

        # Initialize true environment state
        state = initstate(domain, problem)

        # Simulate trajectory and end state
        trajectory = PDDL.simulate(domain, state, plan)
        end_state = trajectory[end]

        # Specify possible goals
        goals = @pddl(
            "(has human gem1)",
            "(has human gem2)",
            "(has human gem3)",
            "(has human gem4)"
        )
        n_goals = length(goals)

        # Determine minimum and maximum count for each key color
        max_color_keys = count_necessary_key_colors(domain, state, goals)
        vis_color_keys = count_visible_key_colors(domain, state)
        min_color_keys = Dict(c => n - get(vis_color_keys, c, 0) for (c, n) in max_color_keys)

        # Enumerate over possible initial states
        initial_states = dkg_enumerate_possible_envs(
            state;
            colors = collect(keys(max_color_keys)),
            min_keys = 0, max_keys = 2,
            min_color_keys, max_color_keys
        )
        state_names = map(box_contents_str, initial_states)
        println("Initial states:", state_names, "\n")

        # Reconstruct environment trajectories
        println("Reconstructing environment trajectories...")
        env_hists, trace_env_hists =
            reconstruct_env_hists(json, domain, initial_states)

        # Reconstruct belief trajectories
        println("Reconstructing belief trajectories...")
        trace_belief_hists = reconstruct_belief_hists(json, env_hists)

        # Extract particle weights over time
        log_weight_hist = reduce(hcat, json.log_weights)

        # Extract other logged data
        timesteps = 0:length(plan)
        T = length(timesteps)
        lml_est = json.lml_est
        actions = write_pddl.([PDDL.no_op; plan])

        # Evaluate goal probabilities
        trace_goal_ids = json.goal_ids
        goal_probs = map(Iterators.product(timesteps, 1:length(goals))) do (t, goal_id)
            log_weights = log_weight_hist[:, t + 1]
            probs = softmax(log_weights)
            return sum(probs[trace_goal_ids .== goal_id])
        end

        # Evaluate current belief statement probabilities
        println("Evaluating current belief statement probabilities...")
        statement_probs = map(Iterators.product(timesteps, curr_logical_statements)) do (t, stmt)
            t_query = t
            env_states = @view(trace_env_hists[:, t_query + 1])
            belief_states = @view(trace_belief_hists[:, t_query + 1])
            log_weights = @view(log_weight_hist[:, t + 1])
            return eval_epistemic_formula_prob(
                domain, env_states, belief_states, log_weights, stmt;
                normalize_prior = false, thresholds = belief_thresholds
            )
        end
        norm_statement_probs = map(Iterators.product(timesteps, curr_logical_statements)) do (t, stmt)
            t_query = t
            env_states = trace_env_hists[:, t_query + 1]
            belief_states = trace_belief_hists[:, t_query + 1]
            log_weights = log_weight_hist[:, t + 1]
            return eval_epistemic_formula_prob(
                domain, env_states, belief_states, log_weights, stmt;
                normalize_prior = true, thresholds = belief_thresholds
            )
        end

        # Create and append dataframe for current belief statements
        is_judgment = [t in curr_splitpoints && t != 0 && t != last(timesteps)
                    for t in timesteps]
        new_df = DataFrame(
            "plan_id" => fill(plan_id, T),
            "belief_type" => fill("current", T),
            "true_goal" => fill(true_goal, T),
            "goal_prior" => fill(string(config["goal_prior"]), T),
            "state_prior" => fill(string(config["state_prior"]), T),
            "belief_prior" => fill(string(config["belief_prior"]), T),
            "n_agent_particles" => fill(config["n_agent_particles"], T),
            "policy_type" => fill(config["policy_type"], T),
            "value_function" => fill(config["value_function"], T),    
            "act_temperature" => fill(config["act_temperature"], T),
            "modal_thresholds" => fill(string(belief_threshold_values), T),
            "timestep" => timesteps,
            "is_judgment" => is_judgment,
            "action" => actions,
            ("goal_probs_$i" => goal_probs[:, i] for i in 1:n_goals)...,
            "true_goal_probs" => goal_probs[:, true_goal],
            "lml_est" => lml_est,
            ("statement_probs_$i" => statement_probs[:, i] for i in 1:length(curr_statements))...,
            ("norm_statement_probs_$i" => norm_statement_probs[:, i] for i in 1:length(curr_statements))...,
        )
        append!(df, new_df, cols=:union)

        # Evaluate initial belief statement probabilities
        println("Evaluating initial belief statement probabilities...")
        statement_probs = map(Iterators.product(timesteps, init_logical_statements)) do (t, stmt)
            t_query = 0
            env_states = trace_env_hists[:, t_query + 1]
            belief_states = trace_belief_hists[:, t_query + 1]
            log_weights = log_weight_hist[:, t + 1]
            return eval_epistemic_formula_prob(
                domain, env_states, belief_states, log_weights, stmt;
                normalize_prior = false, thresholds = belief_thresholds
            )
        end
        norm_statement_probs = map(Iterators.product(timesteps, init_logical_statements)) do (t, stmt)
            t_query = 0
            env_states = trace_env_hists[:, t_query + 1]
            belief_states = trace_belief_hists[:, t_query + 1]
            log_weights = log_weight_hist[:, t + 1]
            return eval_epistemic_formula_prob(
                domain, env_states, belief_states, log_weights, stmt;
                normalize_prior = true, thresholds = belief_thresholds
            )
        end

        # Create and append dataframe for initial belief statements
        is_judgment = [t in init_splitpoints && t != 0 && t != last(timesteps)
                    for t in timesteps]
        new_df = DataFrame(
            "plan_id" => fill(plan_id, T),
            "belief_type" => fill("initial", T),
            "true_goal" => fill(true_goal, T),
            "goal_prior" => fill(string(config["goal_prior"]), T),
            "state_prior" => fill(string(config["state_prior"]), T),
            "belief_prior" => fill(string(config["belief_prior"]), T),
            "n_agent_particles" => fill(config["n_agent_particles"], T),
            "policy_type" => fill(config["policy_type"], T),
            "value_function" => fill(config["value_function"], T),    
            "act_temperature" => fill(config["act_temperature"], T),
            "modal_thresholds" => fill(string(belief_threshold_values), T),
            "timestep" => timesteps,
            "is_judgment" => is_judgment,
            "action" => actions,
            ("goal_probs_$i" => goal_probs[:, i] for i in 1:n_goals)...,
            "true_goal_probs" => goal_probs[:, true_goal],
            "lml_est" => lml_est,
            ("statement_probs_$i" => statement_probs[:, i] for i in 1:length(init_statements))...,
            ("norm_statement_probs_$i" => norm_statement_probs[:, i] for i in 1:length(init_statements))...,
        )
        append!(df, new_df, cols=:union)

        CSV.write(df_path, df)
        println()
    end

    GC.gc()
end

## Append results ##

# all_df_path = "results_btom_all.csv"
# all_df_path = joinpath(RESULTS_DIR, all_df_path)
# all_df = CSV.read(all_df_path, DataFrame)
# append!(all_df, df)

# all_df = df
# sort!(all_df,
#     [:belief_type, # :submethod,
#      :goal_prior, :state_prior, :belief_prior, 
#      :policy_type, :value_function, :act_temperature,
#      :plan_id, :timestep]
# )
# gdf = DataFrames.groupby(all_df, 
#     [:belief_type,
#      :goal_prior, :state_prior, :belief_prior,
#      :policy_type, :value_function, :act_temperature]
# )
# for (key, group) in pairs(gdf)
#     if key.goal_prior == "uniform" && key.state_prior == "uniform" && key.belief_prior == "uniform" && key.value_function == "optimal"
#         group.submethod .= "full"
#     elseif key.goal_prior == "known_goal" && key.state_prior == "uniform" && key.belief_prior == "uniform" && key.value_function == "optimal"
#         group.submethod .= "known_goal"
#     elseif key.goal_prior == "uniform" && key.state_prior == "known_state" && key.belief_prior == "uniform" && key.value_function == "optimal"
#         group.submethod .= "known_state"
#     elseif key.goal_prior == "uniform" && key.state_prior == "uniform" && key.belief_prior == "true_belief" && key.value_function == "optimal"
#         group.submethod .= "true_belief"
#     elseif key.goal_prior == "uniform" && key.state_prior == "uniform" && key.belief_prior == "uniform" && key.value_function == "manhattan"
#         group.submethod .= "non_instrumental"
#     else
#         group.submethod .= string(key)
#     end
# end
# select!(all_df, circshift(names(all_df), 1))
# CSV.write(all_df_path, all_df)
