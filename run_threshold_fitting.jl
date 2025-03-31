using PDDL, SymbolicPlanners
using Gen, GenParticleFilters
using InversePlanning
using CSV, DataFrames, Dates
using Statistics
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

## Initial setup ##

# Define directory paths
PROBLEM_DIR = joinpath(@__DIR__, "dataset", "problems")

CURR_PLAN_DIR = joinpath(@__DIR__, "dataset", "plans", "exp2_current")
CURR_STATEMENT_DIR = joinpath(@__DIR__, "dataset", "statements", "exp2_current")

INIT_PLAN_DIR = joinpath(@__DIR__, "dataset", "plans", "exp2_initial")
INIT_STATEMENT_DIR = joinpath(@__DIR__, "dataset", "statements", "exp2_initial")

RESULTS_DIR = joinpath(@__DIR__, "results")

# Load domain
DOMAIN = load_domain(joinpath(@__DIR__, "dataset", "domain.pddl"))
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
    CSV.read(joinpath(@__DIR__, "dataset", "statements", "exp2_current_translations.csv"), DataFrame)
CURR_TRANSLATIONS_GDF = DataFrames.groupby(CURR_TRANSLATIONS_DF, :plan_id)
CURR_TRANSLATIONS = Dict{String, Vector{String}}(
    key.plan_id => df.gold_elot for (key, df) in pairs(CURR_TRANSLATIONS_GDF)
)
INIT_TRANSLATIONS_DF =
    CSV.read(joinpath(@__DIR__, "dataset", "statements", "exp2_initial_translations.csv"), DataFrame)
INIT_TRANSLATIONS_GDF = DataFrames.groupby(INIT_TRANSLATIONS_DF, :plan_id)
INIT_TRANSLATIONS = Dict{String, Vector{String}}(
    key.plan_id => df.gold_elot for (key, df) in pairs(INIT_TRANSLATIONS_GDF)
)

# Read human data
df_path = joinpath(RESULTS_DIR, "humans", "exp2_current", "mean_human_data.csv")
curr_human_df = CSV.read(df_path, DataFrame)
sort!(curr_human_df, [:plan_id, :timestep])
curr_human_df.method .= "human"
curr_human_df.judgment_id = 1:size(curr_human_df, 1)

df_path = joinpath(RESULTS_DIR, "humans", "exp2_initial", "mean_human_data.csv")
init_human_df = CSV.read(df_path, DataFrame)
sort!(init_human_df, [:plan_id, :timestep])
init_human_df.method .= "human"
init_human_df.judgment_id = 1:size(init_human_df, 1)

human_df = vcat(curr_human_df, init_human_df)
human_statement_probs = Matrix(human_df[:, r"^statement_probs_(\d+)$"])

## Run optimization loop ##

RUN_DIR = joinpath(RESULTS_DIR, "runs")
RUN_ID = "uniform-uniform-uniform-3-qmdp-optimal-0.354"
# RUN_ID = "uniform-uniform-true_belief-3-qmdp-optimal-0.354"
# RUN_ID = "uniform-uniform-uniform-3-qmdp-manhattan-0.354"

OPTIM_DIR = 1 # Whether to maximize (1) or minimize (-1) the objective

function threshold_range(center, max_steps=4, step=0.05)
    min_threshold = max(center - max_steps * step, 0.0)
    max_threshold = min(center + max_steps * step, 1.0)
    return vcat(collect(center:step:max_threshold), collect(min_threshold:step:center-step))
end

init_threshold_values = BELIEF_THRESHOLD_VALUES
best_threshold_values = copy(init_threshold_values)
best_human_model_cor = -Inf * OPTIM_DIR

all_correlations = Float64[]
all_threshold_values = Vector{Float64}[]

for (modal_id, modal_word) in enumerate(BELIEF_MODALS)
    println("===== Optimizing threshold for '$modal_word' ($modal_id) =====")
    println()

    for threshold in threshold_range(init_threshold_values[modal_id])
        # Skip inconsistent thresholds various modals
        threshold == 0.0 && modal_word != :unlikely && continue
        threshold == 1.0 && modal_word == :unlikely && continue
        threshold == 1.0 && modal_word == :uncertain && continue

        println("=== Threshold for `$modal_word`: $threshold ===")
        println()
        belief_threshold_values = copy(best_threshold_values)
        belief_threshold_values[modal_id] = threshold
        belief_thresholds = Dict(zip(BELIEF_MODALS, belief_threshold_values))

        curr_statement_probs = zeros(0, 5)
        init_statement_probs = zeros(0, 5)

        for plan_id in PLAN_IDS
            println("=== Plan $plan_id ===")
            println()

            # Load inference result for plan
            json_path = joinpath(RUN_DIR, RUN_ID, "$plan_id.json")
            json = JSON3.read(read(json_path, String), InferenceResult)
            config = json.config

            # Load problem, plan, splitpoints, and statements
            problem = PROBLEMS[plan_id]
            plan = PLANS[plan_id]
            curr_splitpoints = CURR_SPLITPOINTS[plan_id]
            init_splitpoints = INIT_SPLITPOINTS[plan_id]
            curr_statements = CURR_STATEMENTS[plan_id]
            init_statements = INIT_STATEMENTS[plan_id]
            curr_logical_statements = map(CURR_STATEMENT_JSON[plan_id]) do stmt
                stmt |> Meta.parse |> BELIEF_REWRITER |> expr_to_pddl
            end
            init_logical_statements = map(INIT_STATEMENT_JSON[plan_id]) do stmt
                stmt |> Meta.parse |> BELIEF_REWRITER |> expr_to_pddl
            end  

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

            # Reconstruct environment trajectories
            println("Reconstructing environment trajectories...")
            env_hists, trace_env_hists =
                reconstruct_env_hists(json, domain, initial_states)

            # Reconstruct belief trajectories
            println("Reconstructing belief trajectories...")
            trace_belief_hists = reconstruct_belief_hists(json, env_hists)

            # Extract particle weights over time
            log_weight_hist = reduce(hcat, json.log_weights)

            # Evaluate current belief statement probabilities
            println("Evaluating current belief statement probabilities...")
            ts = curr_splitpoints[2:end-1]
            norm_statement_probs = map(Iterators.product(ts, curr_logical_statements)) do (t, stmt)
                t_query = t
                env_states = trace_env_hists[:, t_query + 1]
                belief_states = trace_belief_hists[:, t_query + 1]
                log_weights = log_weight_hist[:, t + 1]
                return eval_epistemic_formula_prob(
                    domain, env_states, belief_states, log_weights, stmt;
                    normalize_prior = true, thresholds = belief_thresholds
                )
            end
            curr_statement_probs =
                vcat(curr_statement_probs, norm_statement_probs)

            # Evaluate initial belief statement probabilities
            println("Evaluating initial belief statement probabilities...")
            ts = init_splitpoints[2:end-1]
            norm_statement_probs = map(Iterators.product(ts, init_logical_statements)) do (t, stmt)
                t_query = 0
                env_states = trace_env_hists[:, t_query + 1]
                belief_states = trace_belief_hists[:, t_query + 1]
                log_weights = log_weight_hist[:, t + 1]
                return eval_epistemic_formula_prob(
                    domain, env_states, belief_states, log_weights, stmt;
                    normalize_prior = true, thresholds = belief_thresholds
                )
            end
            init_statement_probs =
                vcat(init_statement_probs, norm_statement_probs)
            println()
        end

        # Calculate correlation with human ratings
        model_statement_probs = vcat(curr_statement_probs, init_statement_probs)
        human_model_cor = cor(vec(human_statement_probs), vec(model_statement_probs))
        push!(all_correlations, human_model_cor)
        push!(all_threshold_values, belief_threshold_values)

        println("Correlation: $human_model_cor")
        if human_model_cor * OPTIM_DIR > best_human_model_cor * OPTIM_DIR
            println("New best threshold found!")
            best_human_model_cor = human_model_cor
            best_threshold_values = copy(belief_threshold_values)
            println()
            println("== Thresholds ==")
            for (id, modal_word) in enumerate(BELIEF_MODALS)
                println(rpad(string(modal_word) * ":", 16), best_threshold_values[id])
            end
        end
        println()

        GC.gc()
    end
end

all_threshold_jsons = map(zip(all_threshold_values, all_correlations)) do (ts, c)
    OrderedDict(
        :thresholds => OrderedDict(zip(BELIEF_MODALS, ts)),
        :correlation => c
    )
end
path = joinpath(RESULTS_DIR, "all_thresholds.json")
open(path, "w") do io
    JSON3.pretty(io, all_threshold_jsons)
end

best_threshold_values = all_threshold_values[argmax(all_correlations)]
best_thresholds = OrderedDict(zip(BELIEF_MODALS, best_threshold_values))
best_thresholds = OrderedDict(
    :correlation => maximum(all_correlations),
    :thresholds => best_thresholds
)
path = joinpath(RESULTS_DIR, "best_thresholds.json")
open(path, "w") do io
    JSON3.pretty(io, best_thresholds)
end

worst_threshold_values = all_threshold_values[argmin(all_correlations)]
worst_thresholds = OrderedDict(zip(BELIEF_MODALS, worst_threshold_values))
worst_thresholds = OrderedDict(
    :correlation => minimum(all_correlations),
    :thresholds => worst_thresholds
)
path = joinpath(RESULTS_DIR, "worst_thresholds.json")
open(path, "w") do io
    JSON3.pretty(io, worst_thresholds)
end
