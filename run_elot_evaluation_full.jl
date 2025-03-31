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
PROBLEM_DIR = joinpath(@__DIR__, "dataset", "problems")
PLAN_DIR = joinpath(@__DIR__, "dataset", "plans", "exp1")
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
PLAN_IDS, PLANS, _, SPLITPOINTS = load_plan_dataset(PLAN_DIR)

# Load belief statements
STATEMENTS_DIR = joinpath(@__DIR__, "dataset", "statements", "exp1")
CURR_STATEMENTS_PATH = joinpath(STATEMENTS_DIR, "exp1_current_in_vs_out.csv")
CURR_STATEMENTS_DF = CSV.read(CURR_STATEMENTS_PATH, DataFrame)
INIT_STATEMENTS_PATH = joinpath(STATEMENTS_DIR, "exp1_initial_in_vs_out.csv")
INIT_STATEMENTS_DF = CSV.read(INIT_STATEMENTS_PATH, DataFrame)

# Load threshold values
THRESHOLDS_PATH = joinpath(RESULTS_DIR, "best_thresholds.json")
belief_thresholds = OrderedDict{Symbol, Float64}(JSON3.read(read(THRESHOLDS_PATH, String))[:thresholds])
belief_threshold_values = collect(values(belief_thresholds))

## Run main experiment loop ##

# Select inference runs to evaluate
RUN_DIR = joinpath(RESULTS_DIR, "runs")
RUN_IDS = [
    "btom" => "uniform-uniform-uniform-3-qmdp-optimal-0.354",
    "true_belief" => "uniform-uniform-true_belief-3-qmdp-optimal-0.354",
    "non_instrumental" => "uniform-uniform-uniform-3-qmdp-manhattan-0.354",
]

for (run_name, run_id) in RUN_IDS
    println("===== Run $run_id =====")
    println()

    # Construct output dataframes for this run
    curr_output_path = joinpath(RESULTS_DIR, "exp1", "exp1_current_$(run_name).csv")
    curr_output_df = copy(CURR_STATEMENTS_DF)
    curr_output_gdf = DataFrames.groupby(curr_output_df, :plan_id)
    init_output_path = joinpath(RESULTS_DIR, "exp1", "exp1_initial_$(run_name).csv")
    init_output_df = copy(INIT_STATEMENTS_DF)
    init_output_gdf = DataFrames.groupby(init_output_df, :plan_id)

    for plan_id in PLAN_IDS
        println("=== Plan $plan_id ===")
        println()

        # Load inference result for plan
        json_path = joinpath(RUN_DIR, run_id, "$plan_id.json")
        json = JSON3.read(read(json_path, String), InferenceResult)
        config = json.config

        # Load problem, plan, and splitpoints
        problem = PROBLEMS[plan_id]
        plan = PLANS[plan_id]
        splitpoints = SPLITPOINTS[plan_id]

        # Load statements and translations
        curr_statements_df = filter(r -> r.plan_id == plan_id, CURR_STATEMENTS_DF)
        curr_statements = curr_statements_df.statement
        curr_formulas = curr_statements_df.formula

        init_statements_df = filter(r -> r.plan_id == plan_id, INIT_STATEMENTS_DF)
        init_statements = init_statements_df.statement
        init_formulas = init_statements_df.formula

        # Select output dataframes
        curr_output_sub_df = curr_output_gdf[(plan_id=plan_id,)]
        init_output_sub_df = init_output_gdf[(plan_id=plan_id,)]

        # Display statements and translations
        println("Current Belief Statements:")
        for stmt in curr_statements
            println("  ", stmt)
        end
        println()
        println("Translations:")
        for stmt in curr_formulas
            println("  ", stmt)
        end
        println("\n")

        println("Initial Belief Statements:")
        for stmt in init_statements
            println("  ", stmt)
        end
        println()
        println("Translations:")
        for stmt in init_formulas
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

        # Set evaluation time
        t_eval = splitpoints[1]
        println("Evaluation time: $t_eval\n")

        # Evaluate current belief statement probabilities
        println("Evaluating current belief statement probabilities...")
        curr_formulas =
            [tryparse_formula(domain, state, stmt) for stmt in curr_formulas]
        statement_probs = map(curr_formulas) do stmt
            isnothing(stmt) && return 0.5
            t = t_eval
            t_query = t
            env_states = @view(trace_env_hists[:, t_query + 1])
            belief_states = @view(trace_belief_hists[:, t_query + 1])
            log_weights = @view(log_weight_hist[:, t + 1])
            try
                return eval_epistemic_formula_prob(
                    domain, env_states, belief_states, log_weights, stmt;
                    normalize_prior = false, thresholds = belief_thresholds
                )
            catch
                return 0.5
            end
        end
        curr_output_sub_df.statement_probs .= statement_probs

        norm_statement_probs = map(curr_formulas) do stmt
            isnothing(stmt) && return 0.5
            t = t_eval
            t_query = t
            env_states = trace_env_hists[:, t_query + 1]
            belief_states = trace_belief_hists[:, t_query + 1]
            log_weights = log_weight_hist[:, t + 1]
            try
                return eval_epistemic_formula_prob(
                    domain, env_states, belief_states, log_weights, stmt;
                    normalize_prior = true, thresholds = belief_thresholds
                )
            catch
                return 0.5
            end
        end
        curr_output_sub_df.norm_statement_probs .= norm_statement_probs

        # Evaluate initial belief statement probabilities
        println("Evaluating initial belief statement probabilities...")
        init_formulas =
            [tryparse_formula(domain, state, stmt) for stmt in init_formulas]
        statement_probs = map(init_formulas) do stmt
            isnothing(stmt) && return 0.5
            t = t_eval
            t_query = 0
            env_states = trace_env_hists[:, t_query + 1]
            belief_states = trace_belief_hists[:, t_query + 1]
            log_weights = log_weight_hist[:, t + 1]
            try
                return eval_epistemic_formula_prob(
                    domain, env_states, belief_states, log_weights, stmt;
                    normalize_prior = false, thresholds = belief_thresholds
                )
            catch
                return 0.5
            end
        end
        init_output_sub_df.statement_probs .= statement_probs

        norm_statement_probs = map(init_formulas) do stmt
            isnothing(stmt) && return 0.5
            t = t_eval
            t_query = 0
            env_states = trace_env_hists[:, t_query + 1]
            belief_states = trace_belief_hists[:, t_query + 1]
            log_weights = log_weight_hist[:, t + 1]
            try
                return eval_epistemic_formula_prob(
                    domain, env_states, belief_states, log_weights, stmt;
                    normalize_prior = true, thresholds = belief_thresholds
                )
            catch
                return 0.5
            end
        end
        init_output_sub_df.norm_statement_probs .= norm_statement_probs

        println()
        CSV.write(curr_output_path, curr_output_df)
        CSV.write(init_output_path, init_output_df)
    end

    GC.gc()
end

