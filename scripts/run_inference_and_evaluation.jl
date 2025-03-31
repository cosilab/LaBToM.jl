using PDDL, SymbolicPlanners
using Gen, GenParticleFilters
using InversePlanning
using CSV, DataFrames, Dates
using JSON3

# Register PDDL array theory
PDDL.Arrays.register!()

include("src/plan_io.jl")
include("src/utils.jl")
include("src/heuristics.jl")
include("src/beliefs.jl")
include("src/plans.jl")
include("src/actions.jl")
include("src/interpreter.jl")

"Helper function that iterates over Cartesian product of named arguments."
function namedproduct(args::NamedTuple{Ks}) where {Ks}
    args = map(x -> applicable(iterate, x) ? x : (x,), args)
    iter = (NamedTuple{Ks}(x) for x in Iterators.product(args...))
    return iter
end

# Define directory paths
EXPERIMENT_ID = "exp2_current"
PROJECT_DIR = joinpath(@__DIR__, "..")
PROBLEM_DIR = joinpath(PROJECT_DIR, "dataset", "problems")
PLAN_DIR = joinpath(PROJECT_DIR, "dataset", "plans", EXPERIMENT_ID)
STATEMENT_DIR = joinpath(PROJECT_DIR, "dataset", "statements", EXPERIMENT_ID)
RESULTS_DIR = joinpath(PROJECT_DIR, "results")
mkpath(RESULTS_DIR)

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
PLAN_IDS, PLANS, _, SPLITPOINTS = load_plan_dataset(PLAN_DIR)
_, STATEMENTS = load_statement_dataset(STATEMENT_DIR)

# Load translated statements
STATEMENT_JSON_PATH = joinpath(PROJECT_DIR, "dataset", "statements",
                               "$(EXPERIMENT_ID)_translations_cleaned.json")
STATEMENT_JSON = JSON3.read(read(STATEMENT_JSON_PATH, String))

## Define parameters ##

PARAMS = (
    BELIEF_TYPE = [:current], # [:current, :initial],
    # Goal, state, and belief priors
    GOAL_PRIOR = [:uniform], # [:uniform, :known_goal],
    STATE_PRIOR = [:uniform,], # [:uniform, :known_state],
    BELIEF_PRIOR = [:uniform,], # [:uniform, :true_belief],
    N_AGENT_PARTICLES = [3],
    # Planner and policy parameters
    POLICY_TYPE = [:qmdp], # [:qmdp, :thompson]
    VALUE_FUNCTION = [:optimal], # [:optimal, :manhattan] 
    ACT_TEMPERATURE = [1.0,],
    # Inference parameters
    INFER_METHOD = [:exact], # [:exact, :query_driven_smc, :smc],
    N_PARTICLES = [100],
    # Number of trials
    N_TRIALS = [1,], # 1,
)

## Run main experiment loop ##

df = DataFrame(
    # Plan info
    plan_id = String[],
    belief_type = String[],
    true_goal = Int[],
    trial_id = Int[],
    # Model parameters
    goal_prior = String[],
    state_prior = String[],
    belief_prior = String[],
    n_agent_particles = Int[],
    policy_type = String[],
    value_function = String[],
    act_temperature = Float64[],
    modal_thresholds = String[],
    # Inference parameters
    infer_method = String[],
    n_particles = Int[],    
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
df_path = joinpath(RESULTS_DIR, EXPERIMENT_ID, df_path)

for params in namedproduct(PARAMS)
    println("==== PARAMS ====")
    println()
    for (k, v) in pairs(params)
        println("$k = $v")
    end
    println()
    for plan_id in PLAN_IDS
        println("=== Plan $plan_id ===")
        println()

        # Load problem, plan, splitpoints, and statements
        problem = PROBLEMS[plan_id]
        splitpoints = SPLITPOINTS[plan_id]
        plan = PLANS[plan_id]
        statements = STATEMENTS[plan_id]
        logical_statements = map(STATEMENT_JSON[plan_id]) do stmt
            stmt |> Meta.parse |> BELIEF_REWRITER |> expr_to_pddl
        end

        # Display statements
        println("Statements:")
        for stmt in statements
            println("  ", stmt)
        end
        println()

        # Display logical statements
        println("Logical statements:")
        for stmt in logical_statements
            println("  ", stmt)
        end
        println()

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
        goal_names = ["tri", "square", "hex", "circle"]

        # Define prior over possible goals
        if params.GOAL_PRIOR == :uniform
            goal_ids = 1:length(goals)
            @gen function goal_prior()
                goal_id ~ uniform_discrete(1, length(goals))
                return Specification(goals[goal_id])
            end
        elseif params.GOAL_PRIOR == :known_goal
            goal_ids = [true_goal]
            goal_prior_probs = zeros(length(goals))
            goal_prior_probs[true_goal] = 1.0
            @gen function goal_prior()
                goal_id ~ categorical(goal_prior_probs)
                return Specification(goal[true_goal])
            end
        else
            error("Unknown goal prior: $(params.GOAL_PRIOR)")
        end

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

        # Define prior over possible initial states
        if params.STATE_PRIOR == :uniform
            state_ids = 1:length(initial_states)
            state_probs = fill(1/length(initial_states), length(initial_states))
            @gen function state_prior()
                state_id ~ uniform_discrete(1, length(initial_states))
                return initial_states[state_id]
            end
        elseif params.STATE_PRIOR == :known_state
            true_state_id = findfirst(==(state), initial_states)
            state_ids = [true_state_id]
            state_prior_probs = zeros(length(initial_states))
            state_prior_probs[true_state_id] = 1.0
            @gen function state_prior()
                state_id ~ categorical(state_prior_probs)
                return initial_states[state_id]
            end
        else
            error("Unknown state prior: $(params.STATE_PRIOR)")
        end

        # Set up particle belief configuration
        if params.BELIEF_PRIOR == :uniform
            initial_belief_dists = enumerate_belief_dists(
                length(initial_states), params.N_AGENT_PARTICLES,
                no_zeros = true
            )
            belief_config = ParticleBeliefConfig(
                domain, dkg_observation_model,
                initial_states, initial_belief_dists
            )
        elseif params.BELIEF_PRIOR == :true_belief
            belief_config = ParticleTrueBeliefConfig(
                domain, dkg_observation_model,
            )
        else
            error("Unknown belief prior: $(params.BELIEF_PRIOR)")
        end

        # Set up agent's planning configuration
        manhattan = GoalManhattan()
        if params.VALUE_FUNCTION == :manhattan
            heuristic = manhattan
        elseif params.VALUE_FUNCTION == :optimal
            heuristic = PlannerHeuristic(AStarPlanner(manhattan, max_nodes=2^15))
            heuristic = memoized(heuristic)
        else
            error("Invalid value function type: $(params.VALUE_FUNCTION)")
        end
        planner = RTDP(heuristic=heuristic, n_rollouts=0, max_depth=0)
        plan_config = ParticleBeliefPolicyConfig(domain, planner)

        # Define action noise model
        if params.POLICY_TYPE == :qmdp
            act_config = BoltzmannQMDPActConfig(params.ACT_TEMPERATURE, -100)
        elseif params.POLICY_TYPE == :thompson
            act_config = BoltzmannThompsonActConfig(params.ACT_TEMPERATURE)
        else
            error("Invalid policy type: $(params.POLICY_TYPE)")
        end

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

        # Construct iterator over initial choicemaps for stratified sampling
        init_state_addr = :init => :env => :state_id
        goal_addr = :init => :agent => :goal => :goal_id
        belief_addr = :init => :agent => :belief => :belief_id
        if params.BELIEF_PRIOR == :uniform
            init_strata = choiceproduct(
                (goal_addr, goal_ids),
                (init_state_addr, state_ids),
                (belief_addr, 1:length(initial_belief_dists))
            )
        else
            init_strata = choiceproduct(
                (goal_addr, goal_ids),
                (init_state_addr, state_ids)
            )
        end

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
        logger_cb = DataLoggerCallback(
            t = (t, pf) -> t::Int,
            goal_probs = pf -> probvec(pf, goal_addr, 1:n_goals)::Vector{Float64},
            statement_probs = (t, pf) -> begin
                t_query = params.BELIEF_TYPE == :current ? t : 0
                map(logical_statements) do stmt
                    eval_epistemic_formula_prob(domain, pf, stmt, t_query,
                                                normalize_prior = false)
                end::Vector{Float64}
            end,
            norm_statement_probs = (t, pf) -> begin
                t_query = params.BELIEF_TYPE == :current ? t : 0    
                map(logical_statements) do stmt
                    eval_epistemic_formula_prob(domain, pf, stmt, t_query,
                                                normalize_prior = true)
                end::Vector{Float64}
            end,
            lml_est = pf -> log_ml_estimate(pf)::Float64,
            action = (t, obs, pf) -> begin
                t == 0 && return write_pddl(PDDL.no_op)
                addr = (:timestep => t => :act => :act)
                act = Gen.has_value(obs, addr) ? obs[addr] : PDDL.no_op
                return write_pddl(act)
            end,
            verbose = true
        )

        # Configure SIPS particle filter
        sips = SIPS(world_config, resample_cond=:none, rejuv_cond=:none)

        # Iterate across trials
        n_trials = params.INFER_METHOD == :exact ? 1 : params.N_TRIALS
        for trial_id in 1:n_trials
            if params.INFER_METHOD != :exact
                println("--- Trial $trial_id of $n_trials ---")
            end

            # Run particle filter
            empty!(logger_cb.data)
            n_particles = params.INFER_METHOD == :exact ?
                length(init_strata) : params.N_PARTICLES
            pf_state = sips(
                n_particles, t_obs_iter;
                init_args = (init_strata = init_strata,),
                callback = logger_cb
            );

            # Extract logged data
            timesteps = logger_cb.data[:t]
            goal_probs = reduce(hcat, logger_cb.data[:goal_probs])
            statement_probs = reduce(hcat, logger_cb.data[:statement_probs])
            norm_statement_probs = reduce(hcat, logger_cb.data[:norm_statement_probs])
            lml_est = logger_cb.data[:lml_est]

            actions = write_pddl.([PDDL.no_op; plan])
            is_judgment = [t in splitpoints && t != 0 && t != last(timesteps)
                           for t in timesteps]

            # Create and append dataframe
            T = length(timesteps)
            new_df = DataFrame(
                "plan_id" => fill(plan_id, T),
                "belief_type" => fill(params.BELIEF_TYPE, T),
                "true_goal" => fill(true_goal, T),
                "trial_id" => fill(trial_id, T),
                "goal_prior" => fill(params.GOAL_PRIOR, T),
                "state_prior" => fill(params.STATE_PRIOR, T),
                "belief_prior" => fill(params.BELIEF_PRIOR, T),
                "n_agent_particles" => fill(params.N_AGENT_PARTICLES, T),
                "policy_type" => fill(params.POLICY_TYPE, T),
                "value_function" => fill(params.VALUE_FUNCTION, T),    
                "act_temperature" => fill(params.ACT_TEMPERATURE, T),
                "modal_thresholds" => fill(string(BELIEF_THRESHOLD_VALUES), T),
                "infer_method" => fill(params.INFER_METHOD, T),
                "n_particles" => fill(n_particles, T),
                "timestep" => timesteps,
                "is_judgment" => is_judgment,
                "action" => actions,
                ("goal_probs_$i" => goal_probs[i, :] for i in 1:n_goals)...,
                "true_goal_probs" => goal_probs[true_goal, :],
                "lml_est" => lml_est,
                ("statement_probs_$i" => statement_probs[i, :] for i in 1:length(statements))...,
                ("norm_statement_probs_$i" => norm_statement_probs[i, :] for i in 1:length(statements))...,
            )
            append!(df, new_df, cols=:union)
            CSV.write(df_path, df)
            println()
        end

        # Clear heuristic cache and garbage collect
        if heuristic isa MemoizedHeuristic
            empty!(heuristic)
        end
        GC.gc()
    end
end
