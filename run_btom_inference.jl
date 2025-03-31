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
EXPERIMENT_ID = "exp1"
PROBLEM_DIR = joinpath(@__DIR__, "dataset", "problems")
PLAN_DIR = joinpath(@__DIR__, "dataset", "plans", EXPERIMENT_ID)
RESULTS_DIR = joinpath(@__DIR__, "results")
mkpath(RESULTS_DIR)

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

# Load plans
PLAN_IDS, PLANS, _, _ = load_plan_dataset(PLAN_DIR)

## Define parameters ##

PARAMS = (
    # Goal, state, and belief priors
    GOAL_PRIOR = [:uniform], # [:uniform, :known_goal],
    STATE_PRIOR = [:uniform,], # [:uniform, :known_state],
    BELIEF_PRIOR = [:uniform], # [:uniform, :true_belief],
    N_AGENT_PARTICLES = [3],
    # Planner and policy parameters
    POLICY_TYPE = [:qmdp], # [:qmdp, :thompson]
    VALUE_FUNCTION = [:optimal], # [:optimal, :manhattan] 
    ACT_TEMPERATURE = [0.354], # [0.25, 0.354, 0.5, 0.707, 1.0, 1.414, 2.0],
)

## Run main experiment loop ##

for params in namedproduct(PARAMS)
    println("==== PARAMS ====")
    println()
    for (k, v) in pairs(params)
        println("$k = $v")
    end
    println()
    param_dir = joinpath(RESULTS_DIR, "runs", join(params, "-"))
    mkpath(param_dir)

    for plan_id in PLAN_IDS
        println("=== Plan $plan_id ===")
        println()

        # Load problem and plan
        problem = PROBLEMS[plan_id]
        plan = PLANS[plan_id]

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
                return Specification(goals[true_goal])
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
            initial_belief_dists =
                enumerate_belief_dists(length(initial_states), 
                                    params.N_AGENT_PARTICLES,
                                    no_zeros = true)
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
            action = (t, obs, pf) -> begin
                t == 0 && return write_pddl(PDDL.no_op)
                addr = (:timestep => t => :act => :act)
                act = Gen.has_value(obs, addr) ? obs[addr] : PDDL.no_op
                return write_pddl(act)
            end,
            verbose = true
        )
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
        callback = CombinedCallback(
            logger=logger_cb, particle_logger=particle_logger_cb
        )

        # Configure SIPS particle filter
        sips = SIPS(world_config, resample_cond=:none, rejuv_cond=:none)

        # Run particle filter
        n_particles = length(init_strata)
        pf_state = sips(
            n_particles, t_obs_iter;
            init_args = (init_strata = init_strata,),
            callback = callback
        );

        # Construct JSON dictionary from results
        header = OrderedDict(
            :plan_id => plan_id,
            :state_ids => [trace[init_state_addr] for trace in get_traces(pf_state)],
            :goal_ids => [trace[goal_addr] for trace in get_traces(pf_state)]
        )
        header[:config] = OrderedDict(
            :goal_prior => params.GOAL_PRIOR,
            :state_prior => params.STATE_PRIOR,
            :belief_prior => params.BELIEF_PRIOR,
            :n_agent_particles => params.N_AGENT_PARTICLES,
            :policy_type => params.POLICY_TYPE,
            :value_function => params.VALUE_FUNCTION,
            :act_temperature => params.ACT_TEMPERATURE
        )
        data = sort(particle_logger_cb.data, by = x -> (length(string(x)), x))
        data = merge(header, data)

        # Write out and read as JSON
        json_path = joinpath(RESULTS_DIR, param_dir, "$(plan_id).json")
        open(json_path, "w") do io
            JSON3.pretty(io, data)
        end

        # Clear heuristic cache and garbage collect
        if heuristic isa MemoizedHeuristic
            empty!(heuristic)
        end
        GC.gc()
        println()
    end
end
