using PDDL
using Metatheory
using Gen, GenParticleFilters

using Metatheory: TermInterface
using GenParticleFilters: softmax

# Register max and min functions with PDDL
PDDL.@register(:function, :max, max)
PDDL.@register(:function, :min, min)

BELIEF_RULES = @theory begin
    # Knowledge attributions
    knows_that(~agent, formula(~phi)) -->
        and(believes(~agent, formula(~phi)), ~phi)
    knows_if(~agent, formula(~phi)) -->
        or(and(believes(~agent, formula(~phi)), ~phi),
           and(believes(~agent, formula(not(~phi))), not(~phi)))
    knows_about(~agent, ~cond, ~phi) -->
        exists(~cond, and(prob_of(~agent, ~phi) >= threshold(:believes), ~phi))
    not_knows_that(~agent, formula(~phi)) -->
        and(not(believes(~agent, formula(~phi))), ~phi)
    not_knows_if(~agent, formula(~phi)) -->
        and(or(not(believes(~agent, formula(~phi))), not(~phi)),
            or(not(believes(~agent, formula(not(~phi))), not(~phi))))
    not_knows_about(~agent, ~cond, ~phi) -->
        forall(~cond, or(prob_of(~agent, ~phi) < threshold(:believes), not(~phi)))
    # Hope attributions
    hopes(~agent, formula(~phi)) -->
        (prob_of(~agent, ~phi) >= threshold(:hopes))
    # Belief attributions
    believes(~agent, ~phi, ~psi) -->
        and(believes(~agent, ~phi), believes(~agent, ~psi))
    believes(~agent, formula(~phi)) -->
        (prob_of(~agent, ~phi) >= threshold(:believes))
    believes(~agent, ~f) -->
        apply(~f, ~agent)
    # Certainty attributions
    certain(~agent, formula(~phi)) -->
        (prob_of(~agent, ~phi) >= threshold(:certain))
    certain(~agent, ~f) -->
        apply(~f, ~agent)
    certain_about(~agent, ~cond, ~phi) -->
        exists(~cond, (prob_of(~agent, ~phi) >= threshold(:certain)))
    uncertain_if(~agent, formula(~phi)) -->
        and(prob_of(~agent, ~phi) < threshold(:uncertain),
            prob_of(~agent, ~phi) < threshold(:uncertain))
    uncertain_if(~agent, formula(~phi), formula(~psi)) -->
        and(prob_of(~agent, ~phi) < threshold(:uncertain),
            prob_of(~agent, ~psi) < threshold(:uncertain))
    uncertain_if(~agent, ~cond, ~phi, ~psi) -->
        exists(~cond,
            and(prob_of(~agent, ~phi) < threshold(:uncertain),
                prob_of(~agent, ~psi) < threshold(:uncertain)))
    uncertain_about(~agent, ~cond, ~phi) -->
        forall(~cond, (prob_of(~agent, ~phi) < threshold(:uncertain)))    
    # Comparatives (more, most, less, least)
    apply(more(~adj, ~phi, ~psi), ~agent) -->
        (degree(~adj, ~agent, ~phi) > degree(~adj, ~agent, ~psi))
    apply(more(~adj, ~cond, ~phi, ~psi), ~agent) -->
        exists(~cond, (degree(~adj, ~agent, ~phi) > degree(~adj, ~agent, ~psi)))
    apply(most(~adj, ~obj, ~var, ~cond, ~phi), ~agent) -->
        (degree(~adj, ~agent, substitute(~phi, ~var, ~obj)) >=
         maximum(~cond, degree(~adj, ~agent, ~phi)))
    apply(most(~adj, ~phi), ~agent) -->
        (degree(~adj, ~agent, ~phi) >= threshold(most(~adj)))
    apply(less(~adj, ~phi, ~psi), ~agent) -->
        (degree(~adj, ~agent, ~phi) < degree(~adj, ~agent, ~psi))
    apply(less(~adj, ~cond, ~phi, ~psi), ~agent) -->
        exists(~cond, (degree(~adj, ~agent, ~phi) < degree(~adj, ~agent, ~psi)))
    apply(least(~adj, ~obj, ~var, ~cond, ~phi), ~agent) -->
        (degree(~adj, ~agent, substitute(~phi, ~var, ~obj)) <=
         minimum(~cond, degree(~adj, ~agent, ~phi)))
    apply(equal(~adj, ~phi, ~psi), ~agent) -->
         (degree(~adj, ~agent, ~phi) ==  degree(~adj, ~agent, ~psi))
    apply(equal(~adj, ~theta, ~phi, ~psi), ~agent) -->
        and(==(degree(~adj, ~agent, ~theta), degree(~adj, ~agent, ~phi)), 
            ==(degree(~adj, ~agent, ~phi), degree(~adj, ~agent, ~psi)))
     # Possibility modals (could, can, might, may, must)
    apply(could(~phi), ~agent) -->
        (prob_of(~agent, ~phi) >= threshold(:could))
    apply(can(~phi), ~agent) -->
        (prob_of(~agent, ~phi) >= threshold(:can))
    apply(might(~phi), ~agent) -->
        (prob_of(~agent, ~phi) >= threshold(:might))
    apply(may(~phi), ~agent) -->
        (prob_of(~agent, ~phi) >= threshold(:may))
    apply(should(~phi), ~agent) -->
        (prob_of(~agent, ~phi) >= threshold(:should))
    apply(must(~phi), ~agent) -->
        (prob_of(~agent, ~phi) >= threshold(:must))
    # Probability adjectives
    apply(likely(~phi), ~agent) -->
        (degree(:likely, ~agent, ~phi) >= threshold(:likely))
    apply(unlikely(~phi), ~agent) -->
        (degree(:unlikely, ~agent, ~phi) <= threshold(:unlikely))
    degree(:likely, ~agent, ~phi) -->
        prob_of(~agent, ~phi)
    degree(:unlikely, ~agent, ~phi) -->
        -prob_of(~agent, ~phi)
    # Variable substitution
    substitute(~E, ~x, ~y) => expr_substitute(~E, ~x, ~y)
end

BELIEF_REWRITER = Rewriters.Prewalk(Rewriters.Chain(BELIEF_RULES))

BELIEF_THRESHOLDS = Dict(
    :certain => 0.95,
    :uncertain => 0.50,
    :believes => 0.75,
    :hopes => 0.25,
    :could => 0.20,
    :can => 0.30,
    :might => 0.20,
    :may => 0.30,
    :should => 0.80,
    :must => 0.95,
    :likely => 0.60,
    :unlikely => 0.40
)

BELIEF_MODALS = [:certain, :uncertain, :believes, :hopes,
                 :could, :can, :might, :may, :should, :must,
                 :likely, :unlikely]
BELIEF_THRESHOLD_VALUES = [BELIEF_THRESHOLDS[m] for m in BELIEF_MODALS]

"Lazily maps `f` over `args`, returning the original vector if possible."
function lazy_map(f, args::AbstractVector)
    new_args, ident = nothing, true
    for (i, a) in enumerate(args)
        b = f(a)
        if ident && a !== b
            new_args = collect(args[1:i-1])
            ident = false
        end
        if !ident push!(new_args, b) end
    end
    return ident ? args : new_args
end

"Substitutes all occurrences of `var` with `val` in `expr`."
function expr_substitute(expr::Expr, var::Symbol, val)
    @assert TermInterface.istree(expr)
    head = TermInterface.operation(expr)
    args = TermInterface.arguments(expr)
    new_args = lazy_map(args) do arg
        expr_substitute(arg, var, val)
    end
    return args === new_args ? expr : Expr(:call, head, new_args...)
end
expr_substitute(expr::Symbol, var::Symbol, val) = expr == var ? val : expr

"Converts a Julia expression to a PDDL expression."
function expr_to_pddl(expr::Expr)
    head = TermInterface.operation(expr)
    args = TermInterface.arguments(expr)
    return Compound(head, map(expr_to_pddl, args))
end
expr_to_pddl(expr::Symbol) =
    isuppercase(string(expr)[1]) ? Var(expr) : Const(expr)

"""
    dequantify_formula(term::Term, domain::Domain, state::State)

Replaces quantified expressions with grounded expressions.
"""
function dequantify_formula(term::Term, domain::Domain, state::State)
    if term.name in (:forall, :exists, :maximum, :minimum)
        conds, query = PDDL.flatten_conjs(term.args[1]), term.args[2]
        # Add conditions to query if they are not types
        if any(!PDDL.is_type(c, domain) for c in conds)
            query = Compound(:and, Term[query])
            for cond in conds
                !PDDL.is_type(cond, domain) && push!(query.args, cond)
            end
            conds = filter(c -> PDDL.is_type(c, domain), conds)
        end
        # Dequantify query
        query = dequantify_formula(query, domain, state)
        # Accumulate list of ground terms
        stack = Term[]
        subterms = Term[query]
        for cond in conds
            # Swap array references
            stack, subterms = subterms, stack
            # Substitute all objects of each type
            type, var = cond.name, cond.args[1]
            objects = PDDL.get_objects(domain, state, type)
            while !isempty(stack)
                subterm = pop!(stack)
                for obj in objects
                    push!(subterms, PDDL.substitute(subterm, var, obj))
                end
            end
        end
        # Return conjunction / disjunction of ground terms
        if term.name == :forall
            return isempty(subterms) ? Const(true) : Compound(:and, subterms)
        elseif term.name == :exists
            return isempty(subterms) ? Const(false) : Compound(:or, subterms)
        elseif term.name == :maximum
            return isempty(subterms) ? Const(0.0) : Compound(:max, subterms)
        elseif term.name == :minimum
            return isempty(subterms) ? Const(1.0) : Compound(:min, subterms)
        else
            error("Unknown quantifier: $(term.name)")
        end
    elseif term isa Compound
        # Recursively dequantify subterms
        args = lazy_map(term.args) do a
            dequantify_formula(a, domain, state)
        end
        return args === term.args ? term : Compound(term.name, args)
    else
        return term
    end
end

"Return all free variables in a formula and infer their types."
get_typed_vars(term::Term, domain::Domain, state::State) = 
    get_typed_vars!(Dict{Var, Set{Symbol}}(), term, domain, state)

function get_typed_vars!(vars::Dict, term::Term, domain::Domain, state::State)
    if term isa Const
        return vars
    elseif term isa Var
        get!(vars, term, Set{Symbol}())
        return vars
    elseif term isa Compound
        if PDDL.is_fluent(term, domain)
            sig = PDDL.get_fluent(domain, term.name)
            for (arg, type) in zip(term.args, sig.argtypes)
                if arg isa Var
                    vartypes = get!(vars, arg, Set{Symbol}())
                    push!(vartypes, type)
                else
                    get_typed_vars!(vars, arg, domain, state)
                end
            end
        else
            for arg in term.args
                get_typed_vars!(vars, arg, domain, state)
            end
        end
        return vars
    end
end

"Bind free variables in a formula via an existential quantifier."
function bind_free_vars(term::Term, domain::Domain, state::State)
    typed_vars = get_typed_vars(term, domain, state)
    typeconds = Term[]
    for (var, types) in typed_vars
        if length(types) == 1
            type = first(types)
            push!(typeconds, Compound(type, Term[var]))
        elseif length(types) > 1
            subconds = [Compound(type, Term[var]) for type in types]
            push!(typeconds, Compound(:or, subconds))
        else
            error("Unknown type: $(var)")
        end
    end
    return Compound(:exists, Term[Compound(:and, typeconds), term])
end

"Ground formula by dequantifying and removing any free variables."
function ground_formula(term::Term, domain::Domain, state::State)
    term = dequantify_formula(term, domain, state)
    if !PDDL.is_ground(term)
        term = bind_free_vars(term, domain, state)
        term = dequantify_formula(term, domain, state)
    end
    return PDDL.simplify_statics(term, domain, state)
end

"Evaluates the probability of non-epistemic formula given an agent's belief state."
function eval_base_formula_prob(
    domain::Domain, belief_state::ParticleBeliefState, formula::Term
)
    prob = map(zip(belief_state.env_states, belief_state.probs)) do (s, p)
        p == 0.0 && return 0.0
        return satisfy(domain, s, formula) ? p : 0.0
    end |> sum
    return prob
end

function validate_base_formula(
    domain::Domain, belief_state::ParticleBeliefState, term::Term
)
    env_state = first(belief_state.env_states)
    if term isa Const
        if term.name isa Symbol
            return term in PDDL.get_objects(env_state)
        elseif term.name isa Union{Bool, Real}
            return true
        else
            return false
        end
    elseif term isa Var
        return true
    elseif term isa Compound
        if (PDDL.is_logical_op(term) || PDDL.is_pred(term, domain) ||
            PDDL.is_func(term, domain) || PDDL.is_type(term, domain) ||
            PDDL.is_global_pred(term) || PDDL.is_global_func(term))
            return all(validate_base_formula(domain, belief_state, arg)
                       for arg in term.args)
        else
            return false
        end
    end
    return false
end

"Replaces `prob_of` subterms in an epistemic formula with their values."
function eval_prob_of_subterms(
    domain::Domain, belief_state::ParticleBeliefState, term::Term
)
    if term.name == :prob_of
        agent, formula = term.args
        p = eval_base_formula_prob(domain, belief_state, formula)
        return Const(p)
    elseif term isa Compound
        args = lazy_map(term.args) do a
            eval_prob_of_subterms(domain, belief_state, a)
        end
        return args === term.args ? term : Compound(term.name, args)
    else
        return term
    end
end

function validate_prob_of_subterms(
    domain::Domain, belief_state::ParticleBeliefState, term::Term
)
    if term.name == :prob_of
        agent, formula = term.args
        return validate_base_formula(domain, belief_state, formula)
    elseif term isa Compound
        valid = all(validate_prob_of_subterms(domain, belief_state, arg)
                   for arg in term.args)
        if !valid
            println("Invalid subterm: $term")
        end
        return valid
    elseif term isa Const
        if term.name isa Symbol
            return term in PDDL.get_objects(env_state)
        elseif term.name isa Union{Bool, Real}
            return true
        else
            return false
        end
    elseif term isa Var
        return true
    end
end

"Replaces `threshold` subterms in an epistemic formula with their values."
function eval_threshold_subterms(
    term::Term, thresholds::AbstractDict{Symbol, <:Real};
    multipliers = Dict(:most => 1.5)
)
    if term.name == :threshold
        @assert length(term.args) == 1
        arg = term.args[1]
        if arg.name in keys(multipliers) # Multiplier adverbs
            @assert length(arg.args) == 1
            val = get(thresholds, arg.args[1].name, 0.0)
            mult = get(multipliers, arg.name, 1.0)
            val = min(1.0, val * mult)
        else
            val = get(thresholds, arg.name, 0.0)
        end
        return Const(val)
    elseif term isa Compound
        args = lazy_map(term.args) do a
            eval_threshold_subterms(a, thresholds)
        end
        return args === term.args ? term : Compound(term.name, args)
    else
        return term
    end
end

"Evaluates an epistemic formula in a environment-belief pair."
function eval_epistemic_formula(
    domain::Domain, env_state::State,
    belief_state::ParticleBeliefState, formula::Term;
    dequantified::Bool = false, threshold_free::Bool = false,
    thresholds = BELIEF_THRESHOLDS
)
    # Dequantify formula with respect to environment state
    if !dequantified
        formula = ground_formula(formula, domain, env_state)
    end
    # Replace `prob_of` and `threshold` terms with their resulting values
    formula = eval_prob_of_subterms(domain, belief_state, formula)
    if !threshold_free
        formula = eval_threshold_subterms(formula, thresholds)
    end
    # Evaluate whether resulting formula is true in the environment state
    return satisfy(domain, env_state, formula)
end

function eval_epistemic_formula(
    domain::Domain, env_state::State, belief_state::ParticleBeliefState,
    formula::Expr; kwargs...
)
    formula = expr_to_pddl(BELIEF_REWRITER(formula))
    return eval_epistemic_formula(domain, env_state, belief_state,
                                  formula; kwargs...)
end

function eval_epistemic_formula(
    domain::Domain, trace::Trace, formula::Term, t = nothing;
    dequantified::Bool = false, threshold_free::Bool = false,
    thresholds = BELIEF_THRESHOLDS
)
    # Determine timestep and extract environment and belief state
    t = isnothing(t) ? Gen.get_args(trace)[1] : t
    env_addr = t == 0 ?
        (:init => :env) : (:timestep => t => :env)
    env_state = trace[env_addr]
    belief_addr = t == 0 ?
        (:init => :agent => :belief) : (:timestep => t => :agent => :belief)
    belief_state = trace[belief_addr]
    return eval_epistemic_formula(
        domain, env_state, belief_state, formula;
        dequantified, threshold_free, thresholds
    )
end

function eval_epistemic_formula(
    domain::Domain, trace::Trace, formula::Expr, t = nothing;
    kwargs...
)
    formula = expr_to_pddl(BELIEF_REWRITER(formula))
    return eval_epistemic_formula(domain, trace, formula, t; kwargs...)
end

"Evaluates the probability of an epistemic formula in a particle filter state."
function eval_epistemic_formula_prob(
    domain::Domain, pf_state::ParticleFilterState, formula::Term, t = nothing;
    dequantified::Bool = false, thresholds = BELIEF_THRESHOLDS,
    normalize_prior::Bool = false
)
    if !dequantified
        env_state = pf_state.traces[1][:init => :env]
        formula = ground_formula(formula, domain, env_state)
    end
    formula = eval_threshold_subterms(formula, thresholds)
    vals = map(pf_state.traces) do trace
        eval_epistemic_formula(domain, trace, formula, t;
                               dequantified = true, threshold_free = true)
    end
    if normalize_prior
        # Assumes uniform prior over all traces
        formula_prior = sum(vals) ./ length(vals)
        formula_prior == 1.0 || formula_prior == 0.0 && return formula_prior
        log_weights = map(zip(vals, pf_state.log_weights)) do (v, w)
            if w == -Inf
                return w
            elseif v == true
                return w + log(0.5 / formula_prior)
            else
                return w + log(0.5 / (1 - formula_prior))
            end
        end
        probs = softmax(log_weights)
    else
        probs = get_norm_weights(pf_state)
    end
    formula_prob = sum(vals .* probs)
    return formula_prob
end

function eval_epistemic_formula_prob(
    domain::Domain, pf_state::ParticleFilterState, formula::Expr, t = nothing;
    kwargs...
)
    formula = expr_to_pddl(BELIEF_REWRITER(formula))
    return eval_epistemic_formula_prob(domain, pf_state, formula, t; kwargs...)
end

function eval_epistemic_formula_prob(
    domain::Domain,
    env_states::AbstractVector{<:State},
    belief_states::AbstractVector{<:ParticleBeliefState},
    log_weights::AbstractVector{<:Real},
    formula::Term;
    normalize_prior::Bool = false,
    dequantified::Bool = false,
    thresholds = BELIEF_THRESHOLDS,
)
    if !dequantified
        formula = ground_formula(formula, domain, env_states[1])
    end
    formula = eval_threshold_subterms(formula, thresholds)
    vals = map(eachindex(env_states)) do i
        log_weights[i] <= -1000.0 && return false
        eval_epistemic_formula(domain, env_states[i], belief_states[i], formula;
                               dequantified = true, threshold_free = true)
    end
    if normalize_prior
        # Assumes uniform prior over all traces
        formula_prior = sum(vals) ./ length(vals)
        formula_prior == 1.0 || formula_prior == 0.0 && return formula_prior
        log_weights = map(zip(vals, log_weights)) do (v, w)
            if w <= -1000.0
                return w
            elseif v == true
                return w + log(0.5 / formula_prior)
            else
                return w + log(0.5 / (1 - formula_prior))
            end
        end
        probs = softmax(log_weights)
    else
        probs = softmax(log_weights)
    end
    formula_prob = sum(vals .* probs)
    return formula_prob
end

"Validate an epistemic formula in a given environment state."
function validate_formula(
    domain::Domain, env_state::State, formula::Term;
)
    try 
        # Dequantify formula with respect to environment state
        formula = dequantify_formula(formula, domain, env_state)
        # Replace `prob_of` and `threshold` terms with their resulting values
        belief_state = ParticleBeliefState([env_state], [0.0])
        formula = eval_prob_of_subterms(domain, belief_state, formula)
        formula = eval_threshold_subterms(formula, BELIEF_THRESHOLDS)
        # Evaluate whether resulting formula is true in the environment state
        val = satisfy(domain, env_state, formula)
        return true
    catch
        return false
    end
    return true
end

function validate_formula(
    domain::Domain, env_state::State, formula::Expr;
)
    try
        formula = expr_to_pddl(BELIEF_REWRITER(formula))
        return validate_formula(domain, env_state, formula)
    catch
        return false
    end
    return true
end

function validate_formula(
    domain::Domain, env_state::State, formula::AbstractString;
)
    try
        formula = Meta.parse(formula)
        return validate_formula(domain, env_state, formula)
    catch
        return false
    end
end

function tryparse_formula(
    domain::Domain, env_state::State, formula::AbstractString;
)
    if validate_formula(domain, env_state, formula)
        return formula |> Meta.parse |> BELIEF_REWRITER |> expr_to_pddl
    else
        return nothing
    end
end

function tryparse_formula(
    domain::Domain, env_state::State, formula::Expr;
)
    if validate_formula(domain, env_state, formula)
        return formula |> BELIEF_REWRITER |> expr_to_pddl
    else
        return nothing
    end
end
