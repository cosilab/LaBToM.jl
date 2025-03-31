using PDDL, Printf
using SymbolicPlanners, InversePlanning
using Gen, GenParticleFilters
using PDDLViz, GLMakie, CairoMakie
using PDDLViz.Makie.FileIO
using CSV, DataFrames

include("src/utils.jl")
include("src/render.jl")
include("src/plan_io.jl")

PROJECT_DIR = joinpath(@__DIR__, "..")

"Adds a subplot to a storyboard with a line plot of probabilities."
function storyboard_prob_lines!(
    fig_or_pos::Union{Figure, GridPosition, GridSubposition}, probs, ts=Int[];
    names = ["Series $i" for i in size(probs)[1]],
    colors = PDDLViz.colorschemes[:vibrant][1:length(names)],
    show_legend = false, ts_linewidth = 1, ts_fontsize = 24,
    legend_title = "Legend Title",
    xlabel = "Time", ylabel = "Probability", ylimits = (0, 1),
    upper = nothing, lower = nothing, rangetype = :band,
    ax_args = (), kwargs...
)
    if fig_or_pos isa Figure
        n_rows, n_cols = size(fig_or_pos.layout)
        width, height = size(fig_or_pos.scene)
        grid_pos = fig_or_pos[n_rows+1, 1:n_cols]
    else
        grid_pos = fig_or_pos
    end
    # Add probability subplot
    if length(ts) == size(probs)[2]
        curves = [[Makie.Point2f(t, p) for (t, p) in zip(ts, probs[i, :])]
                  for i in 1:size(probs)[1]]
        ax, _ = series(
            grid_pos, curves;
            color = colors, labels = names,
            axis = (xlabel = xlabel, ylabel = ylabel,
                    limits=((first(ts)-0.5, last(ts)+0.5), ylimits), ax_args...),
            kwargs...
        )
    else
        ax, _ = series(
            grid_pos, probs,
            color = colors, labels = names,
            axis = (xlabel = xlabel, ylabel = ylabel,
                    limits=((1, size(probs, 2)), ylimits), ax_args...),
            kwargs...
        )
    end
    # Add legend to subplot
    if show_legend
        axislegend(legend_title, framevisible=false)
    end
    # Add vertical lines at timesteps
    if !isempty(ts)
        vlines!(ax, ts, color=:black, linestyle=:dash,
                linewidth=ts_linewidth)
    end
    # Add upper and lower bounds
    if !isnothing(upper) && !isnothing(lower)
        ts = ts == Int[] ? collect(1:size(probs, 2)) : ts
        for (j, (l, u)) in enumerate(zip(eachrow(lower), eachrow(upper)))
            if rangetype == :band
                bplt = band!(ax, ts, l, u, color=(colors[j], 0.2))
                translate!(bplt, 0, 0, -1)
            elseif rangetype == :rangebars
                lw = get(kwargs, :linewidth, 1.0)
                bplt = rangebars!(ax, ts, l, u, color=colors[j], linewidth=lw)
            end
        end
    end
    # Resize figure to fit new plot
    if fig_or_pos isa Figure
        rowsize!(fig_or_pos.layout, 1, Auto(1.0))
        rowsize!(fig_or_pos.layout, n_rows+1, Auto(0.3))
        resize!(fig_or_pos, width, trunc(Int, height * 1.35))
        return fig_or_pos
    else
        return ax
    end
end

"Adds a subplot to a storyboard with a stacked bar plot of probabilities."
function storyboard_prob_bars!(
    fig_or_pos::Union{Figure, GridPosition, GridSubposition},
    probs, ts=Int[];
    names = ["Series $i" for i in size(probs)[1]],
    colors = PDDLViz.colorschemes[:vibrant][1:length(names)],
    show_legend = false,
    legend_title = "Legend Title",
    xlabel = "Time", ylabel = "Probability", ylimits = (0, 1),
    upper = nothing, lower = nothing,
    barlabel_size = 24, barlabel_threshold = 0.6, barlabel_offset = 0.05,
    ax_args = (), kwargs...
)
    if fig_or_pos isa Figure
        n_rows, n_cols = size(fig_or_pos.layout)
        width, height = size(fig_or_pos.scene)
        grid_pos = fig_or_pos[n_rows+1, 1:n_cols]
    else
        grid_pos = fig_or_pos
    end
    # Add probability subplot
    ts = ts == Int[] ? collect(1:size(probs, 2)) : ts
    colors = repeat(colors, length(ts))
    group = repeat(1:size(probs)[1], length(ts))
    starts = repeat(ts, inner=size(probs, 1)) .- 0.5
    stops = vec(probs) .+ starts
    ax, _ = barplot(
        grid_pos, group, stops, fillto = starts, direction = :x,
        color = colors, labels = names, 
        axis = (xlabel = xlabel, ylabel = ylabel, yreversed = true,
                limits=((first(ts)-0.5, last(ts)+0.5), (nothing, nothing)),
                ax_args...),
    )
    # Add upper and lower bounds
    if !isnothing(upper) && !isnothing(lower)
        new_lower = vec(lower) .+ starts
        new_upper = vec(upper) .+ starts
        rangebars!(ax, group, new_lower, new_upper,
                   color=:black, whiskerwidth=10, direction=:x)
    end
    # Add bar labels 
    bar_labels = [@sprintf("%.2f", p) for p in vec(probs)]
    bar_offset = barlabel_offset
    for (i, label) in enumerate(bar_labels)
        if probs[i] <= barlabel_threshold
            x = isnothing(upper) || isnothing(lower) ?
                    stops[i] + bar_offset : new_upper[i] + bar_offset
            text!(ax, x, group[i], text=label, fontsize=barlabel_size,
                 align=(:left, :center), color=:black)
        else
            x = isnothing(upper) || isnothing(lower) ?
                    stops[i] - bar_offset : new_lower[i] - bar_offset
            text!(ax, x, group[i], text=label, fontsize=barlabel_size,
                  align=(:right, :center), color=:white)
        end
    end
    # Add legend to subplot
    if show_legend
        axislegend(legend_title, framevisible=false)
    end
    # Add vertical lines at starts of barplots
    vlines!(ax, starts, color=:black, linestyle=:solid)
    # Resize figure to fit new plot
    if fig_or_pos isa Figure
        rowsize!(fig_or_pos.layout, 1, Auto(1.0))
        rowsize!(fig_or_pos.layout, n_rows+1, Auto(0.3))
        resize!(fig_or_pos, width, trunc(Int, height * 1.35))
        return fig_or_pos
    else
        return ax
    end
end

"Wraps text beyond a maximum length to a new line."
function text_wrap(text, max_length=80, min_last_length = 20; min_lines=1)
    words = split(text, " ")
    new_words = String[]
    count = 0
    break_idxs = Int[0]
    for w in words
        count += length(w)
        if count > max_length
            push!(new_words, "\n")
            push!(break_idxs, length(new_words))
            count = 0
        end
        push!(new_words, w)
    end
    last_length = sum(length.(new_words[last(break_idxs)+1:end]))
    n_lines = length(break_idxs)
    if last_length < min_last_length
        shortfall = min_last_length - last_length
        new_max_length = max_length - ceil(Int, shortfall / n_lines)
        return text_wrap(text, new_max_length, min_last_length; min_lines)
    end
    if n_lines < min_lines
        new_max_length = max_length - ceil(Int, min_last_length / n_lines)
        return text_wrap(text, new_max_length, min_last_length; min_lines)
    end
    return join(new_words, " ")
end    

# Switch to CairoMakie for plotting
CairoMakie.activate!()

# Register PDDL array theory
PDDL.Arrays.register!()

# Load domain and problems
DOMAIN = load_domain(joinpath(PROJECT_DIR, "dataset", "domain.pddl"))

PROBLEM_DIR = joinpath(PROJECT_DIR, "dataset", "problems")
PROBLEMS = Dict{String, Problem}()
for path in readdir(PROBLEM_DIR)
    name, ext = splitext(path)
    ext == ".pddl" || continue
    PROBLEMS[name] = load_problem(joinpath(PROBLEM_DIR, path))
end

## Storyboard plots with current and initial belief probabilities ##

# Define directories
PROBLEM_DIR = joinpath(PROJECT_DIR, "dataset", "problems")
CURR_PLAN_DIR = joinpath(PROJECT_DIR, "dataset", "plans", "exp2_current")
INIT_PLAN_DIR = joinpath(PROJECT_DIR, "dataset", "plans", "exp2_initial")
CURR_STATEMENT_DIR = joinpath(PROJECT_DIR, "dataset", "statements", "exp2_current")
INIT_STATEMENT_DIR = joinpath(PROJECT_DIR, "dataset", "statements", "exp2_initial")

RESULTS_DIR = joinpath(PROJECT_DIR, "results")
CURR_HUMAN_DIR = joinpath(RESULTS_DIR, "humans" ,"exp2_current")
INIT_HUMAN_DIR = joinpath(RESULTS_DIR, "humans", "exp2_initial")

FIGURE_DIR = joinpath(PROJECT_DIR, "figures", "storyboard_plots")
mkpath(FIGURE_DIR)

# Load plans
PLAN_IDS, PLANS, _, CURR_SPLITPOINTS = load_plan_dataset(CURR_PLAN_DIR)
_, _, _, INIT_SPLITPOINTS = load_plan_dataset(INIT_PLAN_DIR)
_, CURR_STATEMENTS = load_statement_dataset(CURR_STATEMENT_DIR)
_, INIT_STATEMENTS = load_statement_dataset(INIT_STATEMENT_DIR)

# Load human and model data
path = joinpath(CURR_HUMAN_DIR, "mean_human_data.csv")
curr_human_df = CSV.read(path, DataFrame)
curr_human_df.belief_type .= "current"
path = joinpath(INIT_HUMAN_DIR, "mean_human_data.csv")
init_human_df = CSV.read(path, DataFrame)
init_human_df.belief_type .= "initial"
human_df = vcat(curr_human_df, init_human_df)
human_df.method .= "human"
human_df.submethod .= ""

btom_path = joinpath(PROJECT_DIR, "results", "results_btom_all.csv")
btom_all_df = CSV.read(btom_path, DataFrame)
btom_all_df.method .= "btom"
transform!(btom_all_df,
    :norm_statement_probs_1 => :statement_probs_1,
    :norm_statement_probs_2 => :statement_probs_2,
    :norm_statement_probs_3 => :statement_probs_3,
    :norm_statement_probs_4 => :statement_probs_4,
    :norm_statement_probs_5 => :statement_probs_5,
)
full_df = filter(btom_all_df) do r
    r.submethod == "full" && r.act_temperature == 0.354 && r.is_judgment
end 
truebel_df = filter(btom_all_df) do r
    r.submethod == "true_belief" && r.act_temperature == 0.354 && r.is_judgment
end
noplan_df = filter(btom_all_df) do r
    r.submethod == "non_instrumental" && r.act_temperature == 0.354 && r.is_judgment
end

path = joinpath(RESULTS_DIR, "llm_baselines", "gpt4o_exp2_current_narrative_few_shot.csv")
curr_gpt4o_df = CSV.read(path, DataFrame)
curr_gpt4o_df.belief_type .= "current"
path = joinpath(RESULTS_DIR, "llm_baselines", "gpt4o_exp2_initial_narrative_few_shot.csv")
init_gpt4o_df = CSV.read(path, DataFrame)
init_gpt4o_df.belief_type .= "initial"
gpt4o_df = vcat(curr_gpt4o_df, init_gpt4o_df)
gpt4o_df.method .= "gpt4o"
gpt4o_df.submethod .= ""
transform!(gpt4o_df,
    :statement_rating_1 => :statement_probs_1,
    :statement_rating_2 => :statement_probs_2,
    :statement_rating_3 => :statement_probs_3,
    :statement_rating_4 => :statement_probs_4,
    :statement_rating_5 => :statement_probs_5
)

combined_df = vcat(human_df, full_df, gpt4o_df, cols=:union)
replace!(combined_df.method, missing => "")
replace!(combined_df.submethod, missing => "")
combined_df.method = map(zip(combined_df.method, combined_df.submethod)) do (m, s)
    isempty(s) ? m : "$(m)_$(s)"
end

# Select methods to plot
METHOD_NAMES = ["human", "btom_full", "gpt4o"]
METHOD_LABELS = ["Humans", "LaBToM", "GPT-4o"]

for plan_index in PLAN_IDS
    # Load plan and judgment points
    plan = PLANS[plan_index]
    times = CURR_SPLITPOINTS[plan_index].+1
    times = times[1:end-1]

    curr_times = CURR_SPLITPOINTS[plan_index][2:end-1] .+ 1
    init_times = INIT_SPLITPOINTS[plan_index][2:end-1] .+ 1
    init_idxs = [findfirst(==(t), curr_times) for t in init_times]

    # Initialize state, and set renderer resolution to fit state grid
    state = initstate(DOMAIN, PROBLEMS[plan_index])
    grid = state[pddl"(walls)"]
    height, width = size(grid)
    RENDERER.resolution = (width * 100, (height + 1) * 100 + 50)
    RENDERER.inventory_labelsize = 48

    # Simulate state up to final judgement point
    trajectory = PDDL.simulate(DOMAIN, state, plan[1:times[end]-1])
    # Render final state 
    canvas = render_state(RENDERER, DOMAIN, trajectory[end])
    ax = canvas.blocks[1]

    # Add key outline for any state where key is picked up
    for t in eachindex(plan[1:times[end-1]])
        plan[t].name != :pickup && continue
        obj = plan[t].args[2]
        x, y = get_obj_loc(trajectory[t], pddl"(human)")
        color = get_obj_color(trajectory[t], obj).name
        lightcolor = PDDLViz.lighten(color, 0.8)
        key_graphic = PDDLViz.KeyGraphic(x, height-y+1, color=lightcolor)
        PDDLViz.graphicplot!(ax, key_graphic)
    end

    # Render initial state marker
    canvas = render_trajectory!(
        canvas, RENDERER, DOMAIN, trajectory[1:1],
        track_stopmarker='⦿',
        track_markersize=0.6, object_colors = [:black], # [(:blue, 0.75)]
    )
    # Render rest of trajectory
    canvas = render_trajectory!(
        canvas, RENDERER, DOMAIN, trajectory[2:end],
        track_stopmarker=' ', # track_arrowmarker='⥬', 
        object_colors = [(:red, 0.75)],
        object_start_colors=[(:blue, 0.75)],
    )

    # Add tooltips at judgment points
    ax = canvas.blocks[1]
    tooltip_locs = []
    for (i, t) in enumerate(times[2:end])
        x, y = get_obj_loc(trajectory[t], pddl"(human)")
        y = height - y + 1
        act = t == times[end] ? plan[t-1] : plan[t]
        if act.name in (:up, :down)
            placement = (x == width || (x, y) in tooltip_locs) ? :left : :right
            tooltip!(ax, x, y, string(i), font = :bold, fontsize=40,
                placement=:right, textpadding = (12, 12, 4, 4), offset=20)
        elseif (x, y) in tooltip_locs
            tooltip!(ax, x, y+0.1, string(i), font = :bold, fontsize=40,
                placement=:above, textpadding = (12, 12, 4, 4), offset=20)
        elseif y == 1
            tooltip!(ax, x, y, string(i), font = :bold, fontsize=40,
                placement=:right, textpadding = (12, 12, 4, 4), offset=20)
        else
            tooltip!(ax, x, y-0.1, string(i), font = :bold, fontsize=40,
                placement=:below, textpadding = (12, 12, 4, 4), offset=20)
        end
        push!(tooltip_locs, (x, y))
    end
    canvas

    # Save stimulus image to temporary file
    img_path = joinpath(FIGURE_DIR, "trajectory_p$(plan_index).png")
    save(img_path, canvas)

    # Create new figure with image as top plot
    figure = Figure(resolution = (3800, 1000))
    ax = Axis(figure[1:4, 1], aspect = DataAspect())
    hidedecorations!(ax)
    hidespines!(ax) 
    image!(ax, rotr90(load(img_path)))
    colsize!(figure.layout, 1, Auto(1.75))
    figure

    for method_id in 1:length(METHOD_NAMES)
        # Filter for current belief results for this plan
        sub_df = filter(combined_df) do r
            r.plan_id == plan_index && r.belief_type == "current" &&
            r.method == METHOD_NAMES[method_id]
        end

        # Plot current belief statement ratings
        statement_probs = Matrix(sub_df[:, r"^statement_probs_\d$"])
        statement_colors = PDDLViz.colorschemes[:vibrant][[1, 3, 5, 6, 4]]
        statement_linestyles = [:solid, :dash, :dot, :dashdot, :dashdotdot]

        if method_id == 1
            statement_probs_sem = Matrix(sub_df[:, r"^statement_probs_sem_\d$"])
            statement_probs_ci = 1.96 * statement_probs_sem
            upper = min.(statement_probs .+ statement_probs_sem, 1.0)'
            lower = max.(statement_probs .- statement_probs_sem, 0.0)'
        else
            upper = nothing
            lower = nothing
        end

        ax = storyboard_prob_lines!(
            figure[2, 1+method_id], statement_probs', collect(1:length(curr_times)),
            upper = upper, lower = lower, rangetype=:band,
            names = ["S1", "S2", "S3", "S4", "S5"],
            colors = statement_colors,
            ts_linewidth = 4, ts_fontsize = 48,
            marker = :circle, markersize = 24, strokewidth = 1.0,
            linewidth = 10, linestyle = statement_linestyles,
            xlabel = "", ylabel = "", ylimits = (-0.01, 1.01),
            ax_args = (
                xticks = collect(1:length(curr_times)), 
                xticklabelsize = 48,
                ylabelsize = 48, yticklabelsize = 40,
            )
        )
        text!(ax, 0.5, 0.05, text = " " * METHOD_LABELS[method_id],
            fontsize = 52, font = :regular)
        if method_id == 1
            ax.ylabel = "Belief Ratings"
        else
            ax.yticksvisible = false 
            ax.yticklabelsvisible = false
        end

        # Filter for initial belief results for this plan
        sub_df = filter(combined_df) do r
            r.plan_id == plan_index && r.belief_type == "initial" &&
            r.method == METHOD_NAMES[method_id]
        end

        # Plot initial belief statement ratings
        statement_probs = Matrix(sub_df[:, r"^statement_probs_\d$"])
        statement_colors = PDDLViz.colorschemes[:vibrant][[1, 3, 5, 6, 4]]
        statement_linestyles = [:solid, :dash, :dot, :dashdot, :dashdotdot]

        if method_id == 1
            statement_probs_sem = Matrix(sub_df[:, r"^statement_probs_sem_\d$"])
            statement_probs_ci = 1.96 * statement_probs_sem
            upper = min.(statement_probs .+ statement_probs_sem, 1.0)'
            lower = max.(statement_probs .- statement_probs_sem, 0.0)'
        else
            upper = nothing
            lower = nothing
        end

        ax = storyboard_prob_lines!(
            figure[4, 1+method_id], statement_probs', init_idxs,
            upper = upper, lower = lower,
            names = ["S1", "S2", "S3", "S4", "S5"],
            colors = statement_colors,
            ts_linewidth = 4, ts_fontsize = 48,
            marker = :circle, markersize = 24, strokewidth = 1.0,
            linewidth = 10, linestyle = statement_linestyles,
            xlabel = "", ylabel = "", ylimits = (-0.01, 1.01),
            ax_args = (
                xticks = init_idxs, 
                xticklabelsize = 48,
                ylabelsize = 48, yticklabelsize = 40,
            )
        )
        xlims!(ax, 1 - 0.5, length(curr_times) + 0.5)
        text!(ax, 0.5, 0.05, text = " " * METHOD_LABELS[method_id],
            fontsize = 52, font = :regular)
        if method_id == 1
            ax.ylabel = "Belief Ratings"
        else
            ax.yticksvisible = false 
            ax.yticklabelsvisible = false
        end
    end
    figure

    # Add legends for statement probabilities
    statement_colors = PDDLViz.colorschemes[:vibrant][[1, 3, 5, 6, 4]]
    statement_linestyles = [:solid, :dash, :dot, :dashdot, :dashdotdot]
    elems = [LineElement(;color, linestyle, linewidth=10) 
            for (color, linestyle) in zip(statement_colors, statement_linestyles)]
    labels = ["S$i. " * text_wrap(s, 60) for (i, s) in enumerate(CURR_STATEMENTS[plan_index])]
    lg = Legend(figure[2, 2 + length(METHOD_LABELS)], elems, labels,
                patchsize = (160, 30), rowgap = 30, labelsize = 36,
                framevisible=false, halign=:left)
    labels = ["S$i. " *text_wrap(s, 60) for (i, s) in enumerate(INIT_STATEMENTS[plan_index])]
    lg = Legend(figure[4, 2 + length(METHOD_LABELS)], elems, labels,
                patchsize = (160, 30), rowgap = 30, labelsize = 36,
                framevisible=false, halign=:left)            
    figure
    colsize!(figure.layout, 5, Relative(0.39))

    # Add row labels
    lb = Label(figure[1, 2:end], "Current Beliefs",
            fontsize = 54, font=:bold, halign=:left)
    lb = Label(figure[3, 2:end], "Initial Beliefs",
            fontsize = 54, font=:bold, halign=:left)
    rowgap!(figure.layout, 0)
    figure

    # Save figure
    fig_path = joinpath(FIGURE_DIR, "storyboard_p$(plan_index)")
    save(fig_path * ".png", figure)
    save(fig_path * ".pdf", figure)
    display(figure)
end