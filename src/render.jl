using PDDL
using PDDLViz, GLMakie

include("utils.jl")

# Define colors
vibrant = PDDLViz.colorschemes[:vibrant]
color_dict = Dict(
    :red => vibrant[1],
    :yellow => vibrant[2],
    :blue => colorant"#0072b2",
    :green => :springgreen,
    :none => :gray
)

# Define gem properties
gem_sides = [3, 4, 6, 20]
gem_angles = [pi, pi/4, 0.0, 0.0]
gem_shifts = [(0.0, 0.1), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]
gem_sizes = [1.2, 1.2, 1.1, 1.0]
gem_colors = PDDLViz.colorschemes[:vibrantlight][[4, 5, 6, 8]]

# Construct gridworld renderer
RENDERER = PDDLViz.GridworldRenderer(
    resolution = (800, 800),
    has_agent = false,
    obj_renderers = Dict(
        :agent => (d, s, o) -> HumanGraphic(),
        :key => (d, s, o) -> KeyGraphic(
            visible=!(s[Compound(:offgrid, [o])] || s[Compound(:hidden, [o])]),
            color=color_dict[get_obj_color(s, o).name]
        ),
        :door => (d, s, o) -> LockedDoorGraphic(
            visible=s[Compound(:locked, [o])],
            color=color_dict[get_obj_color(s, o).name]
        ),
        :gem => (d, s, o) -> begin
            idx = parse(Int, string(o.name)[end])
            gem = GemGraphic(
                0.0, 0.0, gem_sizes[idx], gem_sides[idx], 1.0,
                visible=!(s[Compound(:offgrid, [o])] || s[Compound(:hidden, [o])]),
                color=gem_colors[idx]
            )
            gem = gem_angles[idx] != 0 ?
                PDDLViz.rotate(gem, gem_angles[idx]) : gem
            gem = gem_shifts[idx] != (0.0, 0.0) ?
                PDDLViz.translate(gem, gem_shifts[idx]...) : gem
        end,
        :box => (d, s, o) -> begin
            color = PDDLViz.to_color(:burlywood3)
            text_color = PDDLViz.to_color(:black)
            return MultiGraphic(
                BoxGraphic(
                    is_open=!s[Compound(:closed, [o])],
                    color=s[Compound(:closed, [o])] ?
                        color : PDDLViz.lighten(color, 0.6)
                ),
                TextGraphic(
                    string(o.name)[end:end], 0.0, -0.05, 0.4;
                    color=s[Compound(:closed, [o])] ?
                        text_color : PDDLViz.lighten(text_color, 0.6),                        
                    font=:bold
                )
            )
        end
    ),
    obj_type_z_order = [:door, :box, :key, :gem, :agent],
    show_inventory = true,
    inventory_fns = [
        (d, s, o) -> s[Compound(:has, [Const(:human), o])]
    ],
    inventory_types = [:item],
    inventory_labels = ["Inventory"],
    trajectory_options = Dict(
        :tracked_objects => [Const(:human)],
        :tracked_types => Const[],
        :object_colors => [:black]
    )
)

# Plotting utilities

"Plot observer's inferences over agent's beliefs."
function plot_belief_inferences!(
    layout::GridLayout, belief_dists, belief_probs;
    state_colormap = cgrad([colorant"#0072b2", PDDLViz.lighten(colorant"#0072b2", 0.6)]),
    belief_color = :midnightblue, backgroundcolor = :white,
    ylabels = false, plotlabels = true
)
    # Plot space of agent's beliefs over environment states
    n_beliefs = length(belief_dists)
    n_cols = ceil(Int, n_beliefs / 2)
    for (i, dist) in enumerate(belief_dists)
        n_states = length(dist)
        row = i <= n_cols ? 2 : 3
        col = i <= n_cols ? i : i - n_cols
        ax = Axis(layout[row, col])
        hidedecorations!(ax)
        ylims!(ax, 0, 1)
        barplot!(ax, 1:n_states, softmax(dist), gap=0.1,
                 colormap=state_colormap, color=1:n_states)
        for x in 1:n_states
            text!(ax, x, 0, text=rich(rich("s", font=:bold), subscript("$x")),
                fontsize=30, align=(:center, :bottom))
        end
    end
    # Add belief probs as two barplots
    xticks = [rich(rich("b", font=:bold), subscript("$i")) for i in 1:n_cols]
    ax1 = Axis(layout[1, :], limits=((1-0.5, n_cols+0.5), (0, 1)),
              xticks=(1:n_cols, xticks), yticks=[0.0, 1.0], 
              xticklabelsize=36, yticklabelsize=32, xticksvisible=false,
              backgroundcolor=backgroundcolor)
    barplot!(ax1, 1:n_cols, belief_probs[1:n_cols], gap=0.1, color=belief_color,
             bar_labels = :y, flip_labels_at = 0.75, label_size = 30,
             color_over_bar = :white, color_over_background = :black,
             label_formatter = x -> @sprintf("%0.2f", x))
    xticks = [rich(rich("b", font=:bold), subscript("$i")) for i in n_cols.+(1:n_cols)]
    ax2 = Axis(layout[4, :], limits=((0.5, n_cols+0.5), (0, 1)),
               xticks=(1:n_cols, xticks),  yticks=[0.0, 1.0],
               xticklabelsize = 36, yticklabelsize=32,
               xticksvisible=false, yreversed=true, xaxisposition=:top,
               backgroundcolor=backgroundcolor)
    barplot!(ax2, 1:n_cols, belief_probs[n_cols+1:end], gap=0.1, color=belief_color,
             bar_labels = :y, flip_labels_at = 0.0, label_size = 30,
             color_over_bar = :black, color_over_background = :white,
             label_formatter = x -> @sprintf("%0.2f", x))
    # Add y-label
    if ylabels
        Label(layout[1:end, 0], "P(Beliefs | Obs.)", fontsize=34,
            rotation=pi/2, halign=:center)
    end
    # Add plot label
    if plotlabels
        text!(ax1, 0.55, 1.0, text="Beliefs", align=(:left, :top),
             fontsize = 40, font=:bold)
    end
    # Adjust layout gaps and return
    rowsize!(layout, 1, Auto(1.2))
    rowsize!(layout, 4, Auto(1.2))
    rowgap!(layout, Relative(0.01))
    colgap!(layout, Relative(0.005))
    return layout
end

plot_belief_inferences(figure::Figure, args...; kwargs...) =
    (plot_belief_inferences!(figure.laout, args...; kwargs...); figure)

"Plot observer's inferences over states, goals, and beliefs at a timestep."
function plot_step_inferences!(
    layout::GridLayout,
    state_probs, goal_probs, belief_probs, belief_dists;
    state_colormap = cgrad([colorant"#0072b2", PDDLViz.lighten(colorant"#0072b2", 0.6)]),
    goal_colors = gem_colors, belief_color = :midnightblue,
    backgroundcolor = :white, firstcolsize = Auto(0.25), 
    ylabels = false, plotlabels=true,
)
    # Plot state probabilities
    n_states = length(state_probs)
    xticks = [rich(rich("s", font=:bold), subscript("$i")) for i in 1:n_states]
    ax = Axis(layout[1, 1], limits=(nothing, (0, 1)),
              xticks=(1:n_states, xticks), xticklabelsize=36, yticklabelsize=32,
              xticksvisible=false, ylabel = "P(State | Obs.)", ylabelsize=34,
              backgroundcolor=backgroundcolor)
    ax.ylabelvisible = ylabels
    barplot!(ax, 1:n_states, state_probs, gap=0.1,
             color=1:n_states, colormap=state_colormap,
             bar_labels = :y, flip_labels_at = 0.75, label_size = 30,
             color_over_bar = :white, color_over_background = :black,
             label_formatter = x -> @sprintf("%0.2f", x))
    if plotlabels
        text!(ax, 0.5, 1.0, text="States", align=(:left, :top),
             fontsize = 40, font=:bold)
    end
    # Plot goal inferences
    n_goals = length(goal_probs)
    xticks = [rich(rich("g", font=:bold), subscript("$i")) for i in 1:n_goals]
    ax = Axis(layout[2, 1], limits=(nothing, (0, 1)),
              xticks=(1:n_goals, xticks), xticklabelsize=36, yticklabelsize=32,
              xticksvisible=false, ylabel = "P(Goal | Obs.)", ylabelsize=34,
              backgroundcolor=backgroundcolor)
    ax.ylabelvisible = ylabels
    barplot!(ax, 1:n_goals, goal_probs, gap=0.1, color=goal_colors,
             bar_labels = :y, flip_labels_at = 0.75, label_size = 30,
             color_over_bar = :white, color_over_background = :black,
             label_formatter = x -> @sprintf("%0.2f", x))
    if plotlabels
        text!(ax, 0.5, 1.0, text="Goals", align=(:left, :top),
             fontsize = 40, font=:bold)
    end
    # Plot belief inferences
    belief_layout = GridLayout(layout[1:2, 2])
    plot_belief_inferences!(belief_layout, belief_dists, belief_probs;
                            state_colormap, belief_color, backgroundcolor,
                            ylabels, plotlabels)
    # Set column size and return layout
    colsize!(layout, 1, firstcolsize)
    return layout
end

plot_step_inferences!(figure::Figure, args...; kwargs...) =
    (plot_step_inferences!(figure.layout, args...; kwargs...); figure)

