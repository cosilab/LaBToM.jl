using CSV, DataFrames, JSON3
using Statistics, StatsBase
using Printf
using DataStructures: OrderedDict

using CairoMakie
using PDDL, PDDLViz

PROJECT_DIR = joinpath(@__DIR__, "..")
RESULTS_DIR = joinpath(PROJECT_DIR, "results")

## Correlation plots

# Load BToM data
df_path = joinpath(RESULTS_DIR, "results_btom_all.csv")
btom_df = CSV.read(df_path, DataFrame)
btom_df.method .= "btom"
filter!(r -> r.is_judgment, btom_df)
filter!(r -> r.act_temperature == 0.354, btom_df)
sort!(btom_df, [:submethod, :belief_type, :plan_id, :timestep])

# Load LLM data 
df_path = joinpath(RESULTS_DIR, "results_llm_all.csv")
llm_df = CSV.read(df_path, DataFrame)
sort!(llm_df, [:submethod, :belief_type, :plan_id, :timestep])

# Load correlation data
df_path = joinpath(RESULTS_DIR, "human_model_corr_per_factor.csv")
corr_df = CSV.read(df_path, DataFrame)

# Select data to plot
best_btom_df = filter(r -> r.method == "btom" && r.submethod == "full", btom_df)
best_llm_df = filter(r -> r.method == "gpt-4o" && r.submethod == "image, narrative, few-shot", llm_df)
best_corr_df = filter(statement_correlation_df) do r
    (r.method == "btom" && r.submethod == "full") ||
    (r.method == "gpt-4o" && r.submethod == "image, narrative, few-shot")
end

# Load statements
df_path = joinpath(STATEMENT_DIR, "exp2_current_translations.csv")
curr_statement_df = CSV.read(df_path, DataFrame)
curr_statement_df.belief_type .= "current"

df_path = joinpath(STATEMENT_DIR, "exp2_translations.csv")
init_statement_df = CSV.read(df_path, DataFrame)
init_statement_df.belief_type .= "initial"

statement_df = vcat(curr_statement_df, init_statement_df, cols=:union)

# Determine statements that have been mistranslated
curr_btom_df = filter(r -> r.belief_type == "current", best_btom_df)
curr_equiv = [filter(r -> r.plan_id == p, curr_statement_df).elot_equiv for p in curr_btom_df.plan_id]
curr_equiv = permutedims(reduce(hcat, curr_equiv))

init_btom_df = filter(r -> r.belief_type == "initial", best_btom_df)
init_equiv = [filter(r -> r.plan_id == p, init_statement_df).elot_equiv for p in init_btom_df.plan_id]
init_equiv = permutedims(reduce(hcat, init_equiv))

all_equiv = vcat(curr_equiv, init_equiv)

figure = Figure(resolution=(1400, 1000))
conditions = [
    "All Beliefs" => ["current", "initial"],
    "Current Beliefs" => ["current"],
    "Initial Beliefs" => ["initial"]
]
correlations = [
    (best_corr_df.all_cor, best_corr_df.all_cor_ci_lo, best_corr_df.all_cor_ci_hi),
    (best_corr_df.curr_cor, best_corr_df.curr_cor_ci_lo, best_corr_df.curr_cor_ci_hi),
    (best_corr_df.init_cor, best_corr_df.init_cor_ci_lo, best_corr_df.init_cor_ci_hi)
]

btom_colors = [PDDLViz.lighten(PDDLViz.colorschemes[:vibrant][1], x) for x in 0.0:0.1:0.2]
llm_colors = [PDDLViz.lighten(PDDLViz.colorschemes[:vibrant][3], x) for x in 0.0:0.1:0.2]
for (i, (cond, belief_types)) in enumerate(conditions)
    # Filter and extract probabilities
    human_sub_df = filter(r -> r.belief_type in belief_types, human_df)
    btom_sub_df = filter(r -> r.belief_type in belief_types, best_btom_df)
    llm_sub_df = filter(r -> r.belief_type in belief_types, best_llm_df)
    human_probs = vec(Matrix(human_sub_df[:, r"^statement_probs_(\d+)$"]))
    human_probs_sem = vec(Matrix(human_sub_df[:, r"^statement_probs_sem_(\d+)$"]))
    btom_probs = vec(Matrix(btom_sub_df[:, r"^norm_statement_probs_(\d+)$"]))
    llm_probs = vec(Matrix(llm_sub_df[:, r"^statement_probs_(\d+)$"]))

    # Set color based on whether statement was mistranslated
    if belief_types == ["current", "initial"]
        point_colors = [e ? btom_colors[i] : :black for e in vec(all_equiv)]
    elseif belief_types == ["current"]
        point_colors = [e ? btom_colors[i] : :black for e in vec(curr_equiv)]
    elseif belief_types == ["initial"]
        point_colors = [e ? btom_colors[i] : :black for e in vec(init_equiv)]
    end

    # Plot LabToM scatter plot with error bars
    ax = Axis(figure[1, i], aspect=1.0, limits=(-0.01, 1.01, -0.01, 1.01),
              ylabel="Humans", xlabel="LaBToM", ylabelsize=36, xlabelsize=36,
              title=cond, titlesize=40, titlefont=:italic, 
              yticklabelsize=20, xticklabelsize=20)
    ax.ylabelvisible = i == 1
    rangebars!(ax, btom_probs, human_probs-human_probs_sem, human_probs+human_probs_sem,
               color=[(c, 0.5) for c in point_colors])
    scatter!(ax, btom_probs, human_probs, color=point_colors)

    # Plot regression line
    X = [ones(length(btom_probs)) btom_probs]
    y = human_probs
    intercept, slope = X \ y
    ablines!(ax, intercept, slope, color=btom_colors[i],
             linestyle=:dash, linewidth=4)

    # Show correlation text label
    r = correlations[i][1][1]
    r_lo = correlations[i][2][1]
    r_hi = correlations[i][3][1] 
    text!(ax, 0.02, 0.98, text=@sprintf("r = %0.2f", r),
          align=(:left, :top), fontsize=34)
    text!(ax, 0.02, 0.86, text=@sprintf("CI: [%.2f, %.2f]", r_lo, r_hi),
          align=(:left, :top), fontsize=28)

    # Plot GPT-4o scatter plot with error bars
    ax = Axis(figure[2, i], aspect=1.0, limits=(-0.01, 1.01, -0.01, 1.01),
              ylabel="Humans", xlabel="GPT-4o", ylabelsize=36, xlabelsize=36,
              yticklabelsize=20, xticklabelsize=20)
    ax.ylabelvisible = i == 1
    rangebars!(ax, llm_probs, human_probs-human_probs_sem, human_probs+human_probs_sem,
               color=(llm_colors[i], 0.5))    
    scatter!(ax, llm_probs, human_probs, color=llm_colors[i])

    # Plot regression line
    X = [ones(length(llm_probs)) llm_probs]
    y = human_probs
    intercept, slope = X \ y
    ablines!(ax, intercept, slope, color=llm_colors[i],
             linestyle=:dash, linewidth=4)

    # Show correlation text label
    r = correlations[i][1][2]
    r_lo = correlations[i][2][2]
    r_hi = correlations[i][3][2] 
    text!(ax, 0.02, 0.98, text=@sprintf("r = %0.2f", r),
          align=(:left, :top), fontsize=34)
    text!(ax, 0.02, 0.86, text=@sprintf("CI: [%.2f, %.2f]", r_lo, r_hi),
          align=(:left, :top), fontsize=28)

end
display(figure)

# Save correlation plot
FIGURE_DIR = joinpath(PROJECT_DIR, "figures")
save(joinpath(FIGURE_DIR, "correlation.pdf"), figure)
save(joinpath(FIGURE_DIR, "correlation.png"),  figure)

## Compare statement likelihoods

RESULTS_DIR = joinpath(PROJECT_DIR, "results")
likelihood_df =
    CSV.read(joinpath(RESULTS_DIR, "likelihood_per_statement.csv"), DataFrame)
filter!(likelihood_df) do row
    (row.method == "btom" && row.submethod == "full") || row.method == "gpt-4o"
end

# Plot likelihood differences for each statement
figure = Figure(resolution=(1200, 1000))
gdf = groupby(likelihood_df, [:belief_type])
method_colors = PDDLViz.colorschemes[:vibrant][[1, 3, 2, 4]]  
for (key, group) in pairs(gdf)
    row = key.belief_type == "current" ? 1 : 2
    ax = Axis(figure[row, 1], ylabel="Δ Statement Likelihood",
              ylabelsize=36, xticklabelsvisible = false, xticksvisible = false,
              limits=(nothing, (-1.0, 1.0)), xgridvisible=false)
    if row == 1
        ax.title = "In-context minus out-of-context statement likelihoods"
        ax.titlefont = :italic
        ax.titlesize = 40
        ax.titlealign = :left
    elseif row == 2
        ax.xlabel = "Statements (sorted by Δ likelihood)"
        ax.xlabelsize = 36
    end
    for (i, subgroup) in enumerate(groupby(group, [:method, :submethod]))
        score_diffs = subgroup.score_diff
        xs = (1:length(score_diffs)) ./ (length(score_diffs) + 1)
        barplot!(ax, xs, sort(score_diffs),
                 color=(method_colors[i], 0.5), gap=0.05)
        xlims!(ax, 0, 1.0)
    end
    for (i, subgroup) in enumerate(groupby(group, [:method, :submethod]))
        score_diff_mean = mean(subgroup.score_diff)
        hlines!(ax, score_diff_mean, color=method_colors[i],
                linewidth=4, linestyle=:solid)
        score_diff_se = sem(subgroup.score_diff)
        hlines!(ax, score_diff_mean + score_diff_se,
                color=(method_colors[i], 0.5), linewidth=2, linestyle=:dash)
        hlines!(ax, score_diff_mean - score_diff_se,
                color=(method_colors[i], 0.5), linewidth=2, linestyle=:dash)
    end
    if key.belief_type == "current"
        text!(ax, 0.01, 0.95, text="Current Beliefs", align=(:left, :top),
              fontsize=36)
    else
        text!(ax, 0.01, 0.95, text="Initial Beliefs", align=(:left, :top),
              fontsize=36)
    end
end
display(figure)

bar_elems = [PolyElement(;color=(c, 0.5)) for c in method_colors]
mean_elems = [[
    LineElement(;color, linestyle=:solid, linewidth=4),
    LineElement(;color=(color, 0.5), linestyle=:dash, linewidth=2,
                points=Point2f[(0.0, 0.8), (1.0, 0.8)]),
    LineElement(;color=(color, 0.5), linestyle=:dash, linewidth=2,
                points=Point2f[(0.0, 0.2), (1.0, 0.2)])
] for color in method_colors] 
grouped_elems = [bar_elems, mean_elems]
grouped_labels = [["LabToM", "GPT-4o"], ["LabToM Mean", "GPT-4o Mean"]]

Legend(figure[3, 1], grouped_elems, grouped_labels, ["", ""],
       framevisible = false, labelsize=24, titlesize=24, patchsize=(30, 30),
       orientation=:horizontal)
rowgap!(figure.layout, 2, 10)
display(figure)

FIGURE_DIR = joinpath(PROJECT_DIR, "figures")
save(joinpath(FIGURE_DIR, "statement_likelihood_diffs.pdf"), figure)
save(joinpath(FIGURE_DIR, "statement_likelihood_diffs.png"), figure)
