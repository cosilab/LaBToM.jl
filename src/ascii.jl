# Functions for generating gridworld PDDL problems
using PDDL

"Converts ASCII gridworlds to PDDL problem."
function ascii_to_pddl(
    str::AbstractString,
    key_set::Union{AbstractString, Nothing} = nothing,
    name="doors-keys-gems-problem";
    key_dict = Dict(
        'r' => pddl"(red)",
        'b' => pddl"(blue)",
        'y' => pddl"(yellow)" ,
        'e' => pddl"(green)",
        'p' => pddl"(pink)"
    ),
    door_dict = Dict(
        'R' => pddl"(red)",
        'B' => pddl"(blue)",
        'Y' => pddl"(yellow)" ,
        'E' => pddl"(green)",
        'P' => pddl"(pink)"
    )
)
    objects = Dict(
        :door => Const[], :key => Const[], :gem => Const[],
        :box => Const[], :color => Const[], :agent => Const[pddl"(human)"]
    )

    # Parse width and height of grid
    rows = split(str, "\n", keepempty=false)
    width, height = maximum(length.(strip.(rows))), length(rows)
    walls = parse_pddl("(= walls (new-bit-matrix false $height $width))")

    # Parse wall, item, and agent locations
    init = Term[walls]
    init_agent = Term[]
    goal = pddl"(true)"
    for (y, row) in enumerate(rows)
        for (x, char) in enumerate(strip(row))
            if char == '.' # Unoccupied
                continue
            elseif char == 'W' # Wall
                wall = parse_pddl("(= walls (set-index walls true $y $x))")
                push!(init, wall)
            elseif char == 'F'
                n = length(objects[:box]) + 1
                b = Const(Symbol("box$n"))
                push!(objects[:box], b)
                append!(init, parse_pddl("(= (xloc $b) $x)", "(= (yloc $b) $y)"))
                # push!(init, parse_pddl("(not (closed $b))"))
                n = length(objects[:key]) + 1
                k = Const(Symbol("key$n"))
                # Add key associated with box
                push!(objects[:key], k)
                # n_boxes = length(objects[:box])
                color =  Const(:red)
                push!(objects[:color], color)
                append!(init, parse_pddl("(= (xloc $k) $x)", "(= (yloc $k) $y)"))
                push!(init, parse_pddl("(iscolor $k $color)"))
                # push!(init, parse_pddl("(hidden $k)"))
                push!(init, parse_pddl("(inside $k $b)"))
                push!(init, parse_pddl("(hidden $k)"))
                push!(init, parse_pddl("(closed $b)"))

            elseif char == 'O'
                n = length(objects[:box]) + 1
                b = Const(Symbol("box$n"))
                push!(objects[:box], b)
                append!(init, parse_pddl("(= (xloc $b) $x)", "(= (yloc $b) $y)"))
                # push!(init, parse_pddl("(not (closed $b))"))
                n = length(objects[:key]) + 1
                k = Const(Symbol("key$n"))
                # Add key associated with box
                push!(objects[:key], k)
                # n_boxes = length(objects[:box])
                color =  Const(:blue)
                push!(objects[:color], color)
                append!(init, parse_pddl("(= (xloc $k) $x)", "(= (yloc $k) $y)"))
                push!(init, parse_pddl("(iscolor $k $color)"))
                # push!(init, parse_pddl("(hidden $k)"))
                push!(init, parse_pddl("(inside $k $b)"))
                push!(init, parse_pddl("(hidden $k)"))
                push!(init, parse_pddl("(closed $b)"))

            elseif char == 'C' # Box
                n = length(objects[:box]) + 1
                b = Const(Symbol("box$n"))
                push!(objects[:box], b)
                append!(init, parse_pddl("(= (xloc $b) $x)", "(= (yloc $b) $y)"))
                push!(init, parse_pddl("(closed $b)"))
                n = length(objects[:key]) + 1
                k = Const(Symbol("key$n"))
                # Check if key set is provided
                # isnothing(key_set) && continue
                # Add key associated with box
                push!(objects[:key], k)
                # n_boxes = length(objects[:box])
                # if key_set[n_boxes] == 'C' # Box is empty
                append!(init, parse_pddl("(= (xloc $k) -1)", "(= (yloc $k) -1)"))
                push!(init, parse_pddl("(offgrid $k)"))
                # else # Box has key
                #     color = key_dict[key_set[n_boxes]]
                #     push!(objects[:color], color)
                #     append!(init, parse_pddl("(= (xloc $k) $x)", "(= (yloc $k) $y)"))
                #     push!(init, parse_pddl("(iscolor $k $color)"))
                #     push!(init, parse_pddl("(hidden $k)"))
                #     push!(init, parse_pddl("(inside $k $b)"))
                # end
            elseif haskey(door_dict, char) # Door
                n = length(objects[:door]) + 1
                d = Const(Symbol("door$n"))
                color = door_dict[char]
                push!(objects[:door], d)
                push!(objects[:color], color)
                append!(init, parse_pddl("(= (xloc $d) $x)", "(= (yloc $d) $y)"))
                push!(init, parse_pddl("(iscolor $d $color)"))
                push!(init, parse_pddl("(locked $d)"))
            elseif haskey(key_dict, char) # Key
                n = length(objects[:key]) + 1
                k = Const(Symbol("key$n"))
                color = key_dict[char]
                push!(objects[:key], k)
                push!(objects[:color], color)
                append!(init, parse_pddl("(= (xloc $k) $x)", "(= (yloc $k) $y)"))
                push!(init, parse_pddl("(iscolor $k $color)"))
            elseif char == 'g' || char == 'G' # Gem
                n = length(objects[:gem]) + 1
                g = Const(Symbol("gem$n"))
                push!(objects[:gem], g)
                append!(init, parse_pddl("(= (xloc $g) $x)", "(= (yloc $g) $y)"))
                if char == 'G'
                    goal = parse_pddl("(has human $g)")
                end
            elseif char == 'h' # Agent
                append!(init_agent, parse_pddl("(= (xloc human) $x)", "(= (yloc human) $y)"))
            end
        end
    end
    append!(init, init_agent)

    # Create object list
    objlist = Const[]
    for objs in values(objects)
        sort!(unique!(objs), by=string)
        append!(objlist, objs)
    end
    # Create object type dictionary
    objtypes = Dict{Const, Symbol}()
    for (type, objs) in objects
        objs = unique(objs)
        for o in objs
            objtypes[o] = type
        end
    end

    problem = GenericProblem(Symbol(name), Symbol("doors-keys-gems"),
                             objlist, objtypes, init, goal,
                             nothing, nothing)
    return problem
end

function load_ascii_problem(path::AbstractString, keyset=nothing)
    str = open(f->read(f, String), path)
    return ascii_to_pddl(str, keyset)
end

function convert_ascii_problem(path::String)
    str = open(f->read(f, String), path)
    str = ascii_to_pddl(str)
    new_path = splitext(path)[1] * ".pddl"
    write(new_path, write_problem(str))
    return new_path
end