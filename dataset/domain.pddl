(define (domain doors-keys-gems)
    (:requirements :fluents :adl :typing)
    (:types 
        key gem - item
        item box door agent - physical
        physical color - object
    )
    (:predicates 
        (has ?a - agent ?i - item)
        (iscolor ?o - physical ?c - color)
        (offgrid ?i - item) 
        (locked ?d - door)
        (closed ?b - box)
        (empty ?b - box)
        (hidden ?i - item)
        (inside ?i - item ?b - box)
    )
    (:functions 
        (xloc ?o - physical) (yloc ?o - physical) - integer
        (walls) - bit-matrix
    )

    (:derived (empty ?b) (forall (?i - item) (not (inside ?i ?b))))

    (:action pickup
     :parameters (?a - agent ?i - item)
     :precondition
        (and (not (has ?a ?i)) (not (hidden ?i))
            (= (xloc ?a) (xloc ?i)) (= (yloc ?a) (yloc ?i)))
     :effect 
        (and (has ?a ?i) (offgrid ?i)
            (assign (xloc ?i) -1) (assign (yloc ?i) -1)
            (forall (?b - box) (when (inside ?i ?b) (not (inside ?i ?b)))))
    )

    (:action open
     :parameters (?a - agent ?b - box)
     :precondition
        (and (closed ?b)
            (or (and (= (xloc ?a) (xloc ?b)) (= (yloc ?a) (yloc ?b)))
                (and (= (xloc ?a) (xloc ?b)) (= (- (yloc ?a) 1) (yloc ?b)))
                (and (= (xloc ?a) (xloc ?b)) (= (+ (yloc ?a) 1) (yloc ?b)))
                (and (= (- (xloc ?a) 1) (xloc ?b)) (= (yloc ?a) (yloc ?b)))
                (and (= (+ (xloc ?a) 1) (xloc ?b)) (= (yloc ?a) (yloc ?b)))))
     :effect
        (and (not (closed ?b)) 
            (forall (?i - item) (when (inside ?i ?b) (not (hidden ?i)))))
    )

    (:action unlock
     :parameters (?a - agent ?k - key ?d - door)
     :precondition
        (and (has ?a ?k) (locked ?d)
            (exists (?c - color) (and (iscolor ?k ?c) (iscolor ?d ?c)))
            (or (and (= (xloc ?a) (xloc ?d)) (= (- (yloc ?a) 1) (yloc ?d)))
                (and (= (xloc ?a) (xloc ?d)) (= (+ (yloc ?a) 1) (yloc ?d)))
                (and (= (- (xloc ?a) 1) (xloc ?d)) (= (yloc ?a) (yloc ?d)))
                (and (= (+ (xloc ?a) 1) (xloc ?d)) (= (yloc ?a) (yloc ?d)))))
     :effect
        (and (not (has ?a ?k)) (not (locked ?d)))
    )

    (:action up
     :parameters (?a - agent)
     :precondition
        (and (> (yloc ?a) 1)
            (= (get-index walls (- (yloc ?a) 1) (xloc ?a)) false)
            (not (exists (?d - door)
                (and (locked ?d) (= (xloc ?a) (xloc ?d)) (= (- (yloc ?a) 1) (yloc ?d))))))
     :effect
        (and (decrease (yloc ?a) 1))
    )

    (:action down
     :parameters (?a - agent)
     :precondition
        (and (< (yloc ?a) (height walls))
            (= (get-index walls (+ (yloc ?a) 1) (xloc ?a)) false)
            (not (exists (?d - door)
                (and (locked ?d) (= (xloc ?a) (xloc ?d)) (= (+ (yloc ?a) 1) (yloc ?d))))))
     :effect 
        (and (increase (yloc ?a) 1))
    )

    (:action left
     :parameters (?a - agent)
     :precondition
        (and (> (xloc ?a) 1)
            (= (get-index walls (yloc ?a) (- (xloc ?a) 1)) false)
            (not (exists (?d - door)
                 (and (locked ?d) (= (yloc ?a) (yloc ?d)) (= (- (xloc ?a) 1) (xloc ?d))))))
     :effect
        (and (decrease (xloc ?a) 1))
    )

    (:action right
     :parameters (?a - agent)
     :precondition
        (and (< (xloc ?a) (width walls)) 
            (= (get-index walls (yloc ?a) (+ (xloc ?a) 1)) false)
            (not (exists (?d - door)
                 (and (locked ?d) (= (yloc ?a) (yloc ?d)) (= (+ (xloc ?a) 1) (xloc ?d))))))
     :effect
        (and (increase (xloc ?a) 1))
    )

)
