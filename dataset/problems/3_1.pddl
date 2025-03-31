(define (problem doors-keys-gems-problem)
  (:domain doors-keys-gems)
  (:objects blue red - color
            key1 key2 key3 - key
            door1 door2 - door
            human - agent
            gem1 gem2 gem3 gem4 - gem
            box1 box2 box3 - box)
  (:init (= (walls) (new-bit-matrix false 8 10))
         (= (xloc box1) 1)
         (= (yloc box1) 1)
         (= (xloc key1) 1)
         (= (yloc key1) 1)
         (iscolor key1 red)
         (inside key1 box1)
         (hidden key1)
         (closed box1)
         (= (walls) (set-index walls true 1 2))
         (= (walls) (set-index walls true 1 3))
         (= (walls) (set-index walls true 1 4))
         (= (xloc box2) 5)
         (= (yloc box2) 1)
         (closed box2)
         (= (xloc key2) -1)
         (= (yloc key2) -1)
         (offgrid key2)
         (= (walls) (set-index walls true 1 6))
         (= (walls) (set-index walls true 1 7))
         (= (walls) (set-index walls true 1 8))
         (= (walls) (set-index walls true 1 9))
         (= (xloc box3) 10)
         (= (yloc box3) 1)
         (= (xloc key3) 10)
         (= (yloc key3) 1)
         (iscolor key3 blue)
         (inside key3 box3)
         (hidden key3)
         (closed box3)
         (= (walls) (set-index walls true 2 2))
         (= (walls) (set-index walls true 2 3))
         (= (walls) (set-index walls true 2 4))
         (= (walls) (set-index walls true 2 6))
         (= (walls) (set-index walls true 2 7))
         (= (walls) (set-index walls true 2 8))
         (= (walls) (set-index walls true 2 9))
         (= (walls) (set-index walls true 4 2))
         (= (walls) (set-index walls true 4 3))
         (= (walls) (set-index walls true 4 4))
         (= (walls) (set-index walls true 4 5))
         (= (walls) (set-index walls true 4 6))
         (= (walls) (set-index walls true 4 7))
         (= (walls) (set-index walls true 4 8))
         (= (walls) (set-index walls true 4 9))
         (= (walls) (set-index walls true 4 10))
         (= (walls) (set-index walls true 5 2))
         (= (walls) (set-index walls true 5 3))
         (= (walls) (set-index walls true 5 4))
         (= (walls) (set-index walls true 5 5))
         (= (walls) (set-index walls true 5 6))
         (= (walls) (set-index walls true 5 7))
         (= (walls) (set-index walls true 5 8))
         (= (walls) (set-index walls true 5 9))
         (= (xloc gem1) 10)
         (= (yloc gem1) 5)
         (= (walls) (set-index walls true 7 2))
         (= (walls) (set-index walls true 7 3))
         (= (walls) (set-index walls true 7 4))
         (= (walls) (set-index walls true 7 5))
         (= (xloc door1) 6)
         (= (yloc door1) 7)
         (iscolor door1 blue)
         (locked door1)
         (= (walls) (set-index walls true 7 7))
         (= (walls) (set-index walls true 7 8))
         (= (walls) (set-index walls true 7 9))
         (= (xloc door2) 10)
         (= (yloc door2) 7)
         (iscolor door2 red)
         (locked door2)
         (= (xloc gem2) 1)
         (= (yloc gem2) 8)
         (= (walls) (set-index walls true 8 2))
         (= (walls) (set-index walls true 8 3))
         (= (walls) (set-index walls true 8 4))
         (= (walls) (set-index walls true 8 5))
         (= (xloc gem3) 6)
         (= (yloc gem3) 8)
         (= (walls) (set-index walls true 8 7))
         (= (walls) (set-index walls true 8 8))
         (= (walls) (set-index walls true 8 9))
         (= (xloc gem4) 10)
         (= (yloc gem4) 8)
         (= (xloc human) 1)
         (= (yloc human) 6))
  (:goal true)
)