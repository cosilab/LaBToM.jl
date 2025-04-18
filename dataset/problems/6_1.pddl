(define (problem doors-keys-gems-problem)
  (:domain doors-keys-gems)
  (:objects blue red - color
            key1 key2 key3 key4 - key
            door1 door2 door3 door4 - door
            human - agent
            gem1 gem2 gem3 gem4 - gem
            box1 box2 - box)
  (:init (= (walls) (new-bit-matrix false 7 9))
         (= (xloc gem1) 1)
         (= (yloc gem1) 1)
         (= (walls) (set-index walls true 1 2))
         (= (walls) (set-index walls true 1 3))
         (= (walls) (set-index walls true 1 4))
         (= (xloc gem2) 5)
         (= (yloc gem2) 1)
         (= (walls) (set-index walls true 1 6))
         (= (walls) (set-index walls true 1 7))
         (= (walls) (set-index walls true 1 8))
         (= (xloc key1) 9)
         (= (yloc key1) 1)
         (iscolor key1 blue)
         (= (xloc door1) 1)
         (= (yloc door1) 2)
         (iscolor door1 red)
         (locked door1)
         (= (walls) (set-index walls true 2 2))
         (= (walls) (set-index walls true 2 3))
         (= (walls) (set-index walls true 2 4))
         (= (xloc door2) 5)
         (= (yloc door2) 2)
         (iscolor door2 red)
         (locked door2)
         (= (walls) (set-index walls true 2 6))
         (= (walls) (set-index walls true 2 7))
         (= (walls) (set-index walls true 2 8))
         (= (xloc key2) 3)
         (= (yloc key2) 3)
         (iscolor key2 red)
         (= (walls) (set-index walls true 4 2))
         (= (walls) (set-index walls true 4 3))
         (= (walls) (set-index walls true 4 4))
         (= (walls) (set-index walls true 4 5))
         (= (walls) (set-index walls true 4 6))
         (= (walls) (set-index walls true 4 7))
         (= (walls) (set-index walls true 4 8))
         (= (walls) (set-index walls true 4 9))
         (= (xloc door3) 7)
         (= (yloc door3) 5)
         (iscolor door3 red)
         (locked door3)
         (= (xloc door4) 8)
         (= (yloc door4) 5)
         (iscolor door4 blue)
         (locked door4)
         (= (xloc gem3) 9)
         (= (yloc gem3) 5)
         (= (walls) (set-index walls true 6 2))
         (= (walls) (set-index walls true 6 3))
         (= (walls) (set-index walls true 6 4))
         (= (walls) (set-index walls true 6 6))
         (= (walls) (set-index walls true 6 7))
         (= (walls) (set-index walls true 6 8))
         (= (walls) (set-index walls true 6 9))
         (= (xloc gem4) 1)
         (= (yloc gem4) 7)
         (= (walls) (set-index walls true 7 2))
         (= (xloc box1) 3)
         (= (yloc box1) 7)
         (= (xloc key3) 3)
         (= (yloc key3) 7)
         (iscolor key3 red)
         (inside key3 box1)
         (hidden key3)
         (closed box1)
         (= (xloc box2) 7)
         (= (yloc box2) 7)
         (= (xloc key4) 7)
         (= (yloc key4) 7)
         (iscolor key4 blue)
         (inside key4 box2)
         (hidden key4)
         (closed box2)
         (= (walls) (set-index walls true 7 8))
         (= (walls) (set-index walls true 7 9))
         (= (xloc human) 1)
         (= (yloc human) 4))
  (:goal true)
)