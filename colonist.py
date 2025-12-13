
#FOR MODE w/ 4 PLAYERS

"""
This will ideally return:
    Best placement for settlements 1, 2 and direction of road
    Best option given a state of a game
"""


class Cell:
    pings = 0
    material = "Uninitialized"
    def __init__(self, pings, material):
        self.pings = pings
        self.material = material


class gameState:
    board = [[""]*3,[""]*4,[""]*5,[""]*3] #later want "" to be a Cell
    def updateState():
        for row in board:
            for cell in row:
                pass #not sure how to implement

    def clear():
        for row in board:
            for cell in row:
                cell = ""


turn_position = input("Enter turn Position (1-4):")

ranked_materials = [] #rank from 1-5

class Nodes:
    pings_around = 0
    has_three = false
    has_two = false
    if has_three:
        materials = []*3
        #get materials around
    if has_two:
        materials = []*2
        #get materials around
    def get_materials_around(Node) => Array:
        #idk how we'd iplement
        return None

#want to take everything into account (weak materials, turn_position, etc)

if turn_position = 1:
    #sort nodes by max pings
    #favor ore wheat sheep
    #else target brick wood
    #port position


