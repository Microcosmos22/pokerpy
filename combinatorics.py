import math


c_total_boards = math.comb(50,5)


def p_4ofakind(pocket_pair = False):
    ## You hold a pocket pair
    # completing your pair
    if pocket_pair:
        count = math.comb(2,2)*math.comb(48,3)
    else:
        ## You hold unpaired cards. Times 2 because 4-of-a-kind of 1st card or of 2nd card
        count = math.comb(3,3)*math.comb(47,2)*2


    p_4ofakind = (count)/c_total_boards*100
    return p_4ofakind


import math

c_total_boards = math.comb(50, 5)

def p_3ofakind_or_better(pocket_pair=False):
    if pocket_pair:
        # Hitting 1 or more of your cards (includes Full Houses and Quads)
        count = math.comb(2, 1) * math.comb(48, 4)
    else:
        # Hitting 2 or more of either hole card (includes Full Houses and Quads)
        # Fix: math.comb(47, 3) ensures a correct 5-card board size
        count = math.comb(3, 2) * math.comb(47, 3) * 2

    return (count / c_total_boards) * 100

print(f"3-of-a-Kind or Better (Pocket Pair): {p_3ofakind_or_better(pocket_pair=True):.2f}%")
print(f"3-of-a-Kind or Better (Unpaired):    {p_3ofakind_or_better(pocket_pair=False):.2f}%")



p_4ofa=p_4ofakind(True)
print(f"prob 4-of-a-kind: {p_4ofa}")
p_4ofa=p_4ofakind(False)
print(f"prob 4-of-a-kind: {p_4ofa}")
p_3ofa=p_3ofakind(True)
print(f"prob 3-of-a-kind: {p_3ofa}")
p_3ofa=p_3ofakind(False)
print(f"prob 4-of-a-kind: {p_3ofa}")
