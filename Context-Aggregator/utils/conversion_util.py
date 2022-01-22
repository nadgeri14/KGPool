PAD_ENT_CNT = 9

def calculate_order_conversion(order, ent_cnt):
    return order + int(order/(ent_cnt-1)) * (PAD_ENT_CNT - ent_cnt)

def order_to_pair(order, ent_cnt):
    a = int(order / (ent_cnt - 1))
    b = order % (ent_cnt - 1)
    if(b >= a): 
        b += 1
    return (a, b)

def pair_to_order(pair, ent_ent):
    (a, b) = pair
    base = a * (ent_ent - 1) - 1
    for i in range(b+1):
        if(a == i):
            base -= 1
        base += 1
    return base

if __name__ == "__main__":
    for i in range(9):
        for j in range(9):
            if(i != j):
                print(order_to_pair(pair_to_order((i, j), 9), 9), calculate_order_conversion(pair_to_order((i, j), 9), 9))
