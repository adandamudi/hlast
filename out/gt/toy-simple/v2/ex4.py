i = 0

def fun():
    if i == 0:
        seq = [1, 2, 3]
        seq += [1]
        print(seq)
        return seq
fun()
