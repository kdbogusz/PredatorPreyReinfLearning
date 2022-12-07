def sequence3(input1, input2, input3):
        for input in [input1, input2, input3]:
            if input == False:
                return False
            elif input == True:
                continue
            else:
                return input
        return True

def sequence2(input1, input2):
    for input in [input1, input2]:
        if input == False:
            return False
        elif input == True:
            continue
        else:
            return input
    return True

def selector2(input1, input2):
    for input in [input1, input2]:
        if input == False or input == True:
            continue
        else:
            return input
    return False

def selector3(input1, input2, input3):
    for input in [input1, input2, input3]:
        if input == False or input == True:
            continue
        else:
            return input
    return False