class TreeNode(object):
    def __init__(self, action='do_nothing', success=False):
        self.action: str = action
        self.success: bool = success


def sequence3(input1: bool, input2: bool, input3: TreeNode) -> TreeNode:
    for input_n in [input1, input2]:
        if type(input_n) == bool and not input_n:
            return TreeNode()
        elif type(input_n) == bool and input_n:
            continue

    print(input3)
    return input3


def sequence2(input1: bool, input2: TreeNode) -> TreeNode:
    if type(input1) == bool and not input1:
        return TreeNode()

    print(input2)
    return input2


def selector2(input1: TreeNode, input2: TreeNode) -> TreeNode:
    for input_n in [input1, input2]:
        if input_n.success:
            print(input_n)
            return input_n
        else:
            continue

    return TreeNode()


def selector3(input1: TreeNode, input2: TreeNode, input3: TreeNode) -> TreeNode:
    for input_n in [input1, input2, input3]:
        if input_n.success:
            print(input_n)
            return input_n
        else:
            continue

    return TreeNode()


def carry(input1: bool) -> bool:
    return input1
