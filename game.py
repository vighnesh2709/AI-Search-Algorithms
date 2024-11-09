graph={
    'A':['B','C'],
    'B':['D','E'],
    'C':['F','G'],
    'D':['H','I'],
    'E':['J','K'],
    'F':['L','M'],
    'G':['N','O'],
    'H':['P','Q'],
    'I':['R','S'],
    'J':['T','U'],
    'K':['V','W'],
    'L':['X','Y'],
    'M':['Z','AA'],
    'N':['AB','AC'],
    'O':['AD','AE'],
    'P':[],
    'Q':[],
    'R':[],
    'S':[],
    'T':[],
    'U':[],
    'V':[],
    'W':[],
    'X':[],
    'Y':[],
    'Z':[],
    'AA':[],
    'AB':[],
    'AC':[],
    'AD':[],
    'AE':[]

}
evaluations={
    'P':3,
    'Q':4,
    'R':2,
    'S':1,
    'T':7,
    'U':8,
    'V':9,
    'W':10,
    'X':2,
    'Y':11,
    'Z':1,
    'AA':12,
    'AB':14,
    'AC':9,
    'AD':13,
    'AE':16

}

def minimax(node,depth,maximizing):
    if depth==0 or node not in graph or not graph[node]:
        return evaluations[node]
    
    if maximizing:
        max_eval=float('-inf')
        for child in graph[node]:
            eval=minimax(child,depth-1,False)
            max_eval=max(eval,max_eval)
        return max_eval
    else:
        min_eval=float('inf')
        for child in graph[node]:
            eval=minimax(child,depth-1,True)
            min_eval=min(eval,min_eval)
        return min_eval

def alpha_beta(node,depth,alpha,beta,maximizing):
    if depth==0 or node not in graph or not graph[node]:
        return evaluations[node]
    
    if maximizing:
        max_eval=float('-inf')
        for child in graph[node]:
            eval=alpha_beta(child,depth-1,alpha,beta,False)
            max_eval=max(eval,max_eval)
            alpha=max(eval,alpha)
            if alpha>=beta:
                break
        return max_eval
    else:
        min_eval=float('inf')
        for child in graph[node]:
            eval=alpha_beta(child,depth-1,alpha,beta,True)
            min_eval=min(eval,min_eval)
            beta=min(beta,eval)
            if alpha>=beta:
                break
        return min_eval
def main():
    ans=minimax('A',depth=5,maximizing=True)
    print(f"Min Max ans: {ans}")

    ans1=alpha_beta('A',depth=5,alpha=float('-inf'),beta=float('inf'),maximizing=True)
    print(f"Alpha Beta pruning ans :{ans1}")
main()