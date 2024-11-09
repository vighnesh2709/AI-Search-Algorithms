from collections import deque, defaultdict
from typing import Dict, List, Set
import heapq

graph=defaultdict(list)
weights={}
huristic={}
def add_edge(u:str,v:str,weight:int=1):
	graph[u].append(v)
	graph[v].append(u)
	weights[u]=weight
	weights[v]=weight

def print_graph():
	print(graph)

def set_huristic(node:str,value:float):
	huristic[node]=value

def bfs(start:str,goal:str):
	queue=deque([[start]])
	visited=set([start])
	while queue:
		path=queue.popleft()
		node=path[-1]
		
		if node==goal:
			return path
		for neighbour in graph[node]:
			if neighbour not in visited:
				visited.add(neighbour)
				new_path=path
				new_path.append(neighbour)
				queue.append(new_path)
	return[]

def dfs(start: str, goal: str):
    stack = [(start, [start])]
    visited = set()
    while stack:
        node, path = stack.pop()
        if node not in visited:
            if node == goal:
                return path
            visited.add(node)
            for neighbor in reversed(graph[node]):
                if neighbor not in visited:
                    stack.append((neighbor, path + [neighbor]))
    return []
    
def bms(start:str,goal:str)->list[str]:
	visited=set()
	path=[start]
	node = start
	
	
	while node != goal:
		for neighbours in graph[node]:
			if neighbours not in visited:
				print(f"neighbour of {node} are {neighbours}")
				node=neighbours
				visited.add(node)
				path.append(node)
				print(f"visited {visited}")
				
	return path
def hill_climb(start:str,goal:str)->list[str]:
	current=start
	path=[current]
	while current!=goal:
		neighbours=graph[current]
		if not neighbours:
			break
		next_node=find_minimum_heuristic(neighbours)
		print(f"this is the node with the least huristic {next_node}")
		
		if next_node[1]>=huristic[current]:
			print("reached local minima")
			break
		current=next_node[0]
		path.append(current)
	return path if path[-1]==goal else[]
		
def find_minimum_heuristic(neighbours: list) -> list:
    mini = [None, float('inf')]
    for val in neighbours:
        if huristic[val] < mini[1]:
            mini = [val, huristic[val]]
    return mini
    
    


def branch_and_bound(start: str, goal: str) -> list[str]:
    queue = [(0, start, [start])]  # Correctly initializing as a tuple
    visited = set()
    
    while queue:
        cost, node, path = heapq.heappop(queue)
        
        if node == goal:
            return path
        
        if node not in visited:
            visited.add(node)
            for neighbor in graph[node]:  
                if neighbor not in visited:
                    new_cost=cost+weights[neighbor]
                    heapq.heappush(queue,(new_cost,neighbor,path+[neighbor]))
                    print(queue)
def branch_and_bound_huristic(start:str,goal:str)->list[str]:
	queue=[(huristic[start],0,start,[start])]
	visited=set()
	
	while queue:
		estimate,cost,node,path=heapq.heappop(queue)
		
		if node==goal:
			return path
		if node not in visited:
			for neighbour in graph[node]:
				if neighbour not in visited:
					new_cost=cost+weights[neighbour]
					new_estimate=new_cost+huristic[neighbour]
					heapq.heappush(queue,(new_estimate,new_cost,neighbour,path+[neighbour]))
					print(queue)
	return []
def beam(start: str, goal: str, beam_width: int) -> list[str]:
    '''
        You are at the start node,
        see all its neighbors,
        pick least heuristic values of its neighbors,
        traverse through these values and pick the next best.
        If it is the goal, then return the path.
    '''    
    current_beam = [start]
    visited = set([start])
    path = [[start]]
    
    print(f"beam: {current_beam}")

    while current_beam:
        temp_beam = []
        for val in current_beam:    
            if val == goal:
                path.append([val])
                return path
            
            for neighbour in graph[val]:
                if neighbour not in visited:
                    visited.add(neighbour)
                    temp_beam.append((huristic[neighbour], neighbour))

        temp_beam.sort()
        if len(temp_beam) <= beam_width:
            next_beam = temp_beam
            print(f"next_beam : {next_beam}")  
        else:
            next_beam = temp_beam[:beam_width]  
            print(f"next_beam : {next_beam}")  	
        
        next_beam_values=get_next_beam_value(next_beam)
        path.extend([next_beam_values])  
        
        current_beam =next_beam_values  
        print(f"beam: {current_beam}")
    
    return []

def get_next_beam_value(next_beam:list[(int,str)])->list[str]:
	final=[]
	for values in next_beam:
		final.append(values[1])
	return final

def A_star(start:str,goal:str)->list[str]:
        '''
        1.we need to pick the least value of cost + huristic 
        2. pick from all visited nodes, it is not level wise picking
        3. so use a heapq with total, hur,cost,node and path in it, keep picking the least one in the min heap
        4. This is the same as b and b with hurisitcc and extend list
        '''
        queue=[(huristic[start]+weights[start],weights[start],start,[start])]
        visited=set([start])
        while queue:
                net_cost,cost,node,path=heapq.heappop(queue)
                if node==goal:
                    return path 
                for neighbor in graph[node]:
                    if neighbor not in visited:
                            visited.add(neighbor)
                            new_cost=cost+weights[neighbor]
                            new_est=new_cost+huristic[neighbor]
                            new_path=path+[neighbor]
                            heapq.heappush(queue,(new_est,new_cost,neighbor,new_path))
        return []

def best_first_search(start:str,goal:str):
	queue=[(huristic[start],start,[start])]
	visited=set()
	while queue:
		hur_cost,node,path=heapq.heappop(queue)
		if node==goal:
				return path
		for vals in graph[node]:
			if vals not in visited:
				visited.add(vals)
				heapq.heappush(queue,(huristic[vals],vals,path+[vals]))
	return []

def Oracle(self, g, o, d):
        """
        Oracle search performing an exhaustive search to find all possible paths.
        Returns a list of tuples, each containing a path and its cost.
        Parameters:
            g : is the object of class Graph
            o : origin/start/current node
            d : destination node
        """
        all_paths = []
        total_path = []
        stack = [(o, [], 0)]  # (node, path, cost)
        while stack:
            current, path, cost = stack.pop()
            total_path.append(path+[current])
            if current == d:
                all_paths.append((path + [current], cost))
            else:
                for neighbor, weight in zip(g.graph[current], g.weight[current]):
                    if neighbor not in path:
                        stack.append((neighbor, path + [current], cost + weight))
        print(all_paths)
        print(total_path)
        return total_path
    
def OracleH(self, g, o, d):
        all_paths = []
        total_path = []
        stack = [(o, [], 0)]  # (node, path, cost)
        while stack:
            current, path, cost = stack.pop()
            total_path.append(path+[current])
            if current == d:
                all_paths.append((path + [current], cost))
            else:
                for neighbor, weight in zip(g.graph[current], g.weight[current]):
                    if neighbor not in path:
                        stack.append((neighbor, path + [current], cost + weight + g.heuristic[neighbor]))
        print(all_paths)
        return total_path
        
def AOstar(self, g, o, d):
        open_list = [(g.heuristic[o], o, [])]
        closed_list = []
        total_path = []
        while open_list:
            open_list.sort(key=lambda x: x[0])
            h, current, path = open_list.pop(0)
            total_path.append(path+[current])
            if current == d:
                print("Optimal path:", path + [current])
                return total_path

            for neighbor, weight in zip(g.graph[current], g.weight[current]):
                if neighbor not in path and neighbor not in closed_list:
                    g_value = len(path) + weight
                    h_value = g.heuristic[neighbor]
                    f_value = g_value + h_value
                    new_path = path + [current]
                    open_list.append((f_value, neighbor, new_path))
            
            closed_list.append(current)

        print("No path found")
        return None

# def AO_star(start:str,goal:str)->list[str]:

def main():
    edges = [
        ('A', 'B'), ('A', 'C'),
        ('B', 'D'), ('B', 'E'),
        ('C', 'F'),
        ('D', 'G'), ('E', 'G'), ('F', 'G')
    ]
    heuristics = {
        'A': 4, 'B': 3, 'C': 3,
        'D': 1, 'E': 1, 'F': 1,
        'G': 0
    }
    for node,value in heuristics.items():
    	set_huristic(node,value)
    
    for edge in edges:
    	add_edge(edge[0],edge[1])
    
    print_graph()
    
    print("BFS: \n")
    path1=bfs('A','G')
    print(f"Path is {path1}")
    print("\n")

    print("DFS: \n")
    path2=dfs('A','G')
    print(f"Path is {path2}")
    print("\n")

    print("BMS: \n")
    path3=bms('A','G')
    print(f"Path is {path3}")
    print("\n")

    print("Hill Climb: \n")
    path4=hill_climb('A','G')
    print(f"Path is {path4}")
    print("\n")

    print("Branch and Bound: \n")
    path5=branch_and_bound('A','G')
    print(f"Path is {path5}")
    print("\n")

    print("Branch and bound with huristics: \n")
    path6=branch_and_bound_huristic('A','G')
    print(f"Path is {path6}")
    print("\n")

    print("Beam: \n")
    path7=beam('A','G',2)
    print(f"Path is {path7}")
    print("\n")

    print("A Star: \n")
    path8=A_star('A','G')
    print(f"Path is {path8}")
    print("\n")

    print("Best First Search: \n")
    path9=best_first_search('A','G')
    print(f"Path is {path9}")
    print("\n")
  
  
main()

'''
bfs dfs bms hill_climb b and b b and b with hurisitcs beam a star
best first search
'''