'''bfs(done),dfs(done),bms(done),hill climb(done),beam search(done),b and b(done), b and b with extended list(done), b and b with extended list and huristic(A*)(done),best first search(done), oracle, oracle with huristics, AO*'''
'''

oracle(done) 
oracle with huristics(done) 
AO*(done)
'''
from collections import defaultdict,deque
from typing import List,Set,Tuple
import heapq

graph = defaultdict(list)
weights = {}
heuristics = {}
and_nodes = {'B', 'E'}  # AND nodes
or_nodes = {'S', 'C', 'F', 'D'}  # OR nodes

def oracle(start:str,goal:str,oracle:int):
	stack=[(0,start,[start])]
	visited=set([start])
	while stack:
		cost,node,path=stack.pop()
		if node==goal and oracle>=cost:
			return path,cost
		for neigbor in reversed(graph[node]):
			if neigbor not in visited:
				visited.add(neigbor)
				cost_so_far=cost+weights[node,neigbor]
				print(f"path={path+[neigbor]} cost={cost_so_far}")
				if cost_so_far<=oracle:
					stack.append((cost_so_far,neigbor,path+[neigbor]))
	return []

def oracle_with_heur(start:str,goal:str,oracle:int):
	stack=[(0,start,[start])]
	visited=set([start])
	while stack:
		cost,node,path=stack.pop()
		if node==goal and oracle>=cost:
			return path,cost
		for neigbor in reversed(graph[node]):
			if neigbor not in visited:
				visited.add(neigbor)
				cost_so_far=cost+weights[node,neigbor]+heuristics[neigbor]
				print(f"path={path+[neigbor]} cost={cost_so_far}")
				if cost_so_far<=oracle:
					stack.append((cost_so_far,neigbor,path+[neigbor]))

def add_edges(u:str,v:str,weight:int):
	graph[u].append(v)
	graph[v].append(u)
	
	weights[(u,v)]=weight
	weights[(v,u)]=weight

def add_hurisitc(node:str,value:int):
	heuristics[node]=value


def bfs(start:str,goal:str):
	queue=deque([[start]])
	visited=set([start])

	while queue:
		path=queue.popleft()
		node=path[-1]

		if node==goal:
			return path
		
		for neighbor in graph[node]:
			if neighbor not in visited:
				visited.add(neighbor)
				new_path=list(path)
				new_path.append(neighbor)
				queue.append(new_path)
				print(visited)
	return[]

def dfs(start:str,goal:str):
	stack=[(start,[start])]
	visited=set()

	while stack:
		node,path=stack.pop()
		# print(f"node {node}")
		# print(f"path {path}")
		
		if node==goal:
			return path
		
		for neighbor in reversed(graph[node]):
			if neighbor not in visited:
				# print(f"neigbor {neighbor}")
				visited.add(neighbor)
				stack.append((neighbor,path+[neighbor]))
				# print(stack)
				print(visited)
	return [] 

def bms(start:str,goal:str):
	visited=set(start)
	path=[start]
	current=start
	while current!=goal:
		for neigbor in graph[current]:
			if neigbor not in visited:
				visited.add(neigbor)
				path.append(neigbor)
				current=neigbor
	if path[-1]==goal:
		return path
	else:
		return []

def branch_and_bound(start:str,goal:str):
	queue=[(0,start,[start])]

	while queue:
		cost,node,path=heapq.heappop(queue)
		
		if node==goal:
			return path
		for neighbor in graph[node]:
			new_cost=cost+weights[node,neighbor]
			heapq.heappush(queue,(new_cost,neighbor,path+[neighbor]))
	return []

def branch_and_bound_EL(start:str,goal:str):
	queue=[(0,start,[start])]
	visited=set([start])
	while queue:
		cost,node,path=heapq.heappop(queue)

		if node==goal:
			return path
		visited.add(node)
		for neighbor in graph[node]:
			if neighbor not in visited:
				new_cost=cost+weights[node,neighbor]
				heapq.heappush(queue,(new_cost,neighbor,path+[neighbor]))
	return []

def branch_and_bound_EL_hurisitc(start:str,goal:str):
	queue=[(heuristics[start],0,start,[start])]
	visited=set([start])
	while queue:
		est,cost,node,path=heapq.heappop(queue)

		if node==goal:
			return path

		visited.add(node)
		for neigbor in graph[node]:
			if neigbor not in visited:
				new_cost=cost+weights[node,neigbor]
				new_est=new_cost+heuristics[neigbor]
				heapq.heappush(queue,(new_est,new_cost,neigbor,path+[neigbor]))
	return []

def hill_climb(start:str,goal:str):
	current=start
	visited=set(start)
	path=[start]

	while current!=goal:

		neighbor=graph[current]
		if not neighbor:
			break

		min_node=get_min_node(neighbor)
		path.append(min_node)
		current=min_node

	if path[-1]==goal:
		return path
	else:
		return []

def get_min_node(neighbor):
	min_heu=999999
	node='Z'
	for vals in neighbor:
		if heuristics[vals]<min_heu:
			min_heu=heuristics[vals]
			node=vals
	return node

def beam_search(start:str,goal:str,width:int):
	current=start
	beam=[start]
	visited=set(start)
	path=[beam]
	
	while beam:
		temp_beam=[]
		for vals in beam:
			print(vals)
			if vals==goal:
				return path
			for neighbor in graph[vals]:
				if neighbor not in visited:
					visited.add(neighbor)
					temp_beam.append((heuristics[neighbor],neighbor))
		beam=get_beam(temp_beam,width)
		path.append(beam)


def get_beam(temp_beam,width):
	sample=(sorted(temp_beam)[:width])
	final=[]
	print(sample)
	for vals in sample:
		final.append(vals[1])
	return final

def best_first_search(start:str,goal:str):
	queue=[(heuristics,start,[start])]
	visited=set()
	while queue:
		heu,node,path=heapq.heappop(queue)
		if node==goal:
			return path
		visited.add(node)
		for neighbor in graph[node]:
			if neighbor not in visited:
				heapq.heappush(queue,(heuristics[neighbor],neighbor,path+[neighbor]))
	return []



def ao_star(start:str,goal:str):
	def calculate_cost(node:str,visited:set):
		if node==goal:
			return 0,[node]
		if node in visited:
			return float('inf'),[]

		visited.add(node)
		if node in and_nodes:
			total_cost=0
			total_path=[node]
			for neighbour in graph[node]:
				cost,path=calculate_cost(neighbour,visited.copy())
				total_cost=cost+weights[node,neighbour]
				total_path.append(path)
			return total_cost,total_path
		else:
			best_path=[node]
			min_cost=float('inf')
			for neighbor in graph[node]:
				cost,path=calculate_cost(neighbor,visited.copy())
				total_cost=cost+weights[node,neighbor]
				if total_cost<=min_cost:
					total_cost=min_cost
					best_path=[node]+path
			return min_cost,best_path
	_,path=calculate_cost(start,set())

	if path:
		return path
	else:
		return []
	
def main():
   
    edges = [
        ('S', 'B', 4), ('S', 'C', 3), ('S', 'F', 5),
        ('B', 'E', 2),
        ('C', 'D', 7), ('C', 'E', 10),
        ('D', 'G', 6),
        ('E', 'D', 3), ('E', 'G', 5),
        ('F', 'G', 4)
    ]
    
  
    for edge in edges:
        add_edges(edge[0], edge[1], edge[2])
    
    
    heuristic = {'S': 14, 'B': 12, 'C': 11, 'D': 6, 'E': 4, 'F': 11, 'G': 0}
    for node, val in heuristic.items():
        add_hurisitc(node, val)

    print("Graph:", dict(graph))
    print("Heuristics:", heuristics)

    
    path = ao_star('S', 'G')
    
    
    print(f"Final Path from S to G: {path}")


main()


