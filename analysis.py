import heapq

# normalize a list so total prob = ~1
def normalize(list):
    norm = [float(i) / sum(list) for i in list]
    return(norm)

# get maximum normed value from list
# (for """confidence""" value)
def max_norm(list):
    norm = normalize(list)
    return(max(norm))

# heapq for multiple max values
# perhaps use to get next highest?
# to speed up tagging?
# http://stackoverflow.com/questions/2739051/retrieve-the-two-highest-item-from-a-list-containing-100-000-integers

