class Request(object):
    def __init__(self, nid: int, px: float, py: float, dx: float, dy: float, pready: float, pdue: float, dready: float, ddue: float, demand: float, service_time: float, time: float):
        # Initialize the instance attributes
        self.id = nid
        self.px = px
        self.py = py
        self.dx = dx
        self.dy = dy
        self.pready = pready
        self.pdue = pdue
        self.dready = dready
        self.ddue = ddue
        self.demand = demand
        self.service_time = service_time
        self.time = time