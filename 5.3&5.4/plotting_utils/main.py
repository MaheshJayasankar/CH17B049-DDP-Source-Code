from math import ceil
import numpy as np
import matplotlib.pyplot as plt

SAVE_DIR = "images/"
DROPLET_COLOUR_MAP = {
    0: '#b84728',
    1: '#544059',
}
DROPLET_OL_COLOUR_MAP = {
    0: '#ffdfb0',
    1: '#9b6bbf',
}
DROPLET_SIZE_MAP = {
    0: 1,
    1: 0.75
}
PIPE_COLOURS = {
    'blue': '#94d1ff',
    'yellow': '#fff1ab',
}
PIPE_OL_COLOURS = {
    'blue': '#f2fffe',
    'yellow': '#fffef2',
}

OUTER_PIPE_LEN = 50
FRAME_DURATION = 3
FIG_SIZE = (8,8)

class Droplet:
    def __init__(self, dType):
        self.type = dType
        self.edge = (-1, 0)
        self.dist = 0.0

class EventSequence:
    def __init__(self, droplets, spacing = None):
        self.droplets: 'list[Droplet]' = droplets
        self.start = self.FreezeFrame()
        self.endTime = 0
        self.keyframes = []
        if spacing is None:
            indices = list(range(len(self.droplets)))
            indices = [num / len(self.droplets) for num in indices]
            spacing = list(reversed(indices))
        self.spacing = spacing
    def FreezeFrame(self):
        positions = []
        for droplet in self.droplets:
            positions.append((droplet.edge, droplet.dist))
        return positions
    def AddEvent(self, drop, newEdge, time, reverse = False):
        self.keyframes.append((drop, newEdge, time, reverse))
        self.endTime = time
    def DuplicateKeyframe(self, time):
        _, newEdge, _, reverse = self.keyframes[-1]
        newKeyframe = -1, newEdge, time, reverse
        self.AddEvent(*newKeyframe)
    def DropPositionAtTime(self, forDrop, atTime):
        t1 = 0
        time = 0
        edge = self.start[0][0]
        reverse = False
        for idx in range(len(self.keyframes)):
            drop = self.keyframes[idx][0]
            time = self.keyframes[idx][2]
            if drop != forDrop:
                continue
            if time > atTime:
                break
            t1 = time
            edge = self.keyframes[idx][1]
            reverse = self.keyframes[idx][3]
        t2 = time
        if abs(t2 - t1) < 1e-3:
            t1 = t2 - 1e-3
        dist = (atTime - t1) / (t2 - t1)
        if reverse:
            dist = 1 - dist
        if edge[0] == -1:
            dist = self.spacing[forDrop] + dist * (1 - self.spacing[forDrop])
        return (edge, dist)
    def _FrameAtTime(self, time):
        positions = []
        for idx in range(len(self.droplets)):
            positions.append(self.DropPositionAtTime(idx, time))
        return positions
    def _SetDropletStates(self, positions):
        for idx, droplet in enumerate(self.droplets):
            droplet.edge, droplet.dist = positions[idx]
    def GotoTime(self, time):
        self._SetDropletStates(self._FrameAtTime(time))
    def SortKeyframes(self):
        self.keyframes = sorted(self.keyframes, key=lambda x: x[2])
class Graph:
    def __init__(self, A = None, coords = None, dTypes = [], sources = [], sinks = [], spacing = None, figSize = None):
        if figSize is None:
            figSize = FIG_SIZE
        self.figSize = figSize
        if A is None or coords is None:
            self.A = np.zeros((0,0))
            self.coords = []
        else:
            self.A = A
            self.coords = coords
        self.droplets: 'list[Droplet]' = []
        self.sources = sources
        self.sinks = sinks
        self.targets = []
        for dType in dTypes:
            newDroplet = Droplet(dType)
            if len(sources) > 0:
                newDroplet.edge = (-1, sources[0])
                self.targets.append(sources[0])
            else:
                self.targets.append(0)
            self.droplets.append(newDroplet)
        self.whichSink = [-1] * len(dTypes)
        self.eventSequence = EventSequence(droplets=self.droplets, spacing= spacing)
    def plot(self, save = None, title = None):
        plt.close('all')
        self.fig, self.ax = plt.subplots(figsize = self.figSize)
        plt.tight_layout()
        self.fig.axes[0].get_xaxis().set_visible(False)
        self.fig.axes[0].get_yaxis().set_visible(False)
        self.ax.axis('off')
        self.ax.set_xlim([-15, 115])
        self.ax.set_ylim([-15, 115])
        self.plotBranches()
        self.plotSources()
        self.plotSinks()
        self.plotDroplets()

        if title is not None:
            self.ax.set_title(title, y=0.05)

        if save is None:
            plt.show()
        else:
            self.fig.savefig(SAVE_DIR + save + ".png")
    def plotBranches(self):
        for idx in range(self.A.shape[0]):
            for jdx in range(idx + 1, self.A.shape[1]):
                if self.A[idx,jdx] != 0:
                    self.plotLine(idx, jdx)
    def plotSources(self):
        for sourceIdx in self.sources:
            self.plotLine(-1, sourceIdx, 'yellow')
    def plotSinks(self):
        for sinkIdx in self.sinks:
            self.plotLine(-2, sinkIdx, 'yellow')
    def plotLine(self, idx, jdx, colour = 'blue'):
        
        if idx == -1:
            x2, y2 = self.coords[jdx]
            x1 = x2 - OUTER_PIPE_LEN
            y1 = y2
        elif idx == -2:
            x2, y2 = self.coords[jdx]
            x1 = x2 + OUTER_PIPE_LEN
            x1, x2 = x2, x1
            y1 = y2
        else:
            x1, y1 = self.coords[idx]
            x2, y2, = self.coords[jdx]
        LINE_WIDTH = 20
        BORDER_WIDTH = 8
        self.ax.plot([x1, x2], [y1, y2], linewidth = LINE_WIDTH, color = PIPE_OL_COLOURS[colour], alpha = 1, zorder = 1)
        self.ax.plot([x1, x2], [y1, y2], linewidth = LINE_WIDTH + BORDER_WIDTH, color = PIPE_COLOURS[colour], alpha = 1, zorder = 0)
    def plotDroplets(self):
        for dropletIdx in range(len(self.droplets)):
            self.plotDroplet(dropletIdx)
    def plotDroplet(self, dropletIdx):
        droplet: Droplet = self.droplets[dropletIdx]
        idx, jdx = droplet.edge
        if idx == -1:
            x2, y2 = self.coords[jdx]
            x1 = x2 - OUTER_PIPE_LEN
            y1 = y2
        elif idx == -2:
            x2, y2 = self.coords[jdx]
            x1 = x2 + OUTER_PIPE_LEN
            x1, x2 = x2, x1
            y1 = y2
        elif idx == -3:
            x2, y2 = self.coords[jdx]
            x2 += OUTER_PIPE_LEN
            x1 = x2 + OUTER_PIPE_LEN
            x1, x2 = x2, x1
            y1 = y2
        else:
            x1, y1 = self.coords[idx]
            x2, y2, = self.coords[jdx]
        x = (x2 - x1) * droplet.dist + x1
        y = (y2 - y1) * droplet.dist + y1
        self.ax.scatter([x], [y], marker = 'o',
            s=144 * DROPLET_SIZE_MAP[droplet.type], color=DROPLET_OL_COLOUR_MAP[droplet.type],
            zorder=50 + dropletIdx)
        self.ax.scatter([x], [y], marker = 'o',
            s=256 * DROPLET_SIZE_MAP[droplet.type], color=DROPLET_COLOUR_MAP[droplet.type],
            zorder=49 + dropletIdx)
    def addPoint(self, coord):
        new_A = np.zeros((self.A.shape[0] + 1, self.A.shape[1] + 1))
        new_A[:-1, :-1] = self.A
        self.A = new_A
        self.coords.append(coord)
    def addPoints(self, coords):
        for coord in coords:
            self.addPoint(coord)
    def addEdge(self, idx, jdx):
        self.A[idx,jdx] = 1
        self.A[jdx,idx] = 1
    def addEdges(self, edgesList):
        for edge in edgesList:
            self.addEdge(*edge)
    def dropNextTarget(self, drop, target, time):
        prevTarget = self.targets[drop]
        if target == -2:
            edge = (-2, prevTarget)
            self.whichSink[drop] = prevTarget
        elif target == -3:
            edge = (-3, self.whichSink[drop])
        else:
            edge = (prevTarget, target)
        self.eventSequence.AddEvent(drop, edge, time)
        self.targets[drop] = target
    def finishDrop(self, drop, finishTime):
        self.dropNextTarget(drop, -3, finishTime)
    def finishSimulation(self, stopTime):
        self.eventSequence.DuplicateKeyframe(stopTime)
        self.eventSequence.SortKeyframes()
    def plotEventSequence(self, eventSequence: EventSequence = None):
        if eventSequence is None:
            eventSequence = self.eventSequence
        start = 0
        end = eventSequence.endTime
        numFrames = ceil((end - start) / FRAME_DURATION) + 1
        t = 0
        for idx in range(numFrames):
            fileName = f"{idx:05d}"
            eventSequence.GotoTime(t)
            self.plot(fileName)
            t += FRAME_DURATION
    def add3Grid(self, var_strng):
        nodes = [
            (0,100),(0,50),(0,0),(25,75),(25,25),
            (50,100),(50,50),(50,0),(75,75),(75,25),
            (100,100),(100,50),(100,0)
        ]
        edges = [
            (0,1),(0,3),(0,5),
            (1,2),(1,3),(1,4),(1,6),
            (2,4),(2,7),
            (3,5),(3,6),
            (4,6),(4,7),
            (5,6),(5,8),(5,10),
            (6,7),(6,8),(6,9),(6,11),
            (7,9),(7,12),
            (8,10),(8,11),
            (9,11),(9,12),
            (10,11),
            (11,12)
        ]
        edges_chosen = []
        for idx in range(len(edges)):
            if var_strng[idx] != 0:
                edges_chosen.append(edges[idx])
        self.addPoints(nodes)
        self.addEdges(edges_chosen)
