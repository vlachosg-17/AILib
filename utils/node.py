from matplotlib import patches as ptc
from matplotlib import pyplot as plt

class TreeNode:
    def __init__(self, tree_node, width, height=None, text=""):
        self.width = width
        self.height = height
        self.node = tree_node["node"]
        self.father = tree_node["father"]
        self.lab = None
        self.depth = tree_node["depth"]
        if self.node["class"] is None:
            self.leaf = False
            self.gini = self.node["gini"]
            self.size = self.node["size"]
            self.color = "seagreen"
        else:
            self.leaf = True
            self.label = self.node["class"]
            self.color = "tomato"
        self.pos = None
        self.th = 180
        self.lw = 2
        

    def make_ann(self, leaf):
        if not leaf:
            return "depth=%d"%self.depth+"\n"+"gini=%.3f"%self.gini+"\n"+"size="+str(self.size)
        else:
            return str(self.label)

    def plot(self, ax, xy, tp=""):
        self.tp = tp
        self.x, self.y = xy
        self.pos = xy
        if ax is None: ax = plt.gca()
        if tp == "circle":
            self.height=self.width
            self.radius=self.width/2
            self.node_shape = ptc.Circle(self.pos, self.radius,
                                         linewidth=self.lw, 
                                         color=self.color,
                                         fill=True)
            ax.add_artist(self.node_shape)

        elif tp == "rect":
            self.node_shape  = ptc.Rectangle(self.pos, self.width, self.height,
                                             linewidth=self.lw, color=self.color,
                                             fill=True)
            ax.add_artist(self.node_shape)
            cx, cy = self.x + self.width/2.0, self.y + self.height/2.0 
            ax.annotate(self.make_ann(leaf=self.leaf),
                        (cx, cy), color='black', weight='bold', 
                        fontsize=3*self.width, ha='center', va='center')

        else:
            self.node_shape = ptc.Ellipse(self.pos, 
                                          self.width, self.height,
                                          linewidth=self.lw, color=self.color,
                                          fill=True)
            ax.add_artist(self.node_shape)
            cx, cy = self.x, self.y
            ax.annotate(self.make_ann(leaf=self.leaf),
                        (cx, cy), color='black', weight='bold', 
                        fontsize=7*self.width, ha='center', va='center')
        return ax

    def connect(self, ax, next_node):
        if ax is None: ax = plt.gca()
        if self.pos is None: self.x, self.y = 0, 0
        if next_node.pos is None: next_node.x, next_node.y = 0, 0
        if self.tp is not "rect":
            xyA = (self.x, self.y-self.height/2.0)
            xyB = (next_node.x, next_node.y+next_node.height/2.0)
        else:
            xyA = (self.x+self.width/2.0, self.y)
            xyB = (next_node.x+next_node.width/2.0, next_node.y+next_node.height)
        
        node_con = ptc.ConnectionPatch(xyA, xyB, coordsA="data")
        ax.add_artist(node_con)
        return ax

