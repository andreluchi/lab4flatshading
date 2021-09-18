from Renderer import Render, GREEN, BLUE
from polygon import Polygon
from fakers import V2, V3
from collections import namedtuple
V2 = namedtuple('Point2', ['x', 'y'])
V3 = namedtuple('Point3', ['x', 'y', 'z'])

bitmap = Render()
bitmap.glInit()
bitmap.glCreateWindow(1000,1000)
# bitmap.setLightAt(V3(0.4,0,1))
bitmap.load("./sphere.obj", (1, 1, 1), (500, 500, 500))
# bitmap.triangle(V3(554, 260, 497), V3(554, 300, 497), V3(400, 265, 506), 1)
bitmap.glFinish()