import struct
from obj import Obj
import math
import random
from fakers import V2, V3
from collections import namedtuple


# Math utilities
def bbox(*vertices):
  """
    Input: n size 2 vectors
    Output: 2 size 2 vectors defining the smallest bounding rectangle possible
  """
  xs = [ vertex.x for vertex in vertices ]
  ys = [ vertex.y for vertex in vertices ]
  xs.sort()
  ys.sort()

  return V2(xs[0], ys[0]), V2(xs[-1], ys[-1])

def barycentric(A, B, C, P):
  """
    Input: 3 size 2 vectors and a point
    Output: 3 barycentric coordinates of the point in relation to the triangle formed
            * returns -1, -1, -1 for degenerate triangles
  """
  cx, cy, cz = cross(
    V3(B.x - A.x, C.x - A.x, A.x - P.x),
    V3(B.y - A.y, C.y - A.y, A.y - P.y)
  )

  if abs(cz) < 1:
    return -1, -1, -1   # this triangle is degenerate, return anything outside

  # [cx cy cz] = [u v 1]

  u = cx/cz
  v = cy/cz
  w = 1 - (u + v)

  return w, v, u


V2 = namedtuple('Point2', ['x', 'y'])
V3 = namedtuple('Point3', ['x', 'y', 'z'])


def sum(v0, v1):
  """
    Input: 2 size 3 vectors
    Output: Size 3 vector with the per element sum
  """
  return V3(v0.x + v1.x, v0.y + v1.y, v0.z + v1.z)

def sub(v0, v1):
  """
    Input: 2 size 3 vectors
    Output: Size 3 vector with the per element substraction
  """
  return V3(v0.x - v1.x, v0.y - v1.y, v0.z - v1.z)

def mul(v0, k):
  """
    Input: 2 size 3 vectors
    Output: Size 3 vector with the per element multiplication
  """
  return V3(v0.x * k, v0.y * k, v0.z *k)

def dot(v0, v1):
  """
    Input: 2 size 3 vectors
    Output: Scalar with the dot product
  """
  return v0.x * v1.x + v0.y * v1.y + v0.z * v1.z

def cross(v0, v1):
  """
    Input: 2 size 3 vectors
    Output: Size 3 vector with the cross product
  """
  return V3(
    v0.y * v1.z - v0.z * v1.y,
    v0.z * v1.x - v0.x * v1.z,
    v0.x * v1.y - v0.y * v1.x,
  )
def length(v0):
  """
    Input: 1 size 3 vector
    Output: Scalar with the length of the vector
  """
  return (v0.x**2 + v0.y**2 + v0.z**2)**0.5

def norm(v0):
  """
    Input: 1 size 3 vector
    Output: Size 3 vector with the normal of the vector
  """
  v0length = length(v0)

  if not v0length:
    return V3(0, 0, 0)

  return V3(v0.x/v0length, v0.y/v0length, v0.z/v0length)

# ===============================================================
# Utilities
# ===============================================================



def char(c):
    """
    Input: requires a size 1 string
    Output: 1 byte of the ascii encoded char
    """
    return struct.pack('=c', c.encode('ascii'))


def word(w):
    """
    Input: requires a number such that (-0x7fff - 1) <= number <= 0x7fff
           ie. (-32768, 32767)
    Output: 2 bytes
    Example:
    >>> struct.pack('=h', 1)
    b'\x01\x00'
    """
    return struct.pack('=h', w)


def dword(d):
    """
    Input: requires a number such that -2147483648 <= number <= 2147483647
    Output: 4 bytes
    Example:
    >>> struct.pack('=l', 1)
    b'\x01\x00\x00\x00'
    """
    return struct.pack('=l', d)

"Function that parses a color"
def color(r, g, b):
    return bytes([b, g, r])


# ===============================================================
# Constants
# ===============================================================

BLACK = color(0, 0, 0)
GREEN = color(50, 168, 82)
BLUE = color(50, 83, 168)
RED = color(168, 50, 60)
WHITE = color(255, 255, 255)



class ViewPort(object):

    def setSize(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

class Point(object):

    def __init__(self, x, y):
        self.x = x
        self.y = y



# ===============================================================
# Renders a BMP file
# ===============================================================



class Render(object):
    def __init__(self):
        self.paintColor = WHITE
        self.bufferColor = BLACK
        self.light = V3(0, 0, 1)

    def validateRanges(self, upper, lower, percent):
        return upper[0] >= percent >= upper[1] or lower[0] >= percent >= lower[1]

    def setLightAt(self, l):
        self.light = l


    def shader(self, x, y, z, intensity):


        intensity = 1
        if z > 720:
            return color(int(195 * intensity), int(233 * intensity), int(236 * intensity))
        elif z > 700:
            return color(int(183 * intensity), int(221 * intensity), int(224 * intensity))
        elif z > 675:
            return color(int(171 * intensity), int(208 * intensity), int(216 * intensity))
        elif z > 660:
            return color(int(166 * intensity), int(200 * intensity), int(208 * intensity))
        elif z > 650:
            return color(int(155 * intensity), int(192 * intensity), int(200 * intensity))
        elif z > 640:
            return color(int(152 * intensity), int(187 * intensity), int(197 * intensity))
        elif z > 630:
            return color(int(147 * intensity), int(184 * intensity), int(192 * intensity))
        elif z > 615:
            return color(int(137 * intensity), int(174 * intensity), int(182 * intensity))
        elif z > 590:
            return color(int(126 * intensity), int(161 * intensity), int(167 * intensity))
        elif z > 575:
            return color(int(100 * intensity), int(138 * intensity), int(141 * intensity))
        else:
            return color(int(83 * intensity), int(103 * intensity), int(109 * intensity))

    def grayShader(self, x, y, z, intensity):
        scaledGray = int(255 * intensity)
        if scaledGray > 255:
            scaledGray = 255
        if scaledGray < 0:
            scaledGray = 0
        # print((z-500)/(800-500))
        return color(scaledGray, scaledGray, scaledGray)


    def glInit(self):
        self.viewPort = ViewPort()

    def glCreateWindow(self, width, height):
        self.width = width
        self.height = height
        self.glClear()


    def glViewPort(self, x, y, width, height):
        self.viewPort.setSize(x, y, width, height)

    def glClear(self):
        self.framebuffer = [
            [self.bufferColor for x in range(self.width)]
            for y in range(self.height)
        ]
        self.zbuffer = [
            [-float('inf') for x in range(self.width)]
            for y in range(self.height)
        ]

    def glFinish(self, filename='out.bmp'):
        f = open(filename, 'bw')

        # File header (14 bytes)
        f.write(char('B'))
        f.write(char('M'))
        f.write(dword(14 + 40 + self.width * self.height * 3))
        f.write(dword(0))
        f.write(dword(14 + 40))

        # Image header (40 bytes)
        f.write(dword(40))
        f.write(dword(self.width))
        f.write(dword(self.height))
        f.write(word(1))
        f.write(word(24))
        f.write(dword(0))
        f.write(dword(self.width * self.height * 3))
        f.write(dword(0))
        f.write(dword(0))
        f.write(dword(0))
        f.write(dword(0))

        # Pixel data (width x height x 3 pixels)
        for x in range(self.height):
            for y in range(self.width):
                f.write(self.framebuffer[x][y])

        f.close()

    def display(self, filename='out.bmp'):
        self.glFinish(filename)

        try:
            from wand.image import Image
            from wand.display import display

            with Image(filename=filename) as image:
                display(image)
        except ImportError:
            pass  # do nothing if no wand is installed


    # Now glVertex is not the encharged to print he only normalizes the cordenates of a single point
    def glVertex(self, x, y):
        currentYCordinate =  self.viewPort.y + (self.viewPort.height//2) * (y + 1)
        currentXCordinate = self.viewPort.x + (self.viewPort.width//2) * (x + 1)
        self.point(currentXCordinate, currentYCordinate)

    def point(self, normalizedX, normalizedY, color = None):
        if color is None:
            self.framebuffer[int(normalizedY)][int(normalizedX)] = self.paintColor
        else:
            self.framebuffer[int(normalizedY)][int(normalizedX)] = color

    def glClearColor(self, r, g, b):
        self.bufferColor = color(r,g,b)

    def glColor(self, r, g, b):
        self.paintColor= color(r,g,b)

    def line(self, x0, y0, x1, y1, transform = True):
        if transform:
            y0 = self.viewPort.y + (self.viewPort.height // 2) * (y0 + 1)
            y1 = self.viewPort.y + (self.viewPort.height // 2) * (y1 + 1)
            x0 = self.viewPort.x + (self.viewPort.width // 2) * (x0 + 1)
            x1 = self.viewPort.x + (self.viewPort.width // 2) * (x1 + 1)
        dy = abs(y1 - y0)
        dx = abs(x1 - x0)
        steep = dy > dx
        if steep:
            x0, y0 = y0, x0
            x1, y1 = y1, x1
        if x0 > x1:
            x0, x1 = x1, x0
            y0, y1 = y1, y0

        dy = abs(y1 - y0)
        dx = abs(x1 - x0)
        offset = 0
        threshold = 0.5 * 2 * dx

        y = y0
        for x in range(x0, x1 + 1):
            if steep:
                self.point(y, x)
            else:
                self.point(x, y)

            offset += dy * 2
            if offset >= threshold:
                y += 1 if y0 < y1 else -1
                threshold += dx * 2

    def getLine(self, x0, y0, x1, y1):
        linePoints = []
        # y0 = self.viewPort.y + (self.viewPort.height / 2) * (y0 + 1)
        # y1 = self.viewPort.y + (self.viewPort.height / 2) * (y1 + 1)
        # x0 = self.viewPort.x + (self.viewPort.width / 2) * (x0 + 1)
        # x1 = self.viewPort.x + (self.viewPort.width / 2) * (x1 + 1)
        dy = abs(y1 - y0)
        dx = abs(x1 - x0)
        steep = dy > dx
        if steep:
            x0, y0 = y0, x0
            x1, y1 = y1, x1
        if x0 > x1:
            x0, x1 = x1, x0
            y0, y1 = y1, y0

        dy = abs(y1 - y0)
        dx = abs(x1 - x0)
        offset = 0
        threshold = dx

        y = y0
        for x in range(int(x0), int(x1) + 1):
            if steep:
                linePoints.append(Point(y, x))
            else:
                linePoints.append(Point(x, y))

            offset += dy * 2
            if offset >= threshold:
                y += 1 if y0 < y1 else -1
                threshold += dx * 2
        return linePoints

    def drawLines(self, polygon):
        yPointsLines = []
        xPointsLines = []
        lines = []

        for index, point in enumerate(polygon.points):
            point2 = polygon.points[(index + 1) % len(polygon.points)]
            line = self.getLine(point[0], point[1], point2[0], point2[1])
            self.line(point[0], point[1], point2[0], point2[1], False)
            for lp in line:
                lines.append(lp)
                yPointsLines.append(lp.y)
                xPointsLines.append(lp.x)

        # centerY = self.viewPort.height / 2
        # centerX = self.viewPort.width / 2


        minY = min(yPointsLines)
        maxY = max(yPointsLines)
        minX = min(xPointsLines)
        maxX = max(xPointsLines)
        for indexX in range(minX, maxX):
            iterableLinesX = [line for line in lines if line.x == indexX]
            for indexY in range(minY, maxY):
                iterableLinesY = [linep for linep in lines if linep.y == indexY]
                if minY < indexY < maxY:
                    if minX < indexX < maxX:
                        if any(i.y <= indexY for i in iterableLinesX) and any(i.y >= indexY for i in iterableLinesX):
                            if any(i.x <= indexX for i in iterableLinesY) and any(i.x >= indexX for i in iterableLinesY):
                                self.point(indexX, indexY)

    def load(self, filename, translate, scale):
        model = Obj(filename)
        light = self.light

        for face in model.faces:
            vcount = len(face)

            if vcount == 3:
                f1 = face[0][0] - 1
                f2 = face[1][0] - 1
                f3 = face[2][0] - 1

                a = self.transform(model.vertices[f1], translate, scale)
                b = self.transform(model.vertices[f2], translate, scale)
                c = self.transform(model.vertices[f3], translate, scale)

                normal = norm(cross(sub(b, a), sub(c, a)))
                intensity = dot(normal, light)
                grey = round(255 * intensity)
                # if intensity < 0:
                #     return
                # if grey < 0:
                #     continue
                self.triangle(a, b, c, intensity)
            else:
                # assuming 4
                f1 = face[0][0] - 1
                f2 = face[1][0] - 1
                f3 = face[2][0] - 1
                f4 = face[3][0] - 1

                vertices = [
                    self.transform(model.vertices[f1], translate, scale),
                    self.transform(model.vertices[f2], translate, scale),
                    self.transform(model.vertices[f3], translate, scale),
                    self.transform(model.vertices[f4], translate, scale)
                ]

                normal = norm(cross(sub(vertices[0], vertices[1]),
                                    sub(vertices[1], vertices[2])))  # no necesitamos dos normales!!
                intensity = dot(normal, light)
                grey = round(255)
                # if grey < 0:
                #     continue


                A, B, C, D = vertices

                self.triangle(A, B, C, intensity)
                self.triangle(A, C, D, intensity)

    def triangle(self, A, B, C, intensity, color=None):
        bbox_min, bbox_max = bbox(A, B, C)

        for x in range(bbox_min.x, bbox_max.x + 1):
            for y in range(bbox_min.y, bbox_max.y + 1):
                w, v, u = barycentric(A, B, C, V2(x, y))
                if w < 0 or v < 0 or u < 0:  # 0 is actually a valid value! (it is on the edge)
                    continue

                z = A.z * w + B.z * v + C.z * u
                if x < 0 or y < 0:
                    continue
                try:
                    if x < len(self.zbuffer) and y < len(self.zbuffer[x]) and z > self.zbuffer[x][y]:
                        if color is None:
                            self.point(x, y, self.grayShader(x,y,z, intensity))
                        else:
                            self.point(x, y, color)
                        self.zbuffer[x][y] = z
                except Exception as e:
                    # print(e)
                    e.args

    def transform(self, vertex, translate=(0, 0, 0), scale=(1, 1, 1)):
        # returns a vertex 3, translated and transformed
        return V3(
            round((vertex[0] + translate[0]) * scale[0]),
            round((vertex[1] + translate[1]) * scale[1]),
            round((vertex[2] + translate[2]) * scale[2])
        )