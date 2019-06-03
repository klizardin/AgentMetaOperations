import numpy as np
import copy
from geom.utils import froze_class
import math


def between(v1, v2, c):
    v1 = np.float32(v1)
    v2 = np.float32(v2)
    c = np.float32(c)
    c = max((np.float32(0.0), min((np.float32(1.0), c))))
    return v1*(np.float32(1.0) - c) + v2*c


@froze_class
class Point:
    """
    class for point
    you can add its, sub, multiple by value
    """

    def __init__(self, x, y):
        self.x = np.float32(x)
        self.y = np.float32(y)

    def to_numpy_array(self):
        return np.array([self.x,self.y], dtype=np.float32)

    def __eq__(self, other):
        return np.allclose(self.to_numpy_array(), other.to_numpy_array(), rtol=1e-6, atol=1e-2)

    def __add__(self, other):
        pt = Point(x = self.x, y = self.y)
        pt.x += np.float32(other.x)
        pt.y += np.float32(other.y)
        return pt

    def __sub__(self, other):
        pt = Point(x = self.x, y = self.y)
        pt.x -= np.float32(other.x)
        pt.y -= np.float32(other.y)
        return pt

    def __mul__(self, other):
        pt = Point(x = self.x, y = self.y)
        c = np.float32(other)
        pt.x *= c
        pt.y *= c
        return pt

    def rotate(self, pt_base, angle):
        pt = Point(self.x, self.y) - pt_base
        l = pt.length()
        a = math.atan2(pt.y, pt.x)
        angle += a
        pt = Point(math.cos(angle)*l, math.sin(angle)*l)
        pt += pt_base
        return pt

    def transpose(self):
        return Point(self.y, self.x)

    @staticmethod
    def vector_from_angle(angle, r=1.0):
        angle = np.float64(angle)
        return Point(math.cos(angle)*r, math.sin(angle)*r)

    def length(self):
        """
        :return: length from point to the (0,0)
        """
        a = np.array([self.x, self.y], dtype=np.float32)
        a = np.square(a)
        a = np.sum(a)
        return np.sqrt(a)

    @staticmethod
    def scalar(p1, p2):
        l1 = p1.length()
        l2 = p2.length()
        if l1 > 0: p1 *= np.float32(1.0)/l1
        if l2 > 0: p2 *= np.float32(1.0)/l2
        v = p1.x*p2.x + p1.y*p2.y
        v = max((min((v, np.float32(1.0))), np.float32(-1.0)))
        return np.float32(v)

    @staticmethod
    def scalar_angle(p1, p2):
        return np.float32(math.acos(Point.scalar(p1,p2)))

    @staticmethod
    def between(p1, p2, c):
        assert (isinstance(p1, Point))
        assert (isinstance(p2, Point))
        return Point(between(p1.x, p2.x, c), between(p1.y, p2.y, c))

    pass #class Point


MAX_LINE_PT_DISTANCE = np.float32(100)


class Line:
    def __init__(self, pt0 : Point, pt1 : Point):
        self._pts = [copy.copy(pt0), copy.copy(pt1)]

    @property
    def pt0(self):
        return self._pts[0]

    @property
    def pt1(self):
        return self._pts[1]

    @property
    def length(self):
        return (self._pts[1] - self._pts[0]).length()

    def get_pt(self, i : int):
        return self._pts[i]

    def transpose(self):
        pts = [pt.transpose() for pt in self._pts]
        return Line(pts[0], pts[1])

    def rotate(self, pt_base : Point, angle : np.float32):
        pts = [pt.rotate(pt_base, angle) for pt in self._pts]
        return Line(pts[0], pts[1])

    def get_vert_dist(self, x, y):
        pts = (self.pt0,self.pt1)
        minPt = min(pts, key=lambda x:x.x)
        maxPt = max(pts, key=lambda x:x.x)
        if x < minPt.x:
            return False, (Point(x,y) - minPt).length()
        elif x > maxPt.x:
            return False, (Point(x,y) - maxPt).length()

        if self.pt0.x == self.pt1.x:
            ys = (self.pt0.y,self.pt1.y)
            if y >= min(ys) and y <= max(ys):
                return True, np.float32(0.0)
            else:
                return True, np.float32(min((math.fabs(self.pt0.y - y), math.fabs(self.pt1.y - y))))
        else:
            dx = self.pt1.x - self.pt0.x
            dy = self.pt1.y - self.pt0.y
            c = dy/dx
            x0 = x - self.pt0.x
            d = self.pt0.y + c*x0
            return True, d - y

    def point_at_x(self, x):
        dx = self.pt1.x - self.pt0.x
        dy = self.pt1.y - self.pt0.y
        if dx != np.float32(0.0):
            return True, Point(x, self.pt0.y + dy/dx*(x - self.pt0.x))
        else:
            return False, Point(x, self.pt0.y)

    def point_at_y(self, y):
        dx = self.pt1.x - self.pt0.x
        dy = self.pt1.y - self.pt0.y
        if dy != np.float32(0.0):
            return True, Point(self.pt0.x + dx/dy*(y - self.pt0.y), y)
        else:
            return False, Point(self.pt0.x, y)

    def y_inside(self, y):
        if self.pt0.y <= self.pt1.y:
            return y >= self.pt0.y and y <= self.pt1.y
        else:
            return y >= self.pt1.y and y <= self.pt0.y

    def x_inside(self, x):
        if self.pt0.x <= self.pt1.x:
            return x >= self.pt0.x and x <= self.pt1.x
        else:
            return x >= self.pt1.x and x <= self.pt0.x

    @staticmethod
    def distance_line_line_old(pt_base : Point, angle, l1, l2):
        assert(isinstance(l1, Line))
        assert(isinstance(l2, Line))
        angle = np.float32(angle)
        l = [l1.rotate(pt_base, -angle),l2.rotate(pt_base, -angle)]
        l = [li.transpose() for li in l]
        x = [(l[0].pt0, 0, 0), (l[0].pt1, 0, 1), (l[1].pt0, 1, 0), (l[1].pt1, 1, 1)]
        x = sorted(x, key=lambda x:x[0].x)
        if x[0][1] == x[1][1]:
            return False, (l[x[1][1]].get_pt(x[1][2]) - l[x[2][1]].get_pt(x[2][2])).length()
        else:
            d1 = l[1 if x[1][1] == 0 else 0].get_vert_dist(x[1][0].x, x[1][0].y)
            d2 = l[1 if x[2][1] == 0 else 0].get_vert_dist(x[2][0].x, x[2][0].y)
            if d1[1]*(-1 if x[1][1] == 0 else 1)*d2[1]*(-1 if x[2][1] == 0 else 1) <= np.float32(0.0):
                return True, np.float32(0.0)
            dists = [d1,d2]
            dists = [(b,np.float32(math.fabs(v))) for b,v in dists]
            return min(dists, key=lambda x:x[1])
        return True, 0

    @staticmethod
    def distance_line_line(pt_base : Point, angle, l1, l2):
        assert(isinstance(l1, Line))
        assert(isinstance(l2, Line))
        angle = np.float32(angle)
        l1n = l1.rotate(pt_base, angle)
        if np.isclose(l2.pt0.x, l2.pt1.x):
            is_intersect = l1n.x_inside(l2.pt0.x)
            if is_intersect:
                is_intersect, pt_intersect = l1n.point_at_x(l2.pt0.x)
                is_intersect = is_intersect and l2.y_inside(pt_intersect.y)
        elif np.isclose(l2.pt0.y, l2.pt1.y):
            is_intersect = l1n.y_inside(l2.pt0.y)
            if is_intersect:
                is_intersect, pt_intersect = l1n.point_at_y(l2.pt0.y)
                is_intersect = is_intersect and l2.x_inside(pt_intersect.x)
        else:
            return Line.distance_line_line_old(pt_base, angle, l1, l2)
        return is_intersect, 0

    @staticmethod
    def distance_line_pt_old(pt_base : Point, angle, l, pt : Point):
        assert(isinstance(l, Line))
        angle = np.float32(angle)
        l = l.rotate(pt_base, -angle)
        p = pt.rotate(pt_base, -angle)
        l = l.transpose()
        p = p.transpose()
        b,v = l.get_vert_dist(p.x, p.y)
        return b and v >=0, v

    @staticmethod
    def distance_line_pt(pt_base : Point, angle, l, pt : Point):
        assert(isinstance(l, Line))
        angle = np.float32(angle)
        assert(pt_base is pt)
        l1 = Line(pt, pt + Point.vector_from_angle(angle,MAX_LINE_PT_DISTANCE))
        if np.isclose(l.pt0.x, l.pt1.x):
            is_intersect = l1.x_inside(l.pt0.x)
            if is_intersect:
                is_intersect, pt_intersect = l1.point_at_x(l.pt0.x)
                is_intersect = is_intersect and l.y_inside(pt_intersect.y)
        elif np.isclose(l.pt0.y, l.pt1.y):
            is_intersect = l1.y_inside(l.pt0.y)
            if is_intersect:
                is_intersect, pt_intersect = l1.point_at_y(l.pt0.y)
                is_intersect = is_intersect and l.x_inside(pt_intersect.x)
        else:
            return Line.distance_line_pt_old(pt_base, angle, l, pt)
        return is_intersect, (pt_intersect - pt).length() if is_intersect else MAX_LINE_PT_DISTANCE

    pass #class Line


@froze_class
class Size:
    """
    class to specify size
    you can add, sub or multiple Size by value
    """

    def __init__(self, *, cx = None, cy = None, sz = None):
        assert(((cx is not None) and (cy is not None))
            or (sz is not None))
        if ((cx is not None) and (cy is not None)):
            self.cx = np.float32(cx)
            self.cy = np.float32(cy)
        elif (sz is not None) and isinstance(sz,Size):
            self.cx = sz.cx
            self.cy = sz.cy
        elif (sz is not None) and isinstance(sz,Point):
            self.cx = sz.x
            self.cy = sz.y
        if self.is_empty():
            self.cx = np.float32(0)
            self.cy = np.float32(0)

    def __eq__(self, other):
        return self.cx == np.float32(other.cx)  \
            and self.cy == np.float32(other.cy)

    def __add__(self, other):
        sz = Size(cx = self.cx, cy = self.cy)
        if isinstance(other, Size):
            sz.cx += np.float32(other.cx)
            sz.cy += np.float32(other.cy)
        elif isinstance(other, Point):
            sz.cx += other.cx
            sz.cy += other.cy
        return sz

    def __sub__(self, other):
        sz = Size(cx = self.cx, cy = self.cy)
        if isinstance(other, Size):
            sz.cx -= np.float32(other.cx)
            sz.cy -= np.float32(other.cy)
        elif isinstance(other, Point):
            sz.cx -= other.cx
            sz.cy -= other.cy
        return sz

    def __mul__(self, other):
        sz = Size(cx = self.cx, cy = self.cy)
        c = np.float32(other)
        sz.cx *= c
        sz.cy *= c
        return sz

    def is_empty(self):
        """
        :return: True if Size object is empty
        """
        return (self.cx <= 0) or (self.cy <= 0)

    pass #class Size


@froze_class
class Rect:
    """
    class to represent rectagle
    you can work with properties of the class
    """

    def __init__(self, *,
            x = None, y = None, w = None, h = None, sz : Size = None,
            top_left : Point = None, bottom_right : Point = None
            ):
        assert(((x is not None) and (y is not None) and (((w is not None) and (h is not None)) or (sz is not None)))
            or ((top_left is not None) and (bottom_right is not None)))
        if (x is not None) and (y is not None) and (((w is not None) and (h is not None)) or (sz is not None)):
            self._tl = Point(x = x,y = y)
            if sz is not None:
                self._sz = sz
            elif (w is not None) and (h is not None):
                self._sz = Size(cx = w, cy = h)
        elif (top_left is not None) and (bottom_right is not None):
            self._tl = top_left
            self._sz = Size(sz=bottom_right-top_left)

    def is_empty(self):
        """
        :return: True if a rect is empty
        """
        return self._sz.is_empty()

    def __eq__(self, other):
        return self.top_left == other.top_left \
            and self.size == other.size

    @property
    def x(self):
        return self._tl.x

    @x.setter
    def x(self, x):
        self._tl.x = np.float32(x)

    @property
    def left(self):
        return self._tl.x

    @left.setter
    def left(self, x):
        self._tl.x = np.float32(x)

    @property
    def right(self):
        return self._tl.x + self._sz.cx

    @right.setter
    def right(self, right):
        self._sz = Size(cx=np.float32(right)-self._tl.x, cy=self._sz.cy)

    @property
    def y(self):
        return self._tl.y

    @y.setter
    def y(self, y):
        self._tl.y = np.float32(y)

    @property
    def top(self):
        return self._tl.y

    @top.setter
    def top(self, y):
        self._tl.y = np.float32(y)

    @property
    def bottom(self):
        return self._tl.y + self._sz.cy

    @bottom.setter
    def bottom(self, bottom):
        self._sz = Size(cx=self._sz.cx, cy=np.float32(bottom)-self._tl.y)

    @property
    def w(self):
        return self._sz.cx

    @w.setter
    def w(self, w):
        self._sz = Size(cx=np.float32(w),cy=self._sz.cy)

    @property
    def h(self):
        return self._sz.cy

    @h.setter
    def h(self, h):
        self._sz = Size(cx=self._sz.cx,cy=np.float32(h))

    @property
    def top_left(self):
        return copy.copy(self._tl)

    @top_left.setter
    def top_left(self, top_left : Point):
        self._tl = copy.copy(top_left)

    @property
    def top_right(self):
        pt = copy.copy(self._tl)
        pt.x += self._sz.cx
        return pt

    @top_right.setter
    def top_right(self, top_right : Point):
        self._tl.y = top_right.y
        self._sz = Size(cx=top_right.x-self._tl.x, cy=self._sz.cy)

    @property
    def bottom_left(self):
        pt = copy.copy(self._tl)
        pt.y += self._sz.cy
        return pt

    @bottom_left.setter
    def bottom_left(self, bottom_left : Point):
        self._tl.x = bottom_left.x
        self._sz = Size(cx=self._sz.cx, cy=bottom_left.y-self._tl.y)

    @property
    def bottom_right(self):
        pt = copy.copy(self._tl)
        pt += Point(self._sz.cx, self._sz.cy)
        return pt

    @bottom_right.setter
    def bottom_right(self, bottom_right : Point):
        sz = bottom_right - self._tl
        self._sz = Size(sz=sz)

    @property
    def size(self):
        return copy.copy(self._sz)

    @size.setter
    def size(self, size : Size):
        self._sz = size

    def union(self, rc):
        """
        :param rc:
        :return: union to current rect with argument (rc) rect and return that union`s rect
        """
        l = min((self.left, rc.left))
        r = max((self.right, rc.right))
        t = min((self.top, rc.top))
        b = max((self.bottom, rc.bottom))
        return Rect(top_left=Point(l, t), bottom_right=Point(r, b))

    def intersect(self, rc):
        """
        :param rc:
        :return: intersect to current rect with argument (rc) rect and return that intersects`s rect
        """
        l = max((self.left, rc.left))
        r = min((self.right, rc.right))
        t = max((self.top, rc.top))
        b = min((self.bottom, rc.bottom))
        return Rect(top_left=Point(l, t), bottom_right=Point(r, b))

    pass #class Rect


def normal_distribution_density(x, mean, deviation):
    d2 = math.pow(deviation, 2.0)*2
    return np.float32(math.exp(-math.pow(x-mean,2.0)/d2)/math.pow(math.pi*d2, 0.5))


def angle_diff(a1,a2):
    p1 = Point.vector_from_angle(a1, 1.0)
    p2 = Point.vector_from_angle(a2, 1.0)
    v = p1.x*p2.x + p1.y*p2.y
    v = np.clip(v, np.float32(-1.0), np.float32(1.0))
    return np.arccos(v)


