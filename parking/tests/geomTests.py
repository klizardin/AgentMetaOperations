import unittest
from geom.primitives import Point, Size, Rect
import numpy as np
import math


class GeomTestCase(unittest.TestCase):
    #type tests to support needed numpy arrays types
    def test_pt_types(self):
        pts = [
            Point(x=1,y=2),
            Point(x=1.1, y=2.2),
            Point(x=np.float64(1.1), y=np.float64(2.2))
        ]
        for pt in pts:
            self.assertTrue(isinstance(pt.x, np.float32))
            self.assertTrue(isinstance(pt.y, np.float32))
            self.assertTrue(isinstance(pt.length(), np.float32))

    def test_sz_types(self):
        sizes = [
            Size(cx=1, cy=2),
            Size(cx=1.1, cy=2.2),
            Size(cx=np.float64(1.1), cy=np.float64(2.2)),
        ]
        for sz in sizes:
            self.assertTrue(isinstance(sz.cx, np.float32))
            self.assertTrue(isinstance(sz.cy, np.float32))

    def test_rect_types(self):
        rects = [
            Rect(x=1, y=2, w=3, h=4),
            Rect(x=1.1, y=2.2, w=3.3, h=4.4),
            Rect(x=np.float64(1.1), y=np.float64(2.2), w=np.float64(3.3), h=np.float64(4.4)),
            Rect(x=1.1, y=2.2, sz=Size(cx=1.1, cy=2.2)),
            Rect(top_left=Point(1.1,2.2), bottom_right=Point(2.2,3.2)),
        ]
        for rc in rects:
            self.assertTrue(isinstance(rc.x, np.float32))
            self.assertTrue(isinstance(rc.left, np.float32))
            self.assertTrue(isinstance(rc.y, np.float32))
            self.assertTrue(isinstance(rc.top, np.float32))
            self.assertTrue(isinstance(rc.right, np.float32))
            self.assertTrue(isinstance(rc.bottom, np.float32))
            self.assertTrue(isinstance(rc.w, np.float32))
            self.assertTrue(isinstance(rc.h, np.float32))
            self.assertTrue(isinstance(rc.top_left, Point))
            self.assertTrue(isinstance(rc.top_right, Point))
            self.assertTrue(isinstance(rc.bottom_left, Point))
            self.assertTrue(isinstance(rc.bottom_right, Point))
            self.assertTrue(isinstance(rc.size, Size))

    def test_pt_base_ops(self):
        x1,y1,x2,y2,c = np.random.rand(5)
        delta = 1e-6
        p1 = Point(x1,y1)
        p0 = p2 = Point(x2,y2)
        res = Point(x1+x2,y1+y2)
        self.assertAlmostEqual((p1 + p2).x, res.x, delta=delta)
        self.assertAlmostEqual((p1 + p2).y, res.y, delta=delta)
        p2 += p1
        self.assertAlmostEqual(p2.x, res.x, delta=delta)
        self.assertAlmostEqual(p2.y, res.y, delta=delta)
        p3 = p2 - p1
        self.assertAlmostEqual(p3.x, p0.x, delta=delta)
        self.assertAlmostEqual(p3.y, p0.y, delta=delta)
        p2 -= p1
        self.assertAlmostEqual(p2.x, p0.x, delta=delta)
        self.assertAlmostEqual(p2.y, p0.y, delta=delta)
        res2 = Point(x1*c,y1*c)
        self.assertAlmostEqual((p1 * c).x, res2.x, delta=delta)
        self.assertAlmostEqual((p1 * c).y, res2.y, delta=delta)
        p1 *= c
        self.assertAlmostEqual(p1.x, res2.x, delta=delta)
        self.assertAlmostEqual(p1.y, res2.y, delta=delta)
        l = math.sqrt(p1.x*p1.x + p1.y*p1.y)
        self.assertAlmostEqual(p1.length(), np.float32(l), delta=delta)
        p1 = Point.vector_from_angle(0,1.0)
        p2 = Point.vector_from_angle(math.pi*0.5,1.0)
        a1 = Point.scalar_angle(p1, p2)
        self.assertAlmostEqual(a1, math.pi*0.5, delta=delta)
        p1 = Point.vector_from_angle(math.pi*0.1,1.0)
        p2 = Point.vector_from_angle(math.pi*0.4,1.0)
        a1 = Point.scalar_angle(p1, p2)
        self.assertAlmostEqual(a1, math.pi*0.3, delta=delta)


    def test_sz_base_ops(self):
        x1,y1,x2,y2,c = np.random.rand(5)
        delta = 1e-6
        sz1 = Size(cx=x1,cy=y1)
        sz0 = sz2 = Size(cx=x2,cy=y2)
        res = Size(cx=x1+x2,cy=y1+y2)
        self.assertAlmostEqual((sz1 + sz2).cx, res.cx, delta=delta)
        self.assertAlmostEqual((sz1 + sz2).cy, res.cy, delta=delta)
        sz2 += sz1
        self.assertAlmostEqual(sz2.cx, res.cx, delta=delta)
        self.assertAlmostEqual(sz2.cy, res.cy, delta=delta)
        sz3 = sz2 - sz1
        self.assertAlmostEqual(sz3.cx, sz0.cx, delta=delta)
        self.assertAlmostEqual(sz3.cy, sz0.cy, delta=delta)
        sz2 -= sz1
        self.assertAlmostEqual(sz2.cx, sz0.cx, delta=delta)
        self.assertAlmostEqual(sz2.cy, sz0.cy, delta=delta)
        res2 = Size(sz=sz1*c)
        self.assertAlmostEqual((sz1 * c).cx, res2.cx, delta=delta)
        self.assertAlmostEqual((sz1 * c).cy, res2.cy, delta=delta)
        sz1 *= c
        self.assertAlmostEqual(sz1.cx, res2.cx, delta=delta)
        self.assertAlmostEqual(sz1.cy, res2.cy, delta=delta)

    def test_rc_empty_op(self):
        rects = [
            Rect(x=1, y=2, w=-1, h=1),
            Rect(x=1, y=2, w=1, h=-1),
            Rect(x=1, y=2, w=0, h=1),
            Rect(x=1, y=2, w=1, h=0),
            Rect(x=1, y=2, sz=Size(cx=-1, cy=0)),
            Rect(x=1, y=2, sz=Size(cx=-1, cy=1)),
            Rect(x=1, y=2, sz=Size(cx=1, cy=-1)),
            Rect(x=1, y=2, sz=Size(cx=1, cy=0)),
            Rect(top_left=Point(2, 2), bottom_right=Point(2, 4)),
            Rect(top_left=Point(2, 2), bottom_right=Point(1, 4)),
            Rect(top_left=Point(2, 2), bottom_right=Point(4, 2)),
            Rect(top_left=Point(2, 2), bottom_right=Point(4, 1)),
        ]
        for rc in rects:
            self.assertTrue(rc.is_empty())
            self.assertEqual(rc.size, Size(cx=0,cy=0))

    def test_rc_properties(self):
        rects = [
            Rect(x=1, y=2, w=10, h=10),
            Rect(x=3, y=4, sz=Size(cx=10, cy=10)),
            Rect(top_left=Point(2, 2), bottom_right=Point(5, 5)),
        ]
        for rc in rects:
            dl,dt,dr,db = np.float32(np.random.rand(4))
            x = rc.x
            rc.x = x - dl
            self.assertEqual(rc.x, x - dl)
            rc.x = x
            self.assertEqual(rc.x, x)
            rc.left = x + dl
            self.assertEqual(rc.left, x + dl)
            rc.left = x
            self.assertEqual(rc.left, x)

            y = rc.y
            rc.y = y - dt
            self.assertEqual(rc.y, y - dt)
            rc.y = y
            self.assertEqual(rc.y, y)
            rc.top = y + dt
            self.assertEqual(rc.top, y + dt)
            rc.top = y
            self.assertEqual(rc.top, y)

            w = rc.w
            rc.w = w - dr
            self.assertEqual(rc.w, w - dr)
            rc.w = w
            self.assertEqual(rc.w, w)
            r = rc.right
            rc.right = r + dr
            self.assertEqual(rc.right, r + dr)
            rc.right = r
            self.assertEqual(rc.right, r)

            h = rc.h
            rc.h = h - db
            self.assertEqual(rc.h, h - db)
            rc.h = h
            self.assertEqual(rc.h, h)
            b = rc.bottom
            rc.bottom = b + db
            self.assertEqual(rc.bottom, b + db)
            rc.bottom = b
            self.assertEqual(rc.bottom, b)

            tl = rc.top_left
            rc.top_left = tl - Point(dl,dt)
            self.assertEqual(rc.top_left, tl - Point(dl,dt))
            rc.top_left = tl
            self.assertEqual(rc.top_left, tl)
            tl += Point(dl,dt)
            self.assertNotEqual(rc.top_left, tl)

            tr = rc.top_right
            rc.top_right = tr - Point(dr,dt)
            self.assertEqual(rc.top_right, tr - Point(dr,dt))
            rc.top_right = tr
            self.assertEqual(rc.top_right, tr)
            tr += Point(dr,dt)
            self.assertNotEqual(rc.top_right, tr)

            bl = rc.bottom_left
            rc.bottom_left = bl - Point(dl,db)
            self.assertEqual(rc.bottom_left, bl - Point(dl,db))
            rc.bottom_left = bl
            self.assertEqual(rc.bottom_left, bl)
            bl += Point(dl,dt)
            self.assertNotEqual(rc.bottom_left, bl)

            br = rc.bottom_right
            rc.bottom_right = br - Point(dr,db)
            self.assertEqual(rc.bottom_right, br - Point(dr,db))
            rc.bottom_right = br
            self.assertEqual(rc.bottom_right, br)
            br += Point(dr,db)
            self.assertNotEqual(rc.bottom_right, br)

            sz = rc.size
            rc.size = sz - Size(cx=dr, cy=db)
            self.assertEqual(rc.size, sz - Size(cx=dr, cy=db))
            rc.size = sz
            self.assertEqual(rc.size, sz)
            sz += Size(cx=dr, cy=db)
            self.assertNotEqual(rc.size, sz)

    def test_rect_union_op(self):
        rc1 = Rect(top_left=Point(1,1), bottom_right=Point(3,3))
        rc2 = Rect(top_left=Point(2,2), bottom_right=Point(4,4))
        rcU = Rect(top_left=Point(1,1), bottom_right=Point(4,4))
        rcI = Rect(top_left=Point(2,2), bottom_right=Point(3,3))
        self.assertEqual(rc1.union(rc2), rcU)
        self.assertEqual(rc1.intersect(rc2), rcI)

if __name__ == '__main__':
    unittest.main()
