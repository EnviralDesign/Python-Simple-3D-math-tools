import numpy as np
import math

import math

def apply(px, py, pz, tx, ty, tz, rx, ry, rz, sx, sy, sz): # plain python version.
	result = []
	for i in range(len(px)):
		crx, srx = math.cos(rx[i]*math.pi/180), math.sin(rx[i]*math.pi/180)
		cry, sry = math.cos(ry[i]*math.pi/180), math.sin(ry[i]*math.pi/180)
		crz, srz = math.cos(rz[i]*math.pi/180), math.sin(rz[i]*math.pi/180)

		px[i] *= sx[i]
		py[i] *= sy[i]
		pz[i] *= sz[i]

		x = (cry * crz) * px[i] + (srx * sry * crz - srz * crx) * py[i] + (crx * sry * crz + srx * srz) * pz[i] + tx[i]
		y = (cry * srz) * px[i] + (srz * srx * sry + crx * crz) * py[i] + (crx * sry * srz - srx * crz) * pz[i] + ty[i]
		z = -sry * px[i] + (srx * cry) * py[i] + (crx * cry) * pz[i] + tz[i]

		result.append((x, y, z))
	return result

def apply2(px, py, pz, tx, ty, tz, rx, ry, rz, sx, sy, sz): # numpy array math version.
	px = np.asarray(px)
	py = np.asarray(py)
	pz = np.asarray(pz)
	tx = np.asarray(tx)
	ty = np.asarray(ty)
	tz = np.asarray(tz)
	rx = np.asarray(rx)
	ry = np.asarray(ry)
	rz = np.asarray(rz)
	sx = np.asarray(sx)
	sy = np.asarray(sy)
	sz = np.asarray(sz)
	
	crx, srx = np.cos(rx*math.pi/180), np.sin(rx*math.pi/180)
	cry, sry = np.cos(ry*math.pi/180), np.sin(ry*math.pi/180)
	crz, srz = np.cos(rz*math.pi/180), np.sin(rz*math.pi/180)
	
	px *= sx
	py *= sy
	pz *= sz
	
	x = (cry * crz) * px + (srx * sry * crz - srz * crx) * py + (crx * sry * crz + srx * srz) * pz + tx
	y = (cry * srz) * px + (srz * srx * sry + crx * crz) * py + (crx * sry * srz - srx * crz) * pz + ty
	z = -sry * px + (srx * cry) * py + (crx * cry) * pz + tz
	
	result = np.column_stack((x, y, z)).tolist()
	return result
	