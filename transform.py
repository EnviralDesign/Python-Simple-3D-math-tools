import math

def apply(px, py, pz, tx, ty, tz, rx, ry, rz, sx, sy, sz):
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