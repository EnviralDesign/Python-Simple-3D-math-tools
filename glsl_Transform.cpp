uniform int maxItrs;
uniform int numInputs;
uniform int aux;
// uniform vec2 RES;

uniform samplerBuffer translates;
uniform samplerBuffer rotates;
uniform samplerBuffer scales;
uniform samplerBuffer cam;
uniform samplerBuffer misc;

vec4 color;
int uv_set;
float mask;

#define PI 3.1415926535


///////////////////////////////////////////////////////////////////////////////
///////////////////////////// Start of user functions /////////////////////////
///////////////////////////////////////////////////////////////////////////////

// Full Xform Function:

// NOTE: Were not using this function anywhere right now. we might want it later though.
vec3 Transform(vec3 p, vec3 t, vec3 r, vec3 s)
{
	float px = p.x;
	float py = p.y;
	float pz = p.z;
	
	float tx = t.x;
	float ty = t.y;
	float tz = t.z;
	
	float rx = r.x;
	float ry = r.y;
	float rz = r.z;
	
	float sx = s.x;
	float sy = s.y;
	float sz = s.z;
	
	float crx = cos(rx*PI/180);
	float cry = cos(ry*PI/180);
	float crz = cos(rz*PI/180);
	
	float srx = sin(rx*PI/180);
	float sry = sin(ry*PI/180);
	float srz = sin(rz*PI/180);
	
	px *= sx;
	py *= sy;
	pz *= sz;

	vec3 result = vec3(0.);
	result.x = (cry * crz) * px + (srx * sry * crz - srz * crx) * py + (crx * sry * crz + srx * srz) * pz + tx;
	result.y = (cry * srz) * px + (srz * srx * sry + crx * crz) * py + (crx * sry * srz - srx * crz) * pz + ty;
	result.z = -sry * px + (srx * cry) * py + (crx * cry) * pz + tz;
	
	return result;
	
}

// returns a scale only matrix with the supplied scale values.
mat4 ScaleMatrix(vec3 s)
{
	mat4 scaleMat = mat4(1.0);
	scaleMat[0][0] = s.x;
	scaleMat[1][1] = s.y;
	scaleMat[2][2] = s.z;
	return scaleMat;
}

// returns a translation only matrix with the supplied translation values.
mat4 TranslateMatrix(vec3 t)
{
	mat4 transMat = mat4(1.0);
	transMat[3][0] = t.x;
	transMat[3][1] = t.y;
	transMat[3][2] = t.z;
	return transMat;
}

// returns a rotate only matrix with the supplied rotate values.
mat4 RotateMatrix(vec3 r)
{
	
	float dr = PI/180;
	float A = cos(r.x*dr);
	float B = sin(r.x*dr);
	float C = cos(r.y*dr);
	float D = sin(r.y*dr);
	float E = cos(r.z*dr);
	float F = sin(r.z*dr);
	float AD = A * D;
	float BD = B * D;
	
	mat4 rotMat = mat4(1.0);
	
	rotMat[0][0] =  C * E;
	rotMat[1][0] = -C * F;
	rotMat[2][0] = D;
	rotMat[3][0] = 0;
	
	rotMat[0][1] = ( BD*E)+(A*F);
	rotMat[1][1] = (-BD*F)+(A*E);
	rotMat[2][1] = -B * C;
	rotMat[3][1] = 0;
	
	rotMat[0][2] = (-AD*E) + (B*F);
	rotMat[1][2] = ( AD*F) + (B*E);
	rotMat[2][2] = A * C;
	rotMat[3][2] = 0;
	
	rotMat[3][3] = 1;
	
	return rotMat;
}


// returns a rotate only matrix with the rotation order reversed. (as Z.Y.X)
// This is meant for camera / projector so projected points always go the right way.
mat4 RotateMatrix_r(vec3 r)
{
	
	mat3 step1;
	mat3 step2;
	
	float dr = PI/180;
	float A = cos(r.x*dr);
	float B = sin(r.x*dr);
	float C = cos(r.y*dr);
	float D = sin(r.y*dr);
	float E = cos(r.z*dr);
	float F = sin(r.z*dr);
	
	// generate X rotation matrix.
	mat3 rotMatX = mat3(1.0);
	
	rotMatX[0][0] = 1;
	rotMatX[1][0] = 0;
	rotMatX[2][0] = 0;
	
	rotMatX[0][1] = 0;
	rotMatX[1][1] = A;
	rotMatX[2][1] = -B;
	
	rotMatX[0][2] = 0;
	rotMatX[1][2] = B;
	rotMatX[2][2] = A;
	
	// generate Y rotation matrix.
	mat3 rotMatY = mat3(1.0);
	
	rotMatY[0][0] = C;
	rotMatY[1][0] = 0;
	rotMatY[2][0] = D;
	
	rotMatY[0][1] = 0;
	rotMatY[1][1] = 1;
	rotMatY[2][1] = 0;
	
	rotMatY[0][2] = -D;
	rotMatY[1][2] = 0;
	rotMatY[2][2] = C;
	
	// generate Z rotation matrix.
	mat3 rotMatZ = mat3(1.0);
	
	rotMatZ[0][0] = E;
	rotMatZ[1][0] = -F;
	rotMatZ[2][0] = 0;
	
	rotMatZ[0][1] = F;
	rotMatZ[1][1] = E;
	rotMatZ[2][1] = 0;
	
	rotMatZ[0][2] = 0;
	rotMatZ[1][2] = 0;
	rotMatZ[2][2] = 1;
	
	// if we were rotating an object, that was not a camera. we'd use the standard X.Y.Z rotation order.
	// step1 = rotMatX * rotMatY;
	// step2 = step1 * rotMatZ;
	
	// since we're probably using this to rotate a camera, or rather the points that represent our geo
	// or leds or whatever.. we actually want to rotate by the inverse and reverse order.
	step1 = rotMatZ * rotMatY;
	step2 = step1 * rotMatX;
	
	
	// generate the final 4x4 matrix we will hold our X.Y.Z matrix. or Z.Y.X? only time will tell.
	mat4 finalMat = mat4(1.0);
	
	// assign the ste2 3x3 mat to the 4x4 in the areas that require it.
	finalMat[0][0] = step2[0][0];
	finalMat[1][0] = step2[1][0];
	finalMat[2][0] = step2[2][0];
	finalMat[3][0] = 0;
	
	finalMat[0][1] = step2[0][1];
	finalMat[1][1] = step2[1][1];
	finalMat[2][1] = step2[2][1];
	finalMat[3][1] = 0;
	
	finalMat[0][2] = step2[0][2];
	finalMat[1][2] = step2[1][2];
	finalMat[2][2] = step2[2][2];
	finalMat[3][2] = 0;
	
	finalMat[0][3] = 0;
	finalMat[1][3] = 0;
	finalMat[2][3] = 0;
	finalMat[3][3] = 1;
	
	
	return finalMat;
}


// returns a perspective projection only matrix with the supplied camera values.
mat4 PerspectiveProjectionMatrix(vec3 c)
{
	// width/height are arbitrary except that they must be the same.
	float width = 500;
	float height = 500;
	float fov = c.x;
	float near = c.y;
	float far = c.z;
	
	float fovTan = tan( (fov/2) * (PI/180) );
	
	float right = fovTan * near;
	float left = right * -1;
	float top = fovTan * near;
	float bottom = top * -1;
	float aspect = width / height;
	
	mat4 projMat = mat4(1.0);

	projMat[0][0] = ( (2*near) / (right-left) );
	projMat[1][0] = 0;
	projMat[2][0] = ( (right+left) / (right-left) );
	projMat[3][0] = 0;
	
	projMat[0][1] = 0;
	projMat[1][1] = ( (2*near) / (top-bottom) )*aspect;
	projMat[2][1] = ( (top+bottom) / (top-bottom) );
	projMat[3][1] = 0;
	
	projMat[0][2] = 0;
	projMat[1][2] = 0;
	projMat[2][2] = ( (-(far+near)) / (far-near) );
	projMat[3][2] = ( (-2*far*near) / (far-near) );
	
	projMat[0][3] = 0;
	projMat[1][3] = 0;
	projMat[2][3] = -1;
	projMat[3][3] = 0;
	
	return projMat;
}


// given a full set of transforms, returns a matrix representing that full transform.
mat4 Persp_CameraViewMatrix( vec3 t, vec3 r, vec3 s, vec3 c )
{
	// Create identity matrix. this is the default non-translated coordinate.
	mat4 xform = mat4(1.0);
	
	// multiply translate, rotate, and scale to get the TRANSFORM matrix.
	// xform = TranslateMatrix(t) * RotateMatrix(r) * ScaleMatrix(s);
	xform = TranslateMatrix(t) * RotateMatrix_r(r) * ScaleMatrix(s);
	// xform = RotateMatrix(r);
	
	// inverse the matrix.
	xform = inverse(xform);
	
	// multiply the view matrix by the projection matrix now.
	xform = PerspectiveProjectionMatrix(c) * xform;
	
	return xform;
}





// returns an Orthographic projection only matrix with the supplied camera values.
mat4 OrthographicProjectionMatrix(vec3 c)
{
	// width/height are arbitrary except that they must be the same.
	float width = 500;
	float height = 500;
	float fov = c.x/2; // fov is actually ortho width, in the orthoghraphic projection matrix.
	float near = c.y;
	float far = c.z;
	
	float right = fov;
	float left = right * -1;
	float top = fov;
	float bottom = top * -1;
	float aspect = width / height;
	
	mat4 projMat = mat4(1.0);

	projMat[0][0] = 2. / ( right - left );
	projMat[1][0] = 0;
	projMat[2][0] = 0;
	projMat[3][0] = 0;
	
	projMat[0][1] = 0;
	projMat[1][1] = 2. / ( top - bottom );
	projMat[2][1] = 0;
	projMat[3][1] = 0;
	
	projMat[0][2] = 0;
	projMat[1][2] = 0;
	projMat[2][2] = -2 / (far-near);
	// projMat[3][2] = ( ( far + near ) / (far - near) ) * -1;
	projMat[3][2] = 0;
	
	projMat[0][3] = ((right+left) / (right-left)) * -1;
	projMat[1][3] = ((top+bottom) / (top-bottom)) * -1;
	projMat[2][3] = ((far+near)   / (far-near))   * -1;
	projMat[3][3] = 1;
	
	return projMat;
}


// given a full set of transforms, returns a matrix representing that full transform.
mat4 Ortho_CameraViewMatrix( vec3 t, vec3 r, vec3 s, vec3 c )
{
	// Create identity matrix. this is the default non-translated coordinate.
	mat4 xform = mat4(1.0);
	
	// multiply translate, rotate, and scale to get the TRANSFORM matrix.
	// xform = TranslateMatrix(t) * RotateMatrix(r) * ScaleMatrix(s);
	xform = TranslateMatrix(t) * RotateMatrix_r(r) * ScaleMatrix(s);
	// xform = RotateMatrix(r);
	
	// inverse the matrix.
	xform = inverse(xform);
	
	// multiply the view matrix by the projection matrix now.
	xform = OrthographicProjectionMatrix(c) * xform;
	
	return xform;
}



// given a full set of transforms, returns a matrix representing just the world xform.
mat4 World_ViewMatrix( vec3 t, vec3 r, vec3 s)
{
	// Create identity matrix. this is the default non-translated coordinate.
	mat4 xform = mat4(1.0);
	
	// multiply translate, rotate, and scale to get the TRANSFORM matrix.
	// xform = TranslateMatrix(t) * RotateMatrix(r) * ScaleMatrix(s);
	xform = TranslateMatrix(t) * RotateMatrix_r(r) * ScaleMatrix(s);
	// xform = RotateMatrix(r);
	
	// inverse the matrix.
	xform = inverse(xform);
	
	return xform;
}




float isInRange(float a , float b , float c)
{
	return (1.0-step(c, a)) * step(b, a);
}


//	returns a masked version given an input channel and it's UV channel.
vec4 drawRow(vec4 color, int rowID , int uvSet)
{
	
	ivec2 	RES 			= ivec2(uTDOutputInfo.res.zw); // get the res of the glsl top.
	ivec2 	PixelCoords2D 	= ivec2(vUV.st * vec2(RES)); // get the 2d pixel coords (non normalized, whole int)
	int 	PixelCoords1D 	= PixelCoords2D.x + (PixelCoords2D.y * int(RES.x)); // (get a 1d int version of the uvs, counts pixels.)
	
	// get the coordinates the new way.
	// vec4 fetchedColor = texelFetch(sTD2DInputs[ 0 ], PixelCoords2D, 0); // OLD, used before 2dTextureArray.
	vec4 fetchedColor = texelFetch(sTD2DArrayInputs[ 0 ], ivec3(PixelCoords2D , uvSet), 0);
	
	int chanToSample = int(fetchedColor.a); 
	
	
	
	float PIX_COUNTER = int(PixelCoords1D / aux);
	
	
	
	// we get another copy of the world coordinates, we need to test these to see
	// if they are in front of the camera or behind.
	vec4 pixCoords;
	pixCoords.xyz = fetchedColor.xyz;
	pixCoords.w = 1;
	
	// Important! we need this to be 1, before we mult against matrix so we can
	// perspective divide properly later. this may or may not be set by user in TD 
	// So lets do it here just to be safe.
	fetchedColor.w = 1; 
	
	// get camera and matrix info from our chops.
	vec3 t = texelFetch(translates, rowID).xyz; // tx, ty, tz
	vec3 r = texelFetch(rotates, rowID).xyz; // rx, ry, rz
	vec3 s = texelFetch(scales, rowID).xyz; // sx, sy, sz
	vec4 c = texelFetch(cam, rowID).xyzw; // camera specific info (fov, near, far)
	
	vec4 misc = texelFetch(misc, rowID).xyzw; // camera specific info (fov, near, far)
	
	
	// now we transform our world coordinates against our camera's world transform.
	pixCoords = World_ViewMatrix(t,r,s) * pixCoords;
	
	// our coordinates, if in front of our camera should be a negative value.
	// we could incorporate near/far clipping here as well... but for now we aren't.
	bvec2 bvecTest = lessThan(vec2(pixCoords.zz), vec2(0,0))  ;
	
	// uncomment this to mask the pix coords by the front/back bvec test.
	// pixCoords *= float(bvecTest);
	
	float pMASK = 1;
	
	
	// perform perspective camera.
	if (c.w == 0)
	{
		// multiply our view matrix by our current world space coordinates.
		fetchedColor = Persp_CameraViewMatrix(t,r,s,c.xyz) * fetchedColor;

		// perspective divide - very important!
		fetchedColor.x /= fetchedColor.w;
		fetchedColor.y /= fetchedColor.w;
		fetchedColor.z /= fetchedColor.w;
		
		// put coordinates from NDC to 0-1 texture space.
		// AKA from -1:1 to 0:1
		fetchedColor.rg = fetchedColor.rg*.5+.5;
		fetchedColor.b = 1;
		
	}
	
	// perform orthographic camera.
	else if (c.w == 1)
	{
		// multiply our view matrix by our current world space coordinates.
		fetchedColor = Ortho_CameraViewMatrix(t,r,s,c.xyz) * fetchedColor;
		
		// put coordinates from NDC to 0-1 texture space.
		// AKA from -1:1 to 0:1
		fetchedColor.rg = fetchedColor.rg*.5+.5;
		fetchedColor.b = 1;
		
	}
	
	// perform chase camera.
	else if (c.w == 2)
	{
		// calculate our regular orthographic matrix...
		fetchedColor = Ortho_CameraViewMatrix(t,r,s,vec3(100,c.yz)) * fetchedColor;
		fetchedColor.rg = fetchedColor.rg*.5+.5;
		fetchedColor.b = 1;
		
		float xMin = 0;
		if (fetchedColor.r > 0){xMin = 1;}
		float xMax = 0;
		if (fetchedColor.r < 1){xMax = 1;}
		float yMin = 0;
		if (fetchedColor.g > 0){yMin = 1;}
		float yMax = 0;
		if (fetchedColor.g < 1){yMax = 1;}
		
		float typeMask = xMin * xMax * yMin * yMax;
		// pMASK = xMin * xMax * yMin * yMax;
		
		// multiply our view matrix by our current world space coordinates.
		fetchedColor.r = PIX_COUNTER / c.x;
		fetchedColor.g = vUV.t;
		fetchedColor.r = mix(fetchedColor.r , fetchedColor.r+1 , 1-typeMask);
	}
	
	
	
	// error color AKA black
	if (misc.x == 0)
	{
		// do nothing, error color AKA black outside already works as default.
	}
	
	// Extend mode AKA Hold
	else if (misc.x == 1)
	{
		fetchedColor.rg = clamp(fetchedColor.rg,vec2(0.),vec2(0.9999));
	}
	
	// Cycle mode aka Repeat
	else if (misc.x == 2)
	{
		fetchedColor.rg = mod(fetchedColor.rg,1);
	}
	
	// Mirror Mode (Flips back and forth)
	else if (misc.x == 3)
	{
		fetchedColor.rg = fract(fetchedColor.rg*0.5)*2.0;
		fetchedColor.rg = 1 - abs(fetchedColor.rg-1);
	}
	
	
	fetchedColor *= float(bvecTest) * pMASK;
	fetchedColor.b = float(bvecTest) * pMASK;
	
	
	int mixer = 0;
	if (rowID == uTDCurrentDepth)
		mixer = 1;
	
	
	fetchedColor = mix( vec4(0) , fetchedColor , mixer );
	fetchedColor.a = mix( -1 , chanToSample , float(bvecTest) );
	
	vec4 newColor = vec4(-1);
	
	if (fetchedColor.a != -1)
		newColor = fetchedColor;
	
	
	// newColor.rgb = pixCoords.xyz;
	
	return newColor;
	// return fetchedColor;
	// return vec4(chanToSample);
	// return pixCoords;
	// return vec4(float(bvecTest),0,0,0);
}


// out vec4 fragColor;
 layout(location = 0) out vec4 UV_0; // standard coordinate uvs, the positions we see and create in the editor.
 layout(location = 1) out vec4 UV_1; // point consolidate uvs, every pix coord is identical as the fixtures worldspace origin.
 layout(location = 2) out vec4 UV_2; // random shuffle uvs, all pix coords in a fixture have been shuffled amongst themselves.
 layout(location = 3) out vec4 UV_3; // actual UV coordinates.. these are the default coordinates replaced with custom UV coords when they are present.
void main()
{

	// /*
	//////////////////////////////////////////////////////////
	///////////////////// UV SET 0 ///////////////////////////
	//////////////////////////////////////////////////////////
	
	color = vec4(0,0,0,1); // init color to zeros.
	uv_set = 0; // set uv set for this pass.
	
	// for each input, draw it as a row, while applying the projection matrix math.
	for( int i=0; i<numInputs; i++ )
	{
		if (i < maxItrs){
			color = drawRow(color, uTDCurrentDepth , uv_set); 
		}
	}
	UV_0 = TDOutputSwizzle(color);
	
	
	//////////////////////////////////////////////////////////
	///////////////////// UV SET 1 ///////////////////////////
	//////////////////////////////////////////////////////////
	
	color = vec4(0,0,0,1); // init color to zeros.
	uv_set = 1; // set uv set for this pass.
	
	// for each input, draw it as a row, while applying the projection matrix math.
	for( int i=0; i<numInputs; i++ )
	{
		if (i < maxItrs){
			color = drawRow(color, uTDCurrentDepth , uv_set); 
		}
	}
	UV_1 = TDOutputSwizzle(color);
	
	
	//////////////////////////////////////////////////////////
	///////////////////// UV SET 2 ///////////////////////////
	//////////////////////////////////////////////////////////
	
	color = vec4(0,0,0,1); // init color to zeros.
	uv_set = 2; // set uv set for this pass.
	
	// for each input, draw it as a row, while applying the projection matrix math.
	for( int i=0; i<numInputs; i++ )
	{
		if (i < maxItrs){
			color = drawRow(color, uTDCurrentDepth , uv_set); 
		}
	}
	UV_2 = TDOutputSwizzle(color);
	
	
	//////////////////////////////////////////////////////////
	///////////////////// UV SET 3 ///////////////////////////
	//////////////////////////////////////////////////////////
	
	color = vec4(0,0,0,1); // init color to zeros.
	uv_set = 3; // set uv set for this pass.
	
	// for each input, draw it as a row, while applying the projection matrix math.
	for( int i=0; i<numInputs; i++ )
	{
		if (i < maxItrs){
			color = drawRow(color, uTDCurrentDepth , uv_set); 
		}
	}
	UV_3 = TDOutputSwizzle(color);
	// */
	
	
	// ivec2 	RES 			= ivec2(uTDOutputInfo.res.zw); // get the res of the glsl top.
	// ivec2 	PixelCoords2D 	= ivec2(vUV.st * vec2(RES)); // get the 2d pixel coords (non normalized, whole int)
	// int 	PixelCoords1D 	= PixelCoords2D.x + (PixelCoords2D.y * int(RES.x)); // (get a 1d int version of the uvs, counts pixels.)
	
	// get the coordinates the new way.
	// vec4 fetchedColor = texelFetch(sTD2DArrayInputs[ 0 ], ivec3(PixelCoords2D , 2), 0);
	// int chanToSample = int(fetchedColor.a); 
	
	// UV_0 = vec4(int(fetchedColor.r));
}
