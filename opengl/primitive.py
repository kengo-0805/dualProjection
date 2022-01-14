import numpy as np
import OpenGL.GL as gl
import OpenGL.GLU as glu
import glfw


class Cube:
	__vertices = (
		(-0.5,-0.5, 0.5 ),
		( 0.5,-0.5, 0.5 ),
		( 0.5, 0.5, 0.5 ),
		(-0.5, 0.5, 0.5 ),
		( 0.5,-0.5,-0.5 ),
		(-0.5,-0.5,-0.5 ),
		(-0.5, 0.5,-0.5 ),
		( 0.5, 0.5,-0.5 ))

	__faces = (
		(0, 1, 2, 3),
		(4, 5, 6, 7),
		(1, 4, 7, 2),
		(5, 0, 3, 6),
		(3, 2, 7, 6),
		(1, 0, 5, 4),
	)

	__normals = (
		( 0.0, 0.0, 1.0),
		( 0.0, 0.0,-1.0),
		( 1.0, 0.0, 0.0),
		(-1.0, 0.0, 0.0),
		( 0.0, 1.0, 0.0),
		( 0.0,-1.0, 0.0))


	def __init__(self,
				 scale=(1.0, 1.0, 1.0),
				 position=(0.0, 0.0, 0.0),
				 diffuse=(1.0, 1.0, 1.0),
				 ambient=(0.3, 0.3, 0.3),
				 specular=(0.1, 0.1, 0.1),
				 shininess = 3.0):

		self.position = position
		self.scale = scale
		self.vertices = self.calc_vertices()
		self.front_vertices = self.vertices[0:4]

		self.diffuse = diffuse
		self.ambient = ambient
		self.specular = specular
		self.shininess = shininess


	def draw(self):
		gl.glMatrixMode(gl.GL_MODELVIEW)

		gl.glMaterialfv(gl.GL_FRONT_AND_BACK, gl.GL_DIFFUSE, self.diffuse)
		gl.glMaterialfv(gl.GL_FRONT_AND_BACK, gl.GL_AMBIENT, self.ambient)
		gl.glMaterialfv(gl.GL_FRONT_AND_BACK, gl.GL_SPECULAR, self.specular)
		gl.glMaterialfv(gl.GL_FRONT_AND_BACK, gl.GL_SHININESS, self.shininess)

		gl.glPushMatrix()
		gl.glTranslatef(*self.position)
		gl.glScalef(*self.scale)

		for i, f in enumerate(Cube.__faces):
			gl.glNormal3fv(Cube.__normals[i])
			gl.glBegin(gl.GL_QUADS)
			gl.glVertex3fv(Cube.__vertices[f[0]])
			gl.glVertex3fv(Cube.__vertices[f[1]])
			gl.glVertex3fv(Cube.__vertices[f[2]])
			gl.glVertex3fv(Cube.__vertices[f[3]])
			gl.glEnd()
		gl.glPopMatrix()


	def calc_vertices(self):
		vertices = np.empty((len(Cube.__vertices), 3))
		for i, v in enumerate(Cube.__vertices):
			print("cube", v)
			print("scale", self.scale)
			print("position", self.position)
			vertices[i] = np.array(v) * np.array(self.scale) + np.array(self.position)
			print("v", vertices[i])
		print("vertices", vertices)
		return vertices


class Sphere:
	def __init__(self,
				 radius=1.0,
				 slices=16,
				 stacks=16,
				 position=(0.0, 0.0, 0.0),
				 diffuse=(1.0, 1.0, 1.0),
				 ambient=(0.3, 0.3, 0.3),
				 specular=(0.1, 0.1, 0.1),
				 shininess = 3.0):

		self.radius = radius
		self.slices = slices
		self.stacks = stacks
		self.position = position

		self.diffuse = diffuse
		self.ambient = ambient
		self.specular = specular
		self.shininess = shininess

		self.sphere = glu.gluNewQuadric()
		glu.gluQuadricDrawStyle(self.sphere, glu.GLU_FILL)


	def draw(self):
		gl.glMatrixMode(gl.GL_MODELVIEW)

		gl.glMaterialfv(gl.GL_FRONT_AND_BACK, gl.GL_DIFFUSE, self.diffuse)
		gl.glMaterialfv(gl.GL_FRONT_AND_BACK, gl.GL_AMBIENT, self.ambient)
		gl.glMaterialfv(gl.GL_FRONT_AND_BACK, gl.GL_SPECULAR, self.specular)
		gl.glMaterialfv(gl.GL_FRONT_AND_BACK, gl.GL_SHININESS, self.shininess)

		gl.glPushMatrix()
		gl.glTranslatef(*self.position)
		glu.gluSphere(self.sphere, self.radius, self.slices, self.stacks)
		gl.glPopMatrix()


class Polygon:
	def __init__(self,
				 vertices,
				 normal=( 0.0, 0.0, 1.0),
				 diffuse=(1.0, 1.0, 1.0),
				 ambient=(0.3, 0.3, 0.3),
				 specular=(0.1, 0.1, 0.1),
				 shininess = 3.0):

		self.vertices = vertices
		self.normal = normal
		self.diffuse = diffuse
		self.ambient = ambient
		self.specular = specular
		self.shininess = shininess


	def draw(self):
		gl.glMatrixMode(gl.GL_MODELVIEW)

		gl.glMaterialfv(gl.GL_FRONT_AND_BACK, gl.GL_DIFFUSE, self.diffuse)
		gl.glMaterialfv(gl.GL_FRONT_AND_BACK, gl.GL_AMBIENT, self.ambient)
		gl.glMaterialfv(gl.GL_FRONT_AND_BACK, gl.GL_SPECULAR, self.specular)
		gl.glMaterialfv(gl.GL_FRONT_AND_BACK, gl.GL_SHININESS, self.shininess)

		gl.glPushMatrix()
		gl.glBegin(gl.GL_POLYGON)
		gl.glNormal3fv(self.normal)
		for v in self.vertices:
			gl.glVertex3fv(v)
		gl.glEnd()
		gl.glPopMatrix()
