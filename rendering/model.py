'''
This code is copied from
https://github.com/wadimkehl/ssd-6d.git
'''

import os
import numpy as np
from scipy.spatial.distance import pdist
from plyfile import PlyData
import cv2
from vispy import gloo

class Model3D:
    def __init__(self, file_to_load=None):
        self.vertices = None
        self.centroid = None
        self.indices = None
        self.colors = None
        self.texcoord = None
        self.texture = None
        self.collated = None
        self.vertex_buffer = None
        self.index_buffer = None
        self.bb = None
        self.bb_vbuffer = None
        self.bb_ibuffer = None
        self.diameter = None
        if file_to_load:
            self.load(file_to_load)

    def _compute_bbox(self,color_type=0):

        self.bb = []
        minx, maxx = min(self.vertices[:, 0]), max(self.vertices[:, 0])
        miny, maxy = min(self.vertices[:, 1]), max(self.vertices[:, 1])
        minz, maxz = min(self.vertices[:, 2]), max(self.vertices[:, 2])
        self.bb.append([minx, miny, minz])
        self.bb.append([minx, maxy, minz])
        self.bb.append([minx, miny, maxz])
        self.bb.append([minx, maxy, maxz])
        self.bb.append([maxx, miny, minz])
        self.bb.append([maxx, maxy, minz])
        self.bb.append([maxx, miny, maxz])
        self.bb.append([maxx, maxy, maxz])
        self.bb = np.asarray(self.bb, dtype=np.float32)
        #self.diameter = max(pdist(self.bb, 'euclidean'))

        # Set up rendering data
        if(color_type==0):
            colors = [[1, 0, 0],[1, 1, 0], [0, 1, 0], [0, 1, 1],
                      [0, 0, 1], [0, 1, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]
        elif(color_type==1):
            colors = [[0, 0, 1],[0, 0, 1], [0, 0, 1], [0, 0, 1],
                      [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]]
        elif(color_type==2):
            colors = [[0, 1, 0],[0, 1, 0], [0, 1, 0], [0, 1, 0],
                      [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]]
        elif(color_type==3):
            colors = [[1, 1, 1],[1, 1, 1], [1, 1, 1], [1, 1, 1],
                      [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]
        else:
            colors = [[0, 1,0],[0, 1, 0], [0, 1, 0], [0, 1, 0],
                      [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]]

        indices = [0, 1, 0, 2, 3, 1, 3, 2,
                   4, 5, 4, 6, 7, 5, 7, 6,
                   0, 4, 1, 5, 2, 6, 3, 7]

        vertices_type = [('a_position', np.float32, 3), ('a_color', np.float32, 3)]
        collated = np.asarray(list(zip(self.bb, colors)), vertices_type)
        self.bb_vbuffer = gloo.VertexBuffer(collated)
        self.bb_ibuffer = gloo.IndexBuffer(indices)

    def load(self, mesh, demean=False, scale=1.0):
        self.vertices = np.zeros((len(mesh.vertices), 3))
        self.vertices[:, 0] = np.array(mesh.vertices[:, 0])
        self.vertices[:, 1] = np.array(mesh.vertices[:, 1])
        self.vertices[:, 2] = np.array(mesh.vertices[:, 2])
        self.vertices *= scale
        self.centroid = np.mean(self.vertices, 0)

        if demean:
            self.centroid = np.zeros((1, 3), np.float32)
            self.vertices -= self.centroid

        self._compute_bbox()

        #self.indices = np.asarray(list(range(len(mesh.vertices))), np.uint32)
        #print(self.indices)
        self.indices = np.asarray(mesh.faces, np.uint32)

        self.colors = 0.5 * np.ones((len(mesh.vertices), 3))

        print('Loading with vertex colors')
        self.colors[:, 0] = np.array(mesh.visual.vertex_colors[:, 0])
        self.colors[:, 1] = np.array(mesh.visual.vertex_colors[:, 1])
        self.colors[:, 2] = np.array(mesh.visual.vertex_colors[:, 2])
        self.colors /= 255.0

        vertices_type = [('a_position', np.float32, 3), ('a_color', np.float32, 3)]
        self.collated = np.asarray(list(zip(self.vertices, self.colors)), vertices_type)

        self.vertex_buffer = gloo.VertexBuffer(self.collated)
        #self.index_buffer = gloo.IndexBuffer(self.indices.flatten())
        self.index_buffer = gloo.IndexBuffer(self.indices.flatten())
