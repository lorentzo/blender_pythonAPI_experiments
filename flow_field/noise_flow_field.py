
## EXPERIMENT:
# ini positions
# noise scale factor
# offset vector
# noise type

## DEV:
# create modular code
# investigate curves
# investigate convex hull creation and bmesh
# numpy vectorize https://numpy.org/doc/stable/reference/generated/numpy.vectorize.html
# multiple clusters
# blender instancing further!
# Add forces to noise flow field


#########################################################
#   IMPORTS.
#########################################################

# Blender.
import bpy
from bpy import data
from bpy import context
import bmesh
from mathutils import noise
from mathutils import Matrix, Vector
from mathutils import Quaternion
from mathutils import Euler

# Numpy.
import numpy as np
from numpy.random import default_rng

#########################################################
#   Helper functions.
#########################################################

#
# Spline curve creation.
# Function adds linear curve between two points.
#
def add_spline_curve(splines=None,
                     resolution=16,
                     count=2, # no of knots
                     handle_type='FREE', # handle types: ['FREE', 'VECTOR', 'ALIGNED', 'AUTO']
                     origin=Vector((-1.0, 0.0, 0.0)),
                     destination=Vector((1.0, 0.0, 0.0))):
    
    spline = splines.new(type='BEZIER')
    spline.use_cyclic_u = False # we want lines
    spline.resolution_u = resolution
    knots = spline.bezier_points
    knots.add(count = count - 1)
    knots_range = range(0, count, 1)
    # Convert number of points in the array to a percent.
    to_percent = 1.0 / (count - 1)
    one_third = 1.0 / 3.0
    # Loop through bezier points.
    for i in knots_range:
        # Cache shortcut to current bezier point.
        knot = knots[i]
        # Calculate bezier point coordinate.
        step = i * to_percent
        knot.co = origin.lerp(destination, step)
        # Calculate left handle by subtracting 1/3 from step.
        step = (i - one_third) * to_percent
        knot.handle_left = origin.lerp(destination, step)
        # Calculate right handle by adding 1/3 to step.
        step = (i + one_third) * to_percent
        knot.handle_right = origin.lerp(destination, step)
        knot.handle_left_type = handle_type
        knot.handle_right_type = handle_type
    return spline


#########################################################
#   Preparing scene.
#########################################################

# TODO: MEMORY occupation is growing!

# TODO: remove data

# TODO: remove curve

# Remove all materials.
for material in data.materials:
    material.user_clear()
    data.materials.remove(material)

# Remove all meshes.
for mesh in data.meshes:
    data.meshes.remove(mesh, do_unlink=True)
    
# Remove all objects.
#for o in bpy.context.scene.objects:
#    o.select_set(True)    # select
#bpy.ops.object.delete()   # remove selected

# Remove collections.
# https://blender.stackexchange.com/questions/130124/blender-2-80-delete-collection-or-clear-the-intial-scene-in-scripting-mode
# Remove collections.
for c in context.scene.collection.children:
    context.scene.collection.children.unlink(c)
# Remove orphan collections.
for c in bpy.data.collections:
    if not c.users:
        bpy.data.collections.remove(c)



#########################################################
#   Preparing initial particle positions.
#########################################################

# Sample plane (xy) for source particles
n_source_particles = 10
xy_range = 1
z_height = 0
rng = default_rng()
source_particles = rng.standard_normal(n_source_particles * 3)
source_particles = np.reshape(source_particles, (n_source_particles, 3))
source_particles *= xy_range
source_particles[:,2] = z_height

# Regular plane (xy) grid
X,Y = np.mgrid[-10:10:2, -10:10:2]
xy = np.vstack((X.flatten(), Y.flatten())).T
Z = np.ones((np.shape(xy)[0], 1))
source_particles2 = np.hstack((xy, Z))

# Sample Sphere.
def sample_spherical(npoints, ndim=3):
    # https://stackoverflow.com/questions/33976911/generate-a-random-sample-of-points-distributed-on-the-surface-of-a-unit-sphere
    theta = np.random.randn(1, npoints)
    phi = np.random.randn(1, npoints)
    #for i in range(0, npoints):
    #vec /= np.linalg.norm(vec, axis=0)
    #return vec


## Quick testing of source particles.
#for sp in source_particles3:
#    bpy.ops.mesh.primitive_uv_sphere_add(location=sp) # do not use ops later!


#########################################################
#   Preparing materials.
#########################################################

## Create opaque material.
material_opaque = bpy.data.materials.new(name="Opaque")
material_opaque.use_nodes = True
material_opaque.node_tree.nodes.get('Principled BSDF').inputs[0].default_value = (0.9654, 0.92234, 0.34234, 1) ## color 
material_opaque.node_tree.nodes["Principled BSDF"].inputs[7].default_value = 0.1 ## roughness
material_opaque.node_tree.nodes["Principled BSDF"].inputs[4].default_value = 1 ## metallic
#material_opaque.node_tree.nodes["Principled BSDF"].inputs[17].default_value = (0.9654, 0.92234, 0.34234, 1)  ## emission
#material_opaque.node_tree.nodes["Principled BSDF"].inputs[18].default_value = 5.0  ## emission strength

## Create volume material.
material_volume = bpy.data.materials.new(name="Volume")
material_volume.use_nodes = True
material_volume.node_tree.nodes.remove(material_volume.node_tree.nodes.get('Principled BSDF'))
material_output = material_volume.node_tree.nodes.get('Material Output')
volume_node = material_volume.node_tree.nodes.new('ShaderNodeVolumePrincipled')
volume_node.location = (-600, 0)
volume_node.inputs[0].default_value = (0.92, 0.45, 0.33, 1) # color
volume_node.inputs[2].default_value = 0.2 # density
material_volume.node_tree.links.new(volume_node.outputs[0], material_output.inputs[1])

'''
#########################################################
#   Modifiers
# TODO: use modifiers object.modifiers (this seems OK)
# Convert meshes to one big mesh and use volume
# TODO: copy mesh (not object!)
#########################################################
# Create flow_collection collection
flow_collection = bpy.data.collections.new(name="flow_collection")
# Add flow_collection to scene
context.scene.collection.children.link(flow_collection)

verts = [(-1, 1, 0), (-1, -1, 0), (1, 0, 0), (0, 0, 1)]
edges =  []#[(0,1), (0,2), (1,2), (0,3), (1,3), (2,3)] # not needed if faces are defined!
faces = [(1,3,0), (2,3,1), (0,3,2), (0,2,1)]
# create mesh object
orig_particle_mesh = bpy.data.meshes.new('orig_particle_mesh')
orig_particle_mesh.from_pydata(verts, edges, faces)
orig_particle_mesh.copy()
validation_result = orig_particle_mesh.validate(verbose=True, clean_customdata=True) 
print("Is mesh data valid?", validation_result) # TODO: fix invalid mesh!
# create object from mesh and add to flow_collection
orig_particle_object = bpy.data.objects.new('orig_particle_object', orig_particle_mesh)
orig_particle_object.active_material = material_volume
flow_collection.objects.link(orig_particle_object)
'''

'''
#########################################################
#   NP vectorization.
#########################################################
noise_vector_vectorized = np.vectorize(noise.noise_vector)
print(noise.noise_vector((1.2, 3.2, 1.1), noise_basis='VORONOI_F1'))
print(noise_vector_vectorized([(1.2, 3.2, 1.1)], noise_basis='VORONOI_F1'))
'''

#########################################################
#   Helpers.
#########################################################

#########################################################
#   Creating collections.
# https://devtalk.blender.org/t/where-to-find-collection-index-for-moving-an-object/3289/3
# NB: "Do not use operators for moving objects between collections. In general operators are not reliable to use in scripts, they are meant as end user tools."
# Everything should be done with .data and .context: https://blenderartists.org/t/bpy-data-vs-bpy-ops/493671
#########################################################
def create_collection(collection_name):
    # Create collection.
    new_collection = bpy.data.collections.new(name=collection_name)
    # Add flow_collection to scene
    context.scene.collection.children.link(new_collection)
    return new_collection

#########################################################
#   Creating mesh objects
# NB: everything should be done using bpy.data!
# https://b3d.interplanety.org/en/how-to-create-mesh-through-the-blender-python-api/
# https://blender.stackexchange.com/questions/95408/how-do-i-create-a-new-object-using-python-in-blender-2-80
# TWO options: use bmesh or mesh.from_pydata: https://blender.stackexchange.com/questions/61879/create-mesh-then-add-vertices-to-it-in-python
#########################################################
def create_mesh_object(name, material, collection):
    # Define vertices.
    #            0            1           2           3
    verts = [(-1, 1, 0), (-1, -1, 0), (1, 0, 0), (0, 0, 1)]
    edges =  []#[(0,1), (0,2), (1,2), (0,3), (1,3), (2,3)] # not needed if faces are defined!
    faces = [(1,3,0), (2,3,1), (0,3,2), (0,2,1)]
    
    # create mesh object
    new_mesh = bpy.data.meshes.new(name)
    new_mesh.from_pydata(verts, edges, faces)
    validation_result = new_mesh.validate(verbose=True, clean_customdata=True) 
    print("Is mesh data valid?", validation_result) # TODO: fix invalid mesh!
    
    # Create object from mesh.
    new_object = data.objects.new(name, new_mesh)
    
    # Add material.
    if(material != 0):
        new_object.active_material = material
    
    # Add to collection.
    collection.objects.link(new_object)
    
    return new_object

#########################################################
# Create duplicate (instanced, linked) mesh object:  
# https://blender.stackexchange.com/questions/45099/duplicating-a-mesh-object
# Again, ops is not good way to do this! Use object.copy()
#########################################################
def create_object_duplicates(object, locations, collection):
    for loc in locations:
        instance_object = object.copy()
        #instance_object.data = object.data.copy() # without this we do instancing!
        instance_object.animation_data_clear()
        instance_object.location = loc
        collection.objects.link(instance_object)
        
def create_curve():
    pass
    

#########################################################
#   Animation.
#########################################################

def gen_anim():
    
    # Create collections.
    flow_collection = create_collection("flow_collection")
    flow_collection2 = create_collection("flow_collection2")
    static_collection = create_collection("static_collection")
    static_collection2 = create_collection("static_collection2")
    
    # Create objects.
    orig_particle_object = create_mesh_object("orig_particle_object", 0, flow_collection)
    
    # Create duplicate particles.
    create_object_duplicates(orig_particle_object, source_particles, flow_collection)
    
    create_object_duplicates(orig_particle_object, source_particles, flow_collection2)
        
    #########################################################
    # Curves
    #########################################################
    # Create data for curve object.
    # Types: ['CURVE', 'SURFACE', 'FONT']
    curve_data = data.curves.new(name="my_curve", type='CURVE')
    # Dimensions: ['2D', '3D']
    curve_data.dimensions = '3D'
    # Cache shortcut to splines.
    splines = curve_data.splines
    
    #Another Curve.
    # Types: ['CURVE', 'SURFACE', 'FONT']
    curve_data2 = data.curves.new(name="my_curve2", type='CURVE')
    # Dimensions: ['2D', '3D']
    curve_data2.dimensions = '3D'
    # Cache shortcut to splines.
    splines2 = curve_data2.splines
    

    #########################################################
    # Create animation
    # Create static structure
    #########################################################
    frame_no = 0
    n_keyframes = 30
    for i_keyframe in range(0, n_keyframes):
        
        context.scene.frame_set(frame_no)
        
        '''
        # Before each keyframe, duplicate particles in static collection
        for obj in flow_collection.all_objects:
            static_particle_object = obj.copy()
            #instance_particle_object.data = orig_particle_object.data.copy() # without this we do instancing!
            static_particle_object.animation_data_clear()
            static_collection.objects.link(static_particle_object)
        '''
        
        if i_keyframe == 0:
            for obj in flow_collection.all_objects:
                # create keyframe from particles in flow_collection for flow animation.
                obj.keyframe_insert(data_path="location", index=-1)
            for obj in flow_collection2.all_objects:
                # create keyframe from particles in flow_collection for flow animation.
                obj.keyframe_insert(data_path="location", index=-1) 
            frame_no += 20
        
        else:
            for obj in flow_collection.all_objects:
                loc = obj.location
                offset_vec = noise.noise_vector(loc*0.06, noise_basis='VORONOI_F1')
                offset_fact = 30.0
                curve_starting_point = Vector((obj.location))                
                obj.location = loc + offset_fact * offset_vec
                curve_ending_point = Vector((obj.location))
                # TODO: add bevel!
                add_spline_curve(splines, 16, 5, 'FREE', curve_starting_point, curve_ending_point)
                obj.keyframe_insert(data_path="location", index=-1)
            for obj in flow_collection2.all_objects:
                loc = obj.location
                offset_vec = noise.noise_vector(loc*0.5, noise_basis='VORONOI_F1')
                offset_fact = 30.0
                curve_starting_point = Vector((obj.location))                
                obj.location = loc + offset_fact * offset_vec
                curve_ending_point = Vector((obj.location))
                # TODO: add bevel!
                add_spline_curve(splines2, 16, 5, 'FREE', curve_starting_point, curve_ending_point)
                obj.keyframe_insert(data_path="location", index=-1)
            frame_no += 20
            
    # Create object from data
    curve_object = data.objects.new(name="my_curve", object_data=curve_data)
    # Link object to the scene.
    static_collection.objects.link(curve_object)
    
    # Create object from data
    curve_object2 = data.objects.new(name="my_curve2", object_data=curve_data2)
    # Link object to the scene.
    static_collection2.objects.link(curve_object2)
            
gen_anim()

#########################################################
#   Generative Animation 2: Using vertices
#########################################################

## Idea is to use bmesh to create arbitrary vertices and transform them
## and to create an object with those vertices.

# TODOs:
# 1. Add shapekeys: https://behreajj.medium.com/shaping-models-with-bmesh-in-blender-2-9-2f4fcc889bf0
# 2. Convex hull https://behreajj.medium.com/shaping-models-with-bmesh-in-blender-2-9-2f4fcc889bf0

def genAnim2():
    ## Create ini positions - vertices.
    bm = bmesh.new()
    for p in source_particles:
        bm.verts.new(p)

    ## Displace vertices.
    ref_frame = Matrix.Identity(4)
    bm.verts.ensure_lookup_table()
    for i in range(0, len(bm.verts)):
        verts = [bm.verts[i]]
        loc = bm.verts[i].co
        offset_vec = noise.noise_vector(loc*1.0, noise_basis='BLENDER')
        tr_vec = Vector(offset_vec * 5.9)
        bmesh.ops.translate(
            bm,
            vec=tr_vec,
            space=ref_frame,
            verts=verts)
        
    bmesh.ops.convex_hull(bm, input=bm.verts)
    mesh_data = data.meshes.new("Example")
    bm.to_mesh(mesh_data)
    bm.free()
    mesh_obj = data.objects.new(mesh_data.name, mesh_data)
    context.collection.objects.link(mesh_obj)
    mesh_obj.active_material = material_opaque

#genAnim2()

#########################################################
#   OLD TESTS.
#########################################################
'''
bm = bmesh.new()
bmesh.ops.create_cube(bm, size=0.5, matrix=Matrix.Identity(4), calc_uvs=True)
ref_frame = Matrix.Identity(4)
mesh_data = data.meshes.new("Cube")
bm.verts.ensure_lookup_table()
verts = [bm.verts[-1]]
tr_vec = Vector((0.5, 0.0, 0.0))
bmesh.ops.translate(
    bm,
    vec=tr_vec,
    space=ref_frame,
    verts=verts)
bm.to_mesh(mesh_data)
bm.free()
mesh_obj = data.objects.new(mesh_data.name, mesh_data)
context.collection.objects.link(mesh_obj)
'''

'''
bpy.ops.mesh.primitive_uv_sphere_add()
obj = context.active_object
for i in range(0, len(obj.data.vertices)):
    loc = obj.data.vertices[i].co
    offset_vec = noise.noise_vector(loc*1, noise_basis='BLENDER')
    obj.data.vertices[i].co = loc + 5 * offset_vec
'''

#####################
# Displace once and Render at fixed positions.
########################
'''
for sp in source_particles2:
    # Find cell for every point
    #col = noise.cell_vector(sp*0.2)
    offset_vec = noise.noise_vector(sp*0.15)
    col = (offset_vec[0], offset_vec[0], offset_vec[0])
    #gray = noise.noise(sp*0.3)
    #col = (gray, gray, gray)
    # Create material with cell color.
    material_opaque = bpy.data.materials.new(name="Opaque")
    material_opaque.use_nodes = True
    material_opaque.node_tree.nodes.get('Principled BSDF').inputs[0].default_value = (col[0], col[1], col[2], 1)
    # Add sphere on point position.
    bpy.ops.mesh.primitive_uv_sphere_add(location=sp+offset_vec*5)
    # Set material.
    bpy.context.object.active_material = material_opaque
'''

'''
## Create ini positions - spheres.
for p in source_particles:
    bpy.ops.mesh.primitive_uv_sphere_add(location=p)
    # Enable smooth shading
    bpy.ops.object.shade_smooth()
    # Add material.
    bpy.context.object.active_material = material_opaque


## Animate
frame_no = 0
n_frames = 30
for i_frame in range(0, n_frames):
    bpy.context.scene.frame_set(frame_no)
    if i_frame == 0:
        print("TU", i_frame)
        ## Keyframe ini positions.
        for obj in bpy.data.objects:
            obj.keyframe_insert(data_path="location", index=-1)
        frame_no += 20
    else:
        print("TU2", i_frame)
        ## Keyframe other positions.
        for obj in bpy.data.objects:
            loc = obj.location
            offset_vec = noise.noise_vector(loc*0.5, noise_basis='CELLNOISE')
            obj.location = loc + 15 * offset_vec
            obj.keyframe_insert(data_path="location", index=-1)
        frame_no += 20
'''

'''
#########################################################
#   Generative Animation 1: Using instancing
#########################################################

## Idea is to duplicate one object with link option. Therefore, only one
## geometry exists and it is transformed to different positions during rendering.
## This should be equivalent to particle system.

## TODOs:
# 1. experiment with noise
# 2. experiment with scales and magnitudes
# 3. add curve points where particles go
# 4. add colors, materials, light!
# 5. Constant animation curve
# 5. Camera movement and camera properties.

def genAnim1():
    # Create collection for particles. https://devtalk.blender.org/t/how-to-create-collections/11144/2  AND  https://blender.stackexchange.com/questions/132112/whats-the-blender-2-8-command-for-adding-an-object-to-a-collection-using-python
    master_collection = bpy.context.scene.collection
    flow_collection = bpy.data.collections.new("flow_collection")
    bpy.context.scene.collection.children.link(flow_collection)
    flow_collection = bpy.data.collections[ "flow_collection" ]
    structure_collection = bpy.data.collections.new("structure_collection")
    bpy.context.scene.collection.children.link(structure_collection)
    structure_collection = bpy.data.collections[ "structure_collection" ]
    
    # Create original object and add it to the flow_collection.
    bpy.ops.mesh.primitive_uv_sphere_add(segments=3, ring_count=3, radius=1, enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
    # Created sphere is the active object.
    orig = bpy.context.active_object
    # add it to our specific collection https://blender.stackexchange.com/questions/132112/whats-the-blender-2-8-command-for-adding-an-object-to-a-collection-using-python
    flow_collection.objects.link(orig)
    master_collection.objects.unlink(orig)
    
    # Create duplicates in flow_collection.
    #orig = bpy.context.scene.objects.get("Sphere") # tries to fetch object from wrong collection! TODO: figure out how to fetch from correct collection.
    orig.select_set(True)
    bpy.context.object.active_material = material_opaque
    for sp in source_particles:
        orig.select_set(True) # select original
        bpy.ops.object.duplicate(linked=True) # duplicate original (instancing)
        bpy.ops.transform.translate(value=(sp)) # transform duplicate of original
        bpy.ops.object.select_all(action='DESELECT') # deselect all
    
    
    # Animate duplicates.
    frame_no = 0
    n_keyframes = 30
    for i_keyframe in range(0, n_keyframes):
        bpy.context.scene.frame_set(frame_no)
        if i_keyframe == 0:
            # Keyframe ini positions.
            # Iterates only through one collection of objects https://blender.stackexchange.com/questions/144928/how-to-list-all-collections-and-their-objects
            for obj in flow_collection.all_objects:
                # create keyframe from particles in flow_collection for flow animation.
                obj.keyframe_insert(data_path="location", index=-1) 
                
                # Add static particles in structure_collection.
                bpy.ops.object.select_all(action='DESELECT') # deselect all
                obj.select_set(True)
                bpy.data.objects[obj.name].select_set(True)
                # TODO: not working because objects are created and iterated!
                # SOLUTION: na pocetku, prije svakog framea duplicirati kolekciju
                #bpy.ops.object.duplicate(linked=True) # duplicate (instancing)
                #instance = bpy.context.active_object
                
                bpy.ops.object.select_all(action='DESELECT') # deselect all
            frame_no += 20
        else:
            # Keyframe other positions.
            # Iterates only through one collection of objects https://blender.stackexchange.com/questions/134776/how-to-get-the-collection-an-object-belongs-to
            for obj in flow_collection.all_objects:
                loc = obj.location
                #offset_vec = noise.noise_vector(loc*0.5, noise_basis='CELLNOISE')
                offset_vec = noise.noise_vector(loc*0.005, noise_basis='VORONOI_F1')
                #offset_vec = noise.random_unit_vector(size=3)
                #offset_vec = noise.random_vector(size=3)
                offset_fact = 10.0 #noise.multi_fractal(loc*0.5, 0.3, 0.3, 3, noise_basis='PERLIN_ORIGINAL') * 30
                obj.location = loc + offset_fact * offset_vec
                obj.keyframe_insert(data_path="location", index=-1)
            frame_no += 20
'''