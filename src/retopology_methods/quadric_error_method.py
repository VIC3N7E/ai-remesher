import trimesh
import numpy as np
from typing import Tuple, Optional
from scipy.spatial import Delaunay
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

def compute_quadric_error(mesh):
    """
    Compute the quadric error metric for each vertex.
    Args:
        mesh: trimesh.Trimesh object
    Returns:
        quadrics: list of 4x4 matrices representing the quadric error for each vertex
    """
    vertices = mesh.vertices
    faces = mesh.faces
    
    # Initialize quadrics for each vertex
    quadrics = [np.zeros((4, 4)) for _ in range(len(vertices))]
    
    # For each face, compute its plane equation and add to vertex quadrics
    for face in faces:
        v1, v2, v3 = vertices[face]
        # Compute plane normal
        normal = np.cross(v2 - v1, v3 - v1)
        normal = normal / np.linalg.norm(normal)
        # Compute plane equation: ax + by + cz + d = 0
        d = -np.dot(normal, v1)
        # Create quadric matrix
        p = np.array([normal[0], normal[1], normal[2], d])
        K = np.outer(p, p)
        # Add to each vertex's quadric
        for v_idx in face:
            quadrics[v_idx] += K
    
    return quadrics

def find_best_vertex(v1, v2, quadric1, quadric2):
    """
    Find the optimal vertex position when collapsing edge (v1, v2).
    Args:
        v1, v2: vertex positions
        quadric1, quadric2: quadric matrices for v1 and v2
    Returns:
        optimal_vertex: optimal position for the new vertex
    """
    # Combine quadrics
    Q = quadric1 + quadric2
    
    # Try v1, v2, and midpoint
    candidates = [v1, v2, (v1 + v2) / 2]
    errors = []
    
    for v in candidates:
        v_homogeneous = np.append(v, 1)
        error = v_homogeneous @ Q @ v_homogeneous
        errors.append(error)
    
    # Return the vertex with minimum error
    return candidates[np.argmin(errors)]

def quadric_error_method(mesh: trimesh.Trimesh, 
                        target_faces: Optional[int] = None,
                        feature_edges: Optional[np.ndarray] = None) -> trimesh.Trimesh:
    """
    Apply Quadric Error Metric (QEM) mesh simplification.
    Args:
        mesh: trimesh.Trimesh object
        target_faces: target number of faces (if None, will simplify until no valid edges remain)
        feature_edges: deprecated parameter, not used
    Returns:
        simplified_mesh: trimesh.Trimesh object
    """
    print(f"Starting QEM simplification...")
    print(f"Initial mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    # Create a copy of the mesh to work on
    simplified = mesh.copy()
    
    print("Computing initial quadrics...")
    # Compute initial quadrics
    quadrics = compute_quadric_error(simplified)
    print("Quadrics computed")
    
    # Get edges
    print("Preparing edges...")
    edges = np.vstack([simplified.edges, simplified.edges[:, [1, 0]]])
    print(f"Total edges to process: {len(edges)}")
    
    # Initialize vertex mapping
    vertex_map = {i: i for i in range(len(simplified.vertices))}
    
    # Main simplification loop
    print("\nStarting edge collapse iterations...")
    iteration = 0
    last_print = 0
    while target_faces is None or len(simplified.faces) > target_faces:
        # Print progress every 100 iterations or 5% of target faces
        if iteration % 100 == 0 or (target_faces is not None and 
            len(simplified.faces) - target_faces < (len(mesh.faces) - target_faces) * 0.05):
            print(f"Iteration {iteration}: {len(simplified.faces)} faces remaining")
            last_print = iteration
        
        # Find the best edge to collapse
        best_error = float('inf')
        best_edge = None
        best_vertex = None
        
        for edge in edges:
            v1, v2 = edge
            if v1 not in vertex_map or v2 not in vertex_map:
                continue
                
            # Get actual vertex positions
            pos1 = simplified.vertices[vertex_map[v1]]
            pos2 = simplified.vertices[vertex_map[v2]]
            
            # Find optimal vertex position
            new_pos = find_best_vertex(pos1, pos2, quadrics[v1], quadrics[v2])
            
            # Compute error
            v_homogeneous = np.append(new_pos, 1)
            error = v_homogeneous @ (quadrics[v1] + quadrics[v2]) @ v_homogeneous
            
            if error < best_error:
                best_error = error
                best_edge = edge
                best_vertex = new_pos
        
        if best_edge is None:
            print("\nNo more valid edges to collapse")
            break
            
        # Collapse the edge
        v1, v2 = best_edge
        simplified.vertices[vertex_map[v1]] = best_vertex
        vertex_map[v2] = vertex_map[v1]
        
        # Update quadrics
        quadrics[v1] += quadrics[v2]
        
        iteration += 1
    
    print("\nCreating simplified mesh...")
    # Create new mesh with simplified geometry
    new_vertices = []
    new_faces = []
    vertex_idx_map = {}
    
    # Create new vertex list
    for i, v in enumerate(simplified.vertices):
        if i in vertex_map and vertex_map[i] == i:
            vertex_idx_map[i] = len(new_vertices)
            new_vertices.append(v)
    
    # Create new face list
    for face in simplified.faces:
        new_face = [vertex_idx_map[vertex_map[v]] for v in face]
        if len(set(new_face)) == 3:  # Only keep valid faces
            new_faces.append(new_face)
    
    result = trimesh.Trimesh(vertices=new_vertices, faces=new_faces, process=False)
    print(f"Simplification complete: {len(result.vertices)} vertices, {len(result.faces)} faces")
    return result 