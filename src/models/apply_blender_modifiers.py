import bpy
import os
import argparse
import sys
from pathlib import Path

def clear_scene():
    """Clear the current Blender scene."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

def read_off_file(filepath):
    """Read an OFF file and create a Blender mesh."""
    with open(filepath, 'r') as f:
        # Skip the first line (OFF header)
        header = f.readline().strip()
        if not header.startswith('OFF'):
            raise ValueError("Not a valid OFF file")
        
        # Read number of vertices and faces
        line = f.readline().strip()
        while line.startswith('#'):  # Skip comments
            line = f.readline().strip()
        n_vertices, n_faces, _ = map(int, line.split())
        
        # Read vertices
        vertices = []
        for _ in range(n_vertices):
            line = f.readline().strip()
            while line.startswith('#'):  # Skip comments
                line = f.readline().strip()
            x, y, z = map(float, line.split()[:3])
            vertices.append((x, y, z))
        
        # Read faces
        faces = []
        for _ in range(n_faces):
            line = f.readline().strip()
            while line.startswith('#'):  # Skip comments
                line = f.readline().strip()
            values = list(map(int, line.split()))
            n_verts = values[0]
            if n_verts < 3:
                continue  # Skip invalid faces
            faces.append(values[1:n_verts+1])
        
        # Create mesh
        mesh = bpy.data.meshes.new(name="imported_mesh")
        mesh.from_pydata(vertices, [], faces)
        mesh.update()
        
        # Create object
        obj = bpy.data.objects.new("imported_object", mesh)
        bpy.context.scene.collection.objects.link(obj)
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        
        return obj

def convert_to_obj(input_path: str, output_path: str):
    """
    Convert a mesh file to OBJ format.
    Args:
        input_path: Path to input mesh file (.off, .fbx, or .obj)
        output_path: Path to save the OBJ file
    """
    # Clear the scene
    clear_scene()
    
    # Import based on file extension
    file_ext = os.path.splitext(input_path)[1].lower()
    if file_ext == '.off':
        obj = read_off_file(input_path)
    elif file_ext == '.fbx':
        bpy.ops.import_scene.fbx(filepath=input_path)
        obj = bpy.context.selected_objects[0]
    elif file_ext == '.obj':
        bpy.ops.wm.obj_import(filepath=input_path)
        obj = bpy.context.selected_objects[0]
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")
    
    # Export as OBJ
    bpy.ops.wm.obj_export(filepath=output_path)
    
    return output_path

def apply_modifiers_to_mesh(input_path: str, output_path: str, 
                          subdivision_levels: int = 2,
                          noise_scale: float = 0.1,
                          noise_strength: float = 1.0):
    """
    Apply subdivision and noise modifiers to a mesh.
    Args:
        input_path: Path to input OBJ file
        output_path: Path to save modified OBJ file
        subdivision_levels: Number of subdivision levels
        noise_scale: Scale of the noise modifier
        noise_strength: Strength of the noise modifier
    """
    # Clear the scene
    clear_scene()
    
    # Import the OBJ file
    bpy.ops.wm.obj_import(filepath=input_path)
    
    # Get the imported object
    obj = bpy.context.selected_objects[0]
    
    # Add subdivision modifier
    subdiv = obj.modifiers.new(name="Subdivision", type='SUBSURF')
    subdiv.levels = subdivision_levels
    subdiv.render_levels = subdivision_levels
    
    # Add noise modifier
    noise = obj.modifiers.new(name="Noise", type='DISPLACE')
    noise.strength = noise_strength
    
    # Create a new texture for the noise
    tex = bpy.data.textures.new('Noise', type='CLOUDS')
    tex.noise_scale = noise_scale   
    noise.texture = tex
    
    # Apply modifiers and check file size
    bpy.context.view_layer.objects.active = obj
    current_subdiv_levels = subdivision_levels
    
    while current_subdiv_levels >= 0:
        # Clear all modifiers
        obj.modifiers.clear()
        
        subdiv = obj.modifiers.new(name="Subdivision", type='SUBSURF')
        subdiv.levels = current_subdiv_levels
        subdiv.render_levels = current_subdiv_levels
        
        # Re-add noise modifier
        noise = obj.modifiers.new(name="Noise", type='DISPLACE')
        noise.strength = noise_strength
        noise.texture = tex
        
        # Apply modifiers
        for modifier in obj.modifiers:
            bpy.ops.object.modifier_apply(modifier=modifier.name)
        
        # Export and check size
        bpy.ops.wm.obj_export(filepath=output_path)
        
        # If file size is acceptable or we've reached minimum subdivision, break
        if os.path.getsize(output_path) <= 2 * 1024 * 1024 or current_subdiv_levels == 0:
            print(f"Final subdivision level: {current_subdiv_levels}")
            break
            
        # If file is still too large, reduce subdivision level
        print(f"File too large ({os.path.getsize(output_path) / (1024*1024):.2f} MB), reducing subdivision to {current_subdiv_levels-1}")
        current_subdiv_levels -= 1

def process_directory(input_dir: str, output_dir: str, 
                     subdivision_levels: int = 2,
                     noise_scale: float = 0.1,
                     noise_strength: float = 1):
    """
    Process all mesh files in a directory and its subdirectories.
    Args:
        input_dir: Input directory containing mesh files
        output_dir: Output directory for modified meshes
        subdivision_levels: Number of subdivision levels
        noise_scale: Scale of the noise modifier
        noise_strength: Strength of the noise modifier
    """
    # Create output directories
    modified_dir = os.path.join(output_dir, "modified")
    original_dir = os.path.join(output_dir, "original")
    os.makedirs(modified_dir, exist_ok=True)
    os.makedirs(original_dir, exist_ok=True)
    
    # Create a temporary directory for converted OBJ files
    temp_dir = os.path.join(output_dir, "temp_converted")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Walk through the input directory
    for root, dirs, files in os.walk(input_dir):
        # Create corresponding subdirectories in output
        rel_path = os.path.relpath(root, input_dir)
        modified_subdir = os.path.join(modified_dir, rel_path)
        original_subdir = os.path.join(original_dir, rel_path)
        os.makedirs(modified_subdir, exist_ok=True)
        os.makedirs(original_subdir, exist_ok=True)
        i = 0;
        # Process each mesh file
        for file in files:
            i += 1
            if i > 100:
                break
            file_ext = os.path.splitext(file)[1].lower()
            if file_ext in ['.off', '.fbx', '.obj']:
                input_path = os.path.join(root, file)
                base_name = os.path.splitext(file)[0]
                # check if file is bigger than 2 mb
                if os.path.getsize(input_path) > 2 * 1024 * 1024:
                    print(f"Skipping {input_path} because it is too large")
                    continue
                
                # Define paths for original and modified meshes
                temp_obj_path = os.path.join(temp_dir, f"{base_name}.obj")
                original_path = os.path.join(original_subdir, f"{base_name}_original.obj")
                modified_path = os.path.join(modified_subdir, f"{base_name}_modified.obj")
                
                print(f"Processing: {input_path}")
                try:
                    # Convert to OBJ if needed and save original
                    if file_ext != '.obj':
                        print(f"Converting {file_ext} to OBJ...")
                        convert_to_obj(input_path, temp_obj_path)
                        input_path = temp_obj_path
                    
                    # Save the original mesh
                    bpy.ops.wm.obj_export(filepath=original_path)
                    print(f"Saved original to: {original_path}")
                    
                    # Apply modifiers and save modified mesh
                    apply_modifiers_to_mesh(
                        input_path, 
                        modified_path,
                        subdivision_levels,
                        noise_scale,
                        noise_strength
                    )
                    print(f"Saved modified to: {modified_path}")
                except Exception as e:
                    print(f"Error processing {input_path}: {str(e)}")
    
    # Clean up temporary directory
    try:
        import shutil
        shutil.rmtree(temp_dir)
    except Exception as e:
        print(f"Warning: Could not clean up temporary directory: {str(e)}")

def main():
    # Get the arguments after '--'
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(description='Apply Blender modifiers to meshes')
    parser.add_argument('input_dir', help='Input directory containing mesh files')
    parser.add_argument('output_dir', help='Output directory for modified meshes')
    parser.add_argument('--subdivision-levels', type=int, default=2, help='Number of subdivision levels')
    parser.add_argument('--noise-scale', type=float, default=0.1, help='Scale of the noise modifier')
    parser.add_argument('--noise-strength', type=float, default=0.1, help='Strength of the noise modifier')
    
    args = parser.parse_args(argv)
    
    print(f"Processing meshes from {args.input_dir} to {args.output_dir}")
    print(f"Using subdivision levels: {args.subdivision_levels}")
    print(f"Using noise scale: {args.noise_scale}")
    print(f"Using noise strength: {args.noise_strength}")
    
    process_directory(
        args.input_dir,
        args.output_dir,
        args.subdivision_levels,
        args.noise_scale,
        args.noise_strength
    )

if __name__ == '__main__':
    main() 