use anyhow::*;
use rayon::prelude::*;
use std::ops::Range;
use std::path::Path;
use wgpu::util::DeviceExt;

use crate::texture;

pub trait Vertex {
    fn desc<'a>() -> wgpu::VertexBufferDescriptor<'a>;
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ModelVertex {
    position: [f32; 3],
    tex_coords: [f32; 2],
    normal: [f32; 3],
    tangent: [f32; 3],
    bitangent: [f32; 3],
}

impl Vertex for ModelVertex {
    fn desc<'a>() -> wgpu::VertexBufferDescriptor<'a> {
        use std::mem;
        wgpu::VertexBufferDescriptor {
            stride: mem::size_of::<ModelVertex>() as wgpu::BufferAddress, step_mode: wgpu::InputStepMode::Vertex, attributes: &[
                wgpu::VertexAttributeDescriptor {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float3,
                },
                wgpu::VertexAttributeDescriptor {
                    offset: mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float2,
                },
                wgpu::VertexAttributeDescriptor {
                    offset: mem::size_of::<[f32; 5]>() as wgpu::BufferAddress,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float3,
                },
            ],
        }
    }
}

pub struct Material {
    pub name: String,
    // TODO: Change this to Option<Texture> or default white texture
    pub diffuse_texture: texture::Texture,
    pub normal_texture: texture::Texture,
    pub bind_group: wgpu::BindGroup,
}

impl Material {
    pub fn new(device: &wgpu::Device, name: &str, diffuse_texture: texture::Texture, normal_texture: texture::Texture, layout: &wgpu::BindGroupLayout,) -> Self {

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(name),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&diffuse_texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&diffuse_texture.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&normal_texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&normal_texture.sampler),
                },
            ],
        });

        return Material {
            name: String::from(name),
            diffuse_texture,
            normal_texture,
            bind_group
        };
    }
}

pub struct Mesh {
    pub name: String,
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub num_elements: u32,
    pub material: usize,
}

pub struct Model {
    pub meshes: Vec<Mesh>,
    pub materials: Vec<Material>,
}

impl Model {
    pub fn load<P: AsRef<Path>>(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layout: &wgpu::BindGroupLayout,
        path: P,
    ) -> Result<Self> {
        let (obj_models, obj_materials) = tobj::load_obj(path.as_ref(), true)?;

        // We're assuming that the texture files are stored with the obj file
        let containing_folder = path.as_ref().parent().context("Directory has no parent")?;

        let materials = obj_materials.par_iter().map(|mat| {
            let mut textures = [
                (containing_folder.join(&mat.diffuse_texture), false),
                (containing_folder.join(&mat.normal_texture), true),
            ].par_iter().map(|(texture_path, is_normal_map)| {
                texture::Texture::load(device, queue, texture_path, *is_normal_map)
            }).collect::<Result<Vec<_>>>()?;

            let normal_texture = textures.pop().unwrap();
            let diffuse_texture = textures.pop().unwrap();

            Ok(Material::new(device, &mat.name, diffuse_texture, normal_texture, layout))
        }).collect::<Result<Vec<Material>>>()?;

        let meshes = obj_models
            .par_iter()
            .map(|m| {
                let mut vertices = (0..m.mesh.positions.len() / 3)
                    .into_par_iter()
                    .map(|i| {
                        ModelVertex {
                            position: [
                                m.mesh.positions[i * 3],
                                m.mesh.positions[i * 3 + 1],
                                m.mesh.positions[i * 3 + 2],
                            ]
                            .into(),
                            tex_coords: [m.mesh.texcoords[i * 2], m.mesh.texcoords[i * 2 + 1]]
                                .into(),
                            normal: [
                                m.mesh.normals[i * 3],
                                m.mesh.normals[i * 3 + 1],
                                m.mesh.normals[i * 3 + 2],
                            ]
                            .into(),
                            // We'll calculate these later
                            tangent: [0.0; 3].into(),
                            bitangent: [0.0; 3].into(),
                        }
                    })
                    .collect::<Vec<_>>();

                let indices = &m.mesh.indices;

                // Calculate tangents and bitangets. We're going to
                // use the triangles, so we need to loop through the
                // indices in chunks of 3
                for c in indices.chunks(3) {
                    let v0 = vertices[c[0] as usize];
                    let v1 = vertices[c[1] as usize];
                    let v2 = vertices[c[2] as usize];

                    let pos0: cgmath::Vector3<_> = v0.position.into();
                    let pos1: cgmath::Vector3<_> = v1.position.into();
                    let pos2: cgmath::Vector3<_> = v2.position.into();

                    let uv0: cgmath::Vector2<_> = v0.tex_coords.into();
                    let uv1: cgmath::Vector2<_> = v1.tex_coords.into();
                    let uv2: cgmath::Vector2<_> = v2.tex_coords.into();

                    // Calculate the edges of the triangle
                    let delta_pos1 = pos1 - pos0;
                    let delta_pos2 = pos2 - pos0;

                    // This will give us a direction to calculate the
                    // tangent and bitangent
                    let delta_uv1 = uv1 - uv0;
                    let delta_uv2 = uv2 - uv0;

                    // Solving the following system of equations will
                    // give us the tangent and bitangent.
                    //     delta_pos1 = delta_uv1.x * T + delta_u.y * B
                    //     delta_pos2 = delta_uv2.x * T + delta_uv2.y * B
                    // Luckily, the place I found this equation provided
                    // the solution!
                    let r = 1.0 / (delta_uv1.x * delta_uv2.y - delta_uv1.y * delta_uv2.x);
                    let tangent = (delta_pos1 * delta_uv2.y - delta_pos2 * delta_uv1.y) * r;
                    let bitangent = (delta_pos2 * delta_uv1.x - delta_pos1 * delta_uv2.x) * r;

                    // We'll use the same tangent/bitangent for each vertex in the triangle
                    vertices[c[0] as usize].tangent = tangent.into();
                    vertices[c[1] as usize].tangent = tangent.into();
                    vertices[c[2] as usize].tangent = tangent.into();

                    vertices[c[0] as usize].bitangent = bitangent.into();
                    vertices[c[1] as usize].bitangent = bitangent.into();
                    vertices[c[2] as usize].bitangent = bitangent.into();
                }

                let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("{:?} Vertex Buffer", m.name)),
                    contents: bytemuck::cast_slice(&vertices),
                    usage: wgpu::BufferUsage::VERTEX,
                });
                let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("{:?} Index Buffer", m.name)),
                    contents: bytemuck::cast_slice(&m.mesh.indices),
                    usage: wgpu::BufferUsage::INDEX,
                });

                Ok(Mesh {
                    name: m.name.clone(),
                    vertex_buffer,
                    index_buffer,
                    num_elements: m.mesh.indices.len() as u32,
                    material: m.mesh.material_id.unwrap_or(0),
                })
            }).collect::<Result<Vec<_>>>()?;

        Ok(Self { meshes, materials })
    }
}

pub trait DrawModel<'a>
{
    fn draw_mesh(&mut self, mesh: &'a Mesh, material: &'a Material, uniforms: &'a wgpu::BindGroup, light: &'a wgpu::BindGroup);
    fn draw_mesh_instanced(&mut self, mesh: &'a Mesh, material: &'a Material, instances: Range<u32>, uniforms: &'a wgpu::BindGroup, light: &'a wgpu::BindGroup);
    fn draw_model(&mut self, model: &'a Model, uniforms: &'a wgpu::BindGroup, light: &'a wgpu::BindGroup);
    fn draw_model_instanced(&mut self, model: &'a Model, instances: Range<u32>, uniforms: &'a wgpu::BindGroup, light: &'a wgpu::BindGroup);
    fn draw_model_instanced_with_material(&mut self, model: &'a Model, material: &'a Material, instances: Range<u32>, uniforms: &'a wgpu::BindGroup, light: &'a wgpu::BindGroup);
}

impl<'a> DrawModel<'a> for wgpu::RenderPass<'a> {
    fn draw_mesh(&mut self, mesh: &'a Mesh, material: &'a Material, uniforms: &'a wgpu::BindGroup, light: &'a wgpu::BindGroup) {
        self.draw_mesh_instanced(mesh, material, 0..1, uniforms, light);
    }

    fn draw_mesh_instanced(&mut self, mesh: &'a Mesh, material: &'a Material, instances: Range<u32>, uniforms: &'a wgpu::BindGroup, light: &'a wgpu::BindGroup) {
        self.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
        self.set_index_buffer(mesh.index_buffer.slice(..));
        self.set_bind_group(0, &material.bind_group, &[]);
        self.set_bind_group(1, &uniforms, &[]);
        self.set_bind_group(2, &light, &[]);
        self.draw_indexed(0..mesh.num_elements, 0, instances);
    }

    fn draw_model(&mut self, model: &'a Model, uniforms: &'a wgpu::BindGroup, light: &'a wgpu::BindGroup) {
        self.draw_model_instanced(model, 0..1, uniforms, light);
    }

    fn draw_model_instanced(&mut self, model: &'a Model, instances: Range<u32>, uniforms: &'a wgpu::BindGroup, light: &'a wgpu::BindGroup) {
        for mesh in &model.meshes {
            let material = &model.materials[mesh.material];
            self.draw_mesh_instanced(mesh, material, instances.clone(), uniforms, light);
        }
    }

    fn draw_model_instanced_with_material(&mut self, model: &'a Model, material: &'a Material, instances: Range<u32>, uniforms: &'a wgpu::BindGroup, light: &'a wgpu::BindGroup) {
         for mesh in &model.meshes {
            self.draw_mesh_instanced(mesh, material, instances.clone(), uniforms, light);
        }
    }
}

pub trait DrawLight<'a> {
    fn draw_light_mesh(&mut self, mesh: &'a Mesh, uniforms: &'a wgpu::BindGroup, light: &'a wgpu::BindGroup,);
    fn draw_light_mesh_instanced(&mut self, mesh: &'a Mesh, instances: Range<u32>, uniforms: &'a wgpu::BindGroup, light: &'a wgpu::BindGroup,);
    fn draw_light_model(&mut self, model: &'a Model, uniforms: &'a wgpu::BindGroup, light: &'a wgpu::BindGroup,);
    fn draw_light_model_instanced(&mut self, model: &'a Model, instances: Range<u32>, uniforms: &'a wgpu::BindGroup, light: &'a wgpu::BindGroup,);
}

impl<'a> DrawLight<'a> for wgpu::RenderPass<'a> {
    fn draw_light_mesh(&mut self, mesh: &'a Mesh, uniforms: &'a wgpu::BindGroup, light: &'a wgpu::BindGroup,) {
        self.draw_light_mesh_instanced(mesh, 0..1, uniforms, light);
    }

    fn draw_light_mesh_instanced(&mut self, mesh: &'a Mesh, instances: Range<u32>, uniforms: &'a wgpu::BindGroup, light: &'a wgpu::BindGroup,) {
        self.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
        self.set_index_buffer(mesh.index_buffer.slice(..));
        self.set_bind_group(0, uniforms, &[]);
        self.set_bind_group(1, light, &[]);
        self.draw_indexed(0..mesh.num_elements, 0, instances);
    }

    fn draw_light_model(&mut self, model: &'a Model, uniforms: &'a wgpu::BindGroup, light: &'a wgpu::BindGroup,) {
        self.draw_light_model_instanced(model, 0..1, uniforms, light);
    }

    fn draw_light_model_instanced(&mut self, model: &'a Model, instances: Range<u32>, uniforms: &'a wgpu::BindGroup, light: &'a wgpu::BindGroup,) {
        for mesh in &model.meshes {
            self.draw_light_mesh_instanced(mesh, instances.clone(), uniforms, light);
        }
    }
}
