#![allow(unused_imports)]
mod texture;
mod model;
mod camera; // means -> mod camera { // contents of camera.rs }
mod light;

use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
    dpi::PhysicalSize,
};

use futures::executor::block_on;
use wgpu::util::DeviceExt;

use texture::Texture;
use model::{Vertex, DrawModel, DrawLight};
use camera::{Camera, CameraController};
use light::Light;

use cgmath::prelude::*;

//=============================================================================

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    // We can't use cgmath with bytemuck directly so we'll have
    // to convert the Matrix4 into a 4x4 f32 array
    // TODO: Build converter (from/into) form cgmath to bytemuck
    view_proj: [[f32; 4]; 4],
}

impl Uniforms {
    fn new() -> Self {
        use cgmath::SquareMatrix;
        return Self {
            view_proj: cgmath::Matrix4::identity().into()
        };
    }

    fn update_view_proj(&mut self, camera: &Camera) {
        self.view_proj = camera.build_view_projection_matrix().into();
    }
}

//=============================================================================

const NUM_INSTANCES_PER_ROW: u32 = 10;

struct Instance {
    position: cgmath::Vector3<f32>,
    rotation: cgmath::Quaternion<f32>, // TODO: Review quaternions
}

impl Instance {
    fn to_raw(&self) -> InstanceRaw {
        return InstanceRaw {
            model: (cgmath::Matrix4::from_translation(self.position) * cgmath::Matrix4::from(self.rotation)).into(),
        };
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct InstanceRaw {
    model: [[f32; 4]; 4],
}

impl InstanceRaw {
    fn desc<'a>() -> wgpu::VertexBufferDescriptor<'a> {
        use std::mem;

        let attrib = &[
            wgpu::VertexAttributeDescriptor {
                offset: 0,
                // While our vertex shader only uses locations 0, and 1 now, in later tutorials we'll
                // be using 2, 3, and 4, for Vertex. We'll start at slot 5 not conflict with them later
                shader_location: 5,
                format: wgpu::VertexFormat::Float4,
            },
            // A mat4 takes up 4 vertex slots as it is technically 4 vec4s. We need to define a slot
            // for each vec4. We don't have to do this in code though.
            wgpu::VertexAttributeDescriptor {
                offset: mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
                shader_location: 6,
                format: wgpu::VertexFormat::Float4,
            },
            wgpu::VertexAttributeDescriptor {
                offset: mem::size_of::<[f32; 8]>() as wgpu::BufferAddress,
                shader_location: 7,
                format: wgpu::VertexFormat::Float4,
            },
            wgpu::VertexAttributeDescriptor {
                offset: mem::size_of::<[f32; 12]>() as wgpu::BufferAddress,
                shader_location: 8,
                format: wgpu::VertexFormat::Float4,
            },
        ];

        return wgpu::VertexBufferDescriptor {
            stride: mem::size_of::<InstanceRaw>() as wgpu::BufferAddress,
            // We need to switch from using a step mode of Vertex to Instance
            // This means that our shaders will only change to use the next
            // instance when the shader starts processing a new instance
            step_mode: wgpu::InputStepMode::Instance,
            attributes: attrib,
        };
    }
}

//=============================================================================

fn create_render_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    color_format: wgpu::TextureFormat,
    depth_format: Option<wgpu::TextureFormat>,
    vertex_descs: &[wgpu::VertexBufferDescriptor],
    vs_src: wgpu::ShaderModuleSource,
    fs_src: wgpu::ShaderModuleSource,
) -> wgpu::RenderPipeline {
    // Create ShaderModules
    let vs_module = device.create_shader_module(vs_src);
    let fs_module = device.create_shader_module(fs_src);

    // Create Render Pipeline
    let render_pipeline_desc = wgpu::RenderPipelineDescriptor {
        label: Some("Render Pipeline"),
        layout: Some(&layout),
        vertex_stage: wgpu::ProgrammableStageDescriptor {
            module: &vs_module,
            entry_point: "main",
        },
        fragment_stage: Some(wgpu::ProgrammableStageDescriptor {
            module: &fs_module,
            entry_point: "main",
        }),
        rasterization_state: Some(wgpu::RasterizationStateDescriptor{
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: wgpu::CullMode::Back,
            depth_bias: 0,
            depth_bias_slope_scale: 0.0,
            depth_bias_clamp: 0.0,
            clamp_depth: false,
        }),
        color_states: &[wgpu::ColorStateDescriptor{ // Define how colors are stored and processed
            format: color_format,
            color_blend: wgpu::BlendDescriptor::REPLACE,
            alpha_blend: wgpu::BlendDescriptor::REPLACE,
            write_mask: wgpu::ColorWrite::ALL,
        }],
        primitive_topology: wgpu::PrimitiveTopology::TriangleList,
        depth_stencil_state: depth_format.map(|format| wgpu::DepthStencilStateDescriptor {
            format,
            depth_write_enabled: true,
            // When to discard a new pixel. Drawn front to back. Depth should be less (closer
            // to camera) to discard the previous pixel on the texture
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilStateDescriptor::default(),
        }),
        vertex_state: wgpu::VertexStateDescriptor {
            index_format: wgpu::IndexFormat::Uint32,
            vertex_buffers: vertex_descs,
        },
        sample_count: 1,
        sample_mask: !0, // Use all samples
        alpha_to_coverage_enabled: false,
    };
    let render_pipeline = device.create_render_pipeline(&render_pipeline_desc);
    return render_pipeline;
}

struct State {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    swap_chain_desc: wgpu::SwapChainDescriptor,
    swap_chain: wgpu::SwapChain,
    render_pipeline: wgpu::RenderPipeline,
    obj_model: model::Model,
    camera: Camera,
    camera_controller: CameraController,
    uniforms: Uniforms,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    instances: Vec<Instance>,
    light: Light,
    light_buffer: wgpu::Buffer,
    light_bind_group: wgpu::BindGroup,
    light_render_pipeline: wgpu::RenderPipeline,
    #[allow(dead_code)]
    instance_buffer: wgpu::Buffer,
    depth_texture: texture::Texture,
    size: PhysicalSize<u32>, // INFO: PhysicalSize takes into account device's scale factor
}

impl State {
    async fn new(window: &Window) -> Self {
        // The instance is a handle to our GPU
        // BackendBit::PRIMARY => Vulkan + Metal + DX12 + Browser WebGPU
        let instance = wgpu::Instance::new(wgpu::BackendBit::PRIMARY);
        let surface = unsafe { instance.create_surface(window) };

        // Create Adapter
        let adapter_options = &wgpu::RequestAdapterOptions {
            // Default gets LowP on battery and HighP when on mains
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
        };
        // The adapter identifies both an instance of a physical hardware accelerator (CPU, GPU),
        // and an instance of a browser's implementation of WebGPU on top of the accelerator
        let adapter = instance.request_adapter(adapter_options).await.unwrap();

        // Create Device and Queue
        let desc = &wgpu::DeviceDescriptor {
            features: wgpu::Features::empty(),
            limits: wgpu::Limits::default(),
            shader_validation: true,
        };
        let (device, queue) = adapter.request_device(desc, None).await.unwrap();

        // Create SwapChain
        let size = window.inner_size(); // INFO: Has into account the scale factor
        let swap_chain_desc = wgpu::SwapChainDescriptor {
            usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
            format: wgpu::TextureFormat::Bgra8UnormSrgb, // TODO: Should be swap_chain_get_current_texture_view but not available atm
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
        };
        let swap_chain = device.create_swap_chain(&surface, &swap_chain_desc);

        // Describe a set of resources and how are they accessed by a Shader
        let texture_bind_group_layout_desc = wgpu::BindGroupLayoutDescriptor {
            label: Some("texture_bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStage::FRAGMENT, // Bitwise comparison
                    ty: wgpu::BindingType::SampledTexture {
                        multisampled: false,
                        dimension: wgpu::TextureViewDimension::D2,
                        component_type: wgpu::TextureComponentType::Uint,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::Sampler {
                        comparison: false,
                    },
                    count: None,
                },
            ],
        };
        let texture_bind_group_layout = device.create_bind_group_layout(&texture_bind_group_layout_desc);

        let depth_texture = texture::Texture::create_depth_texture(&device, &swap_chain_desc, "depth_texture");

        let camera = Camera {
            eye: (0.0, 1.0, 2.0).into(),
            target: (0.0, 0.0, 0.0).into(),
            up: cgmath::Vector3::unit_y(),
            aspect: swap_chain_desc.width as f32 / swap_chain_desc.height as f32,
            fovy: 45.0,
            znear: 0.1,
            zfar: 100.0,
        };

        let camera_controller = CameraController::new(0.2);

        // Create Uniform Buffers
        let mut uniforms = Uniforms::new();
        uniforms.update_view_proj(&camera);

        let uniforms_array = &[uniforms];
        let uniform_buffer_desc = wgpu::util::BufferInitDescriptor {
            label: Some("Uniform Buffer"),
            contents: bytemuck::cast_slice(uniforms_array),
            usage: wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
        };

        let uniform_buffer = device.create_buffer_init(&uniform_buffer_desc);

        // Create Uniform Bind Group
        let uniform_bind_group_layout_entry = wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStage::VERTEX,
            ty: wgpu::BindingType::UniformBuffer { dynamic: false, min_binding_size: None, },
            count: None,
        };
        let uniform_bind_group_layout_desc = wgpu::BindGroupLayoutDescriptor{
            label: Some("Uniform Bind Group Layout"),
            entries: &[uniform_bind_group_layout_entry]
        }; 
        let uniform_bind_group_layout = device.create_bind_group_layout(&uniform_bind_group_layout_desc);
        let uniform_bind_group_desc = wgpu::BindGroupDescriptor {
            label: Some("Uniform Bind Group"),
            layout: &uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(uniform_buffer.slice(..)),
            }],
        };
        let uniform_bind_group = device.create_bind_group(&uniform_bind_group_desc);

        // Load Model
        let res_dir = std::path::Path::new(env!("OUT_DIR")).join("res");
        let obj_model = model::Model::load(
            &device,
            &queue,
            &texture_bind_group_layout,
            res_dir.join("cube.obj"),
        ).unwrap();

        // Create Instances
        const SPACE_BETWEEN: f32 = 3.0;
        let instances = (0..NUM_INSTANCES_PER_ROW).flat_map(|z| {
            (0..NUM_INSTANCES_PER_ROW).map(move |x| {
                let x = SPACE_BETWEEN * (x as f32 - NUM_INSTANCES_PER_ROW as f32 / 2.0);
                let z = SPACE_BETWEEN * (z as f32 - NUM_INSTANCES_PER_ROW as f32 / 2.0);

                let position = cgmath::Vector3 { x, y: 0.0, z };

                let rotation = if position.is_zero() {
                    cgmath::Quaternion::from_axis_angle(
                        cgmath::Vector3::unit_z(),
                        cgmath::Deg(0.0),
                    )
                } else {
                    cgmath::Quaternion::from_axis_angle(
                        position.clone().normalize(),
                        cgmath::Deg(45.0),
                    )
                };

                Instance { position, rotation }
            })
        })
        .collect::<Vec<_>>();

        let instance_data = instances.iter().map(Instance::to_raw).collect::<Vec<InstanceRaw>>();
        let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
            label: Some("Instance Buffer"),
            contents: bytemuck::cast_slice(&instance_data),
            usage: wgpu::BufferUsage::VERTEX,
        });

        // Create Light
        // TODO: Change wgpu tutorial from .into() to [f32; 3] 
        let light = Light {
            position: [2.0, 2.0, 2.0],
            _padding: 0,
            color: [1.0, 1.0, 1.0],
        };
        let lights_array = &[light];
        let light_buffer_init_desc = wgpu::util::BufferInitDescriptor {
            label: Some("light_buffer_init_desc"),
            contents: bytemuck::cast_slice(lights_array),
            usage: wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
        };
        let light_buffer = device.create_buffer_init(&light_buffer_init_desc);

        let light_bind_group_layout_desc = wgpu::BindGroupLayoutDescriptor {
            label: Some("light_bind_group_layout_desc"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStage::VERTEX | wgpu::ShaderStage::FRAGMENT,
                ty: wgpu::BindingType::UniformBuffer { dynamic: false, min_binding_size: None, },
                count: None,
            }],
        };
        let light_bind_group_layout = device.create_bind_group_layout(&light_bind_group_layout_desc);

        let light_bind_group_desc = wgpu::BindGroupDescriptor {
            label: Some("light_bind_group"),
            layout: &light_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(light_buffer.slice(..)),
            }],
        };
        let light_bind_group = device.create_bind_group(&light_bind_group_desc);

        // Create Pipeline Layout
        let pipeline_layout_desc = wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[
                &texture_bind_group_layout,
                &uniform_bind_group_layout,
                &light_bind_group_layout,
            ],
            push_constant_ranges: &[],
        };
        let render_pipeline_layout = device.create_pipeline_layout(&pipeline_layout_desc);

        let render_pipeline = create_render_pipeline(
            &device,
            &render_pipeline_layout,
            swap_chain_desc.format,
            Some(texture::Texture::DEPTH_FORMAT),
            &[model::ModelVertex::desc(), InstanceRaw::desc()],
            wgpu::include_spirv!("../shaders/shader.vert.spv"),
            wgpu::include_spirv!("../shaders/shader.frag.spv"),
        );

        let light_render_pipeline = {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("light_render_pipeline_layout"),
                bind_group_layouts: &[
                    &uniform_bind_group_layout,
                    &light_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });

            let vs_src = wgpu::include_spirv!("../shaders/light.vert.spv");
            let fs_src = wgpu::include_spirv!("../shaders/light.frag.spv");

            create_render_pipeline(
                &device, 
                &layout, 
                swap_chain_desc.format, 
                Some(texture::Texture::DEPTH_FORMAT), 
                &[model::ModelVertex::desc()], 
                vs_src, 
                fs_src,
            )
        };
 
        return State {
            surface,
            device,
            queue,
            swap_chain_desc,
            swap_chain,
            render_pipeline,
            obj_model,
            camera,
            camera_controller,
            uniforms,
            uniform_buffer,
            uniform_bind_group,
            instances,
            light,
            light_buffer,
            light_bind_group,
            light_render_pipeline,
            instance_buffer,
            depth_texture,
            size,
        };
    }

    fn resize(&mut self, new_size: PhysicalSize<u32>) {
        self.size = new_size;
        self.swap_chain_desc.width = new_size.width;
        self.swap_chain_desc.height = new_size.height;
        self.swap_chain = self.device.create_swap_chain(&self.surface, &self.swap_chain_desc);
        self.depth_texture = texture::Texture::create_depth_texture(&self.device, &self.swap_chain_desc, "depth_texture");
    }

    // Returns a bool to indicate whether an event has been fully processed. If `true` the main
    // loop won't process the event any further
    fn input(&mut self, event: &WindowEvent) -> bool {
        return self.camera_controller.process_events(event);
    }

    fn update(&mut self) {
        self.camera_controller.update_camera(&mut self.camera);
        self.uniforms.update_view_proj(&self.camera);
        self.queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[self.uniforms]));

        // Update light's position
        let old_position: cgmath::Vector3<f32> = self.light.position.into();
        let new_position = cgmath::Quaternion::from_axis_angle((0.0, 1.0, 0.0).into(), cgmath::Deg(1.0)) * old_position;
        self.light.position = new_position.into();
        self.queue.write_buffer(&self.light_buffer, 0, bytemuck::cast_slice(&[self.light]));
    }

    fn render(&mut self) -> Result<(), wgpu::SwapChainError> {
        // Get next frame
        let frame = self.swap_chain.get_current_frame()?.output;

        // Create command encoder
        let command_encoder_desc = wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        };
        let mut encoder = self.device.create_command_encoder(&command_encoder_desc);

        {
            // Create Render Pass
            let clear_color = wgpu::Color { r: 0.1, g: 0.1, b: 0.1, a: 1.0, };
            let render_pass_desc = wgpu::RenderPassDescriptor {
                // Color Attachments
                color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
                    attachment: &frame.view, // Current frame texture view
                    resolve_target: None, // Only used if multisampling is enabled
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(clear_color),
                        store: true,
                    },
                }],
                // Depth Stencil Attachments
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachmentDescriptor {
                    attachment: &self.depth_texture.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0), // Clear before use
                        store: true, // Render Pass will write here: true
                    }),
                    stencil_ops: None,
                }),
            };
            let mut render_pass = encoder.begin_render_pass(&render_pass_desc);
            render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
            render_pass.set_pipeline(&self.light_render_pipeline);
            // TODO: Refactor draw_light_model to light_model
            render_pass.draw_light_model(
                &self.obj_model,
                &self.uniform_bind_group,
                &self.light_bind_group,
            );
            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.draw_model_instanced(
                &self.obj_model,
                0..self.instances.len() as u32,
                &self.uniform_bind_group,
                &self.light_bind_group,
            );
        }

        self.queue.submit(std::iter::once(encoder.finish()));

        return Ok(());
    }
}

fn handle_keyboard_input(_state: &mut State, input: KeyboardInput, control_flow: &mut ControlFlow) {
    if input.state == ElementState::Pressed {
        match input.virtual_keycode {
            Some(VirtualKeyCode::Escape) => {
                *control_flow = ControlFlow::Exit;
            }
            _ => {}
        }
    }
}

fn handle_window_events(state: &mut State, event: WindowEvent, control_flow: &mut ControlFlow) {
    match event {
        WindowEvent::KeyboardInput { input, ../*device_id, is_synthetic*/ } => {
            handle_keyboard_input(state, input, control_flow);
        },
        WindowEvent::Resized(physical_size) => {
            state.resize(physical_size)
        },
        WindowEvent::ScaleFactorChanged {new_inner_size, ../*scale_factor*/ } => {
            state.resize(*new_inner_size)
        },
        WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
        _ => {}
    }
}

fn handle_redraw_requested(state: &mut State, control_flow: &mut ControlFlow) {
    state.update();
    match state.render() {
        Err(wgpu::SwapChainError::Lost) => state.resize(state.size),
        Err(wgpu::SwapChainError::OutOfMemory) => *control_flow = ControlFlow::Exit,
        Err(e) => eprintln!("{:?}", e),
        Ok(_) => {}
    }
}

fn main() {
    env_logger::init(); // INFO: error!, warn!, info!, debug! and trace!
    let event_loop = EventLoop::new();

    let window = WindowBuilder::new()
        .with_title("WGPU Renderer")
        .with_resizable(false)
        .build(&event_loop)
        .unwrap();

    // INFO: This is just for debugging purposes
    window.set_outer_position(winit::dpi::PhysicalPosition::new(2561.0, 1.0));

    let mut state = block_on(State::new(&window));

    // INFO: move -> moves any variables you reference which are outside the scope of the closure into the closure's object.
    event_loop.run(move |event, _event_loop_window_target, control_flow| {
        match event {
            Event::WindowEvent { event, window_id } => {
                if window_id == window.id() && !state.input(&event) {
                    handle_window_events(&mut state, event, control_flow);
                }
            }
            Event::RedrawRequested(_window_id) => {
                handle_redraw_requested(&mut state, control_flow);
            }
            Event::MainEventsCleared => {
                // Emitted when all of the event loop's input events have been processed and redraw processing
                // is about to begin.
                window.request_redraw();
            }
            _ => {}
        };
    });
}
