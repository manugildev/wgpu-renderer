mod texture;

use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

use futures::executor::block_on;
use wgpu::util::DeviceExt;
use texture::Texture;

//=============================================================================

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    color: [f32; 3],
    tex_coords: [f32; 2],
}
impl Vertex {
    fn desc<'a>() -> wgpu::VertexBufferDescriptor<'a> {
        let attrib_descs = &[
            // Position
            wgpu::VertexAttributeDescriptor {
                offset: 0,
                shader_location: 0,
                format: wgpu::VertexFormat::Float3,
            },
            // Color
            wgpu::VertexAttributeDescriptor {
                offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                shader_location: 1,
                format: wgpu::VertexFormat::Float3,
            },
            // Tex Coords
            wgpu::VertexAttributeDescriptor {
                offset: std::mem::size_of::<[f32; 6]>() as wgpu::BufferAddress,
                shader_location: 2,
                format: wgpu::VertexFormat::Float2,
            }
        ];

        return wgpu::VertexBufferDescriptor {
            stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::InputStepMode::Vertex,
            attributes: attrib_descs
        };
    }
}

const VERTICES: &[Vertex] = &[
    Vertex { position: [-0.5,  0.5, 0.0], color: [1.0, 0.0, 0.0], tex_coords: [0.0, 0.0], }, // A
    Vertex { position: [ 0.5,  0.5, 0.0], color: [0.0, 1.0, 0.0], tex_coords: [1.0, 0.0], }, // C
    Vertex { position: [-0.5, -0.5, 0.0], color: [0.0, 0.0, 1.0], tex_coords: [0.0, 1.0], }, // B
    Vertex { position: [ 0.5, -0.5, 0.0], color: [1.0, 0.0, 1.0], tex_coords: [1.0, 1.0], }, // D
];

const INDICES: &[u16] = &[
    2, 1, 0,
    3, 1, 2,
];

//=============================================================================

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    // We can't use cgmath with bytemuck directly so we'll have
    // to convert the Matrix4 into a 4x4 f32 array
    // TODO: Build converter form cgmath to bytemuck
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

struct Camera {
    eye: cgmath::Point3<f32>,
    target: cgmath::Point3<f32>,
    up: cgmath::Vector3<f32>,
    aspect: f32,
    fovy: f32,
    znear: f32,
    zfar: f32,
}

impl Camera {
    fn build_view_projection_matrix(&self) -> cgmath::Matrix4<f32> {
        let view = cgmath::Matrix4::look_at(self.eye, self.target, self.up);
        let proj = cgmath::perspective(cgmath::Deg(self.fovy), self.aspect, self.znear, self.zfar);

        return OPENGL_TO_WGPU_MATRIX * proj * view;
    }
}

#[rustfmt::skip]
pub const OPENGL_TO_WGPU_MATRIX: cgmath::Matrix4<f32> = cgmath::Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.0,
    0.0, 0.0, 0.5, 1.0,
);

//=============================================================================

struct State {
    surface: wgpu::Surface,
    // Logical Device
    device: wgpu::Device,
    queue: wgpu::Queue,
    swap_chain_desc: wgpu::SwapChainDescriptor,
    swap_chain: wgpu::SwapChain,
    size: winit::dpi::PhysicalSize<u32>, // INFO: PhysicalSize takes into account device's scale factor
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    num_vertices: u32,
    index_buffer: wgpu::Buffer,
    num_indices: u32,
    diffuse_bind_group: wgpu::BindGroup,
    diffuse_texture: Texture,
    camera: Camera,
    uniforms: Uniforms,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
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
            power_preference: wgpu::PowerPreference::default(),
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

        let diffuse_bytes = include_bytes!("happy-tree.png");
        let diffuse_texture = Texture::from_bytes(&device, &queue, diffuse_bytes, "happy-tree-texture").unwrap();

        // Describe a set of resources and how are they accessed by a Shader
        let bind_group_layout_desc = wgpu::BindGroupLayoutDescriptor {
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
        let texture_bind_group_layout = device.create_bind_group_layout(&bind_group_layout_desc);
        let bind_group_desc = wgpu::BindGroupDescriptor {
            label: Some("difuse_bind_group"),
            layout: &texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&diffuse_texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&diffuse_texture.sampler),
                },
            ],
        };
        let diffuse_bind_group = device.create_bind_group(&bind_group_desc);

        let camera = Camera {
            eye: (0.0, 1.0, 2.0).into(),
            target: (0.0, 0.0, 0.0).into(),
            up: cgmath::Vector3::unit_y(),
            aspect: swap_chain_desc.width as f32 / swap_chain_desc.height as f32,
            fovy: 45.0,
            znear: 0.1,
            zfar: 100.0,
        };

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

        // Create ShaderModules
        let vs_module = device.create_shader_module(wgpu::include_spirv!("../shaders/shader.vert.spv"));
        let fs_module = device.create_shader_module(wgpu::include_spirv!("../shaders/shader.frag.spv"));

        // Create Vertex Buffer
        let buffer_init_desc = wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(VERTICES), // INFO: cast to &[u8]
            usage: wgpu::BufferUsage::VERTEX,
        };

        let vertex_buffer = device.create_buffer_init(&buffer_init_desc);
        let num_vertices = VERTICES.len() as u32;

        // Create Vertex Buffer
        let buffer_init_desc = wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(INDICES), // INFO: cast to &[u8]
            usage: wgpu::BufferUsage::INDEX,
        };

        let index_buffer = device.create_buffer_init(&buffer_init_desc);
        let num_indices = INDICES.len() as u32;

        // Create Pipeline Layout
        let pipeline_layout_desc = wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[
                &texture_bind_group_layout,
                &uniform_bind_group_layout,
            ],
            push_constant_ranges: &[],
        };
        let render_pipeline_layout = device.create_pipeline_layout(&pipeline_layout_desc);

        // Create Render Pipeline
        let render_pipeline_desc = wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
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
                format: swap_chain_desc.format,
                color_blend: wgpu::BlendDescriptor::REPLACE,
                alpha_blend: wgpu::BlendDescriptor::REPLACE,
                write_mask: wgpu::ColorWrite::ALL,
            }],
            primitive_topology: wgpu::PrimitiveTopology::TriangleList,
            depth_stencil_state: None,
            vertex_state: wgpu::VertexStateDescriptor {
                index_format: wgpu::IndexFormat::Uint16,
                vertex_buffers: &[Vertex::desc()],
            },
            sample_count: 1,
            sample_mask: !0, // Use all samples
            alpha_to_coverage_enabled: false,
        };
        let render_pipeline = device.create_render_pipeline(&render_pipeline_desc);

        return State {
            surface,
            device,
            queue,
            swap_chain_desc,
            swap_chain,
            size,
            render_pipeline,
            vertex_buffer,
            num_vertices,
            index_buffer,
            num_indices,
            diffuse_bind_group,
            diffuse_texture,
            camera,
            uniforms,
            uniform_buffer,
            uniform_bind_group,
        };
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        self.size = new_size;
        self.swap_chain_desc.width = new_size.width;
        self.swap_chain_desc.height = new_size.height;
        self.swap_chain = self.device.create_swap_chain(&self.surface, &self.swap_chain_desc);
    }

    // Returns a bool to indicate whether an event has been fully processed. If `true` the main
    // loop won't process the event any further
    fn input(&mut self, event: &WindowEvent) -> bool {
        false
    }

    fn update(&mut self) {}

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
                    resolve_target: None,    // Only used if multisampling is enabled
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(clear_color),
                        store: true,
                    },
                }],
                // Depth Stencil Attachments
                depth_stencil_attachment: None,
            };
            let mut render_pass = encoder.begin_render_pass(&render_pass_desc);
            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.diffuse_bind_group, &[]);
            render_pass.set_bind_group(1, &self.uniform_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_index_buffer(self.index_buffer.slice(..));
            render_pass.draw_indexed(0..self.num_indices, 0, 0..1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));

        return Ok(());
    }
}

fn handle_keyboard_input(state: &mut State, input: KeyboardInput, control_flow: &mut ControlFlow) {
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

    let window = WindowBuilder::new().with_title("WGPU Renderer").build(&event_loop).unwrap();

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
